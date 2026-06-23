#!/usr/bin/env python3
"""Run DeltaKV on the KVPress leaderboard evaluation format.

This runner intentionally does not implement DeltaKV as a KVPress ``BasePress``.
KVPress presses prune or merge the Hugging Face DynamicCache in-place after
context prefill, while DeltaKV owns a custom cache representation with centers,
latent residuals, and optional quantized full-layer storage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from deltakv.get_chat_api import get_generate_api


@dataclass
class RunConfig:
    dataset: str
    data_dir: str
    model: str
    device: str
    press_name: str
    compression_ratio: float
    key_channel_compression_ratio: None = None
    threshold: None = None
    fraction: float = 1.0
    max_new_tokens: int | None = None
    max_context_length: int | None = None
    query_aware: bool = False
    needle_depth: None = None
    compression_interval: None = None
    target_size: None = None
    hidden_states_buffer_size: None = None
    output_dir: str = ""
    log_level: str = "INFO"
    model_kwargs: dict[str, Any] | None = None
    press_init_command: str = ""
    seed: int = 42
    fp8: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvpress-repo", default="/tmp/kvpress", help="Local NVIDIA/kvpress checkout.")
    parser.add_argument("--dataset", default="ruler")
    parser.add_argument("--data-dir", default="4096")
    parser.add_argument("--hf-dataset", default="simonjegou/ruler")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--hyper-params", required=True)
    parser.add_argument("--deltakv-checkpoint-path", default=None)
    parser.add_argument("--sparse-method", default=None)
    parser.add_argument("--backend", default="hf", choices=["hf", "sparsevllm"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="deltakv")
    parser.add_argument("--leaderboard-compression-ratio", type=float, required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit total rows after optional sampling.")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--query-aware", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def _load_hyper_params(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"DeltaKV hyper-params file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metric(kvpress_repo: str):
    repo = Path(kvpress_repo)
    if not (repo / "evaluation" / "benchmarks" / "ruler" / "calculate_metrics.py").exists():
        raise FileNotFoundError(f"Could not find KVPress ruler metric under {repo}")
    sys.path.insert(0, str(repo / "evaluation"))
    from benchmarks.ruler.calculate_metrics import calculate_metrics

    return calculate_metrics


def _prepare_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = load_dataset(args.hf_dataset, args.data_dir, split="test").to_pandas()
    if args.fraction < 1.0:
        df = df.sample(frac=args.fraction, random_state=args.seed)
    if args.limit_per_task is not None:
        df = df.groupby("task", group_keys=False).head(args.limit_per_task)
    if args.limit is not None:
        df = df.head(args.limit)
    if df.empty:
        raise ValueError("No evaluation rows selected.")
    return df.reset_index(drop=True)


def _split_context_and_question(tokenizer, context: str, question: str, answer_prefix: str, query_aware: bool) -> tuple[str, str]:
    if query_aware:
        context = context + question
        question = ""

    if tokenizer.chat_template is None:
        bos_token = getattr(tokenizer, "bos_token", "") or ""
        context_text = bos_token + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        context_text, question_suffix = templated.split(separator)
    return context_text, question + question_suffix + answer_prefix


@torch.inference_mode()
def _generate_one(model, tokenizer, context_text: str, question_text: str, max_new_tokens: int) -> str:
    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    if context_ids.shape[0] != 1:
        raise ValueError(f"Expected batch_size=1 context ids, got shape {tuple(context_ids.shape)}")

    outputs = model(input_ids=context_ids, use_cache=True, logits_to_keep=1)
    past_key_values = outputs.past_key_values

    question_ids = tokenizer.encode(question_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    if question_ids.numel() == 0:
        question_ids = torch.empty((1, 0), dtype=context_ids.dtype, device=model.device)

    generated: list[torch.Tensor] = []
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    eos_ids = [int(x) for x in eos_ids if x is not None]

    if question_ids.shape[1] > 0:
        outputs = model(input_ids=question_ids, past_key_values=past_key_values, use_cache=True, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[0, -1].argmax()
        generated.append(next_token)
    else:
        next_token = outputs.logits[0, -1].argmax()
        generated.append(next_token)

    for _ in range(max_new_tokens - 1):
        if eos_ids and int(generated[-1].item()) in eos_ids:
            break
        outputs = model(
            input_ids=generated[-1].view(1, 1),
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = outputs.past_key_values
        generated.append(outputs.logits[0, -1].argmax())

    if eos_ids and generated and int(generated[-1].item()) in eos_ids:
        generated = generated[:-1]
    if not generated:
        return ""
    return tokenizer.decode(torch.stack(generated), skip_special_tokens=True)


def _results_dir(args: argparse.Namespace) -> Path:
    model_name = args.model_path.rstrip("/").split("/")[-1]
    if args.model_path.startswith(("meta-llama/", "Qwen/")):
        model_component = args.model_path.replace("/", "--")
    else:
        model_component = model_name
    parts = [
        args.dataset,
        args.data_dir,
        model_component,
        args.method_name,
        f"{args.leaderboard_compression_ratio:.2f}",
    ]
    if args.query_aware:
        parts.append("query_aware")
    if args.fraction < 1.0:
        parts.append(f"fraction{args.fraction:.3f}")
    if args.limit is not None:
        parts.append(f"limit{args.limit}")
    if args.limit_per_task is not None:
        parts.append(f"limit_per_task{args.limit_per_task}")
    out = Path(args.output_dir) / "__".join(parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = _parse_args()
    if not (0.0 <= args.leaderboard_compression_ratio <= 1.0):
        raise ValueError("--leaderboard-compression-ratio must be in [0, 1].")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)

    hyper_params = _load_hyper_params(args.hyper_params)
    if args.deltakv_checkpoint_path is not None:
        hyper_params["deltakv_checkpoint_path"] = args.deltakv_checkpoint_path

    _, model = get_generate_api(
        args.model_path,
        hyper_params,
        tokenizer_path=args.tokenizer_path,
        sparse_method=args.sparse_method,
        cuda_device=args.device,
        backend=args.backend,
        return_model=True,
    )
    model.eval()
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    df = _prepare_dataset(args)
    calculate_metrics = _load_metric(args.kvpress_repo)
    result_dir = _results_dir(args)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DeltaKV KVPress eval"):
        max_new_tokens = args.max_new_tokens or int(row["max_new_tokens"])
        try:
            context_text, question_text = _split_context_and_question(
                tokenizer,
                str(row["context"]),
                str(row["question"]),
                str(row["answer_prefix"]),
                args.query_aware,
            )
            if args.max_context_length is not None:
                context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False)
                if context_ids.shape[1] > args.max_context_length:
                    context_ids = context_ids[:, : args.max_context_length]
                    context_text = tokenizer.decode(context_ids[0], skip_special_tokens=False)
            predicted_answer = _generate_one(model, tokenizer, context_text, question_text, max_new_tokens)
            status = "success"
            error = ""
        except Exception as exc:
            predicted_answer = ""
            status = "model_failed"
            error = repr(exc)
            raise
        record = {k: row[k] for k in row.index if k != "context"}
        record.update(
            {
                "predicted_answer": predicted_answer,
                "compression_ratio": args.leaderboard_compression_ratio,
                "status": status,
                "error": error,
            }
        )
        rows.append(record)
        pd.DataFrame(rows).to_csv(result_dir / "predictions.csv", index=False)

    pred_df = pd.DataFrame(rows)
    if not (pred_df["status"] == "success").all():
        raise RuntimeError(f"Non-success rows found:\n{pred_df[pred_df['status'] != 'success'][['task', 'status', 'error']]}")

    metrics = calculate_metrics(pred_df.copy())
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    run_config = RunConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        model=args.model_path if args.model_path.startswith(("meta-llama/", "Qwen/")) else Path(args.model_path).name,
        device=args.device,
        press_name=args.method_name,
        compression_ratio=args.leaderboard_compression_ratio,
        fraction=args.fraction,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        query_aware=args.query_aware,
        output_dir=args.output_dir,
        model_kwargs={"attn_implementation": args.attn_implementation, "dtype": args.dtype},
        press_init_command=(
            "DeltaKV native HF cache runner "
            f"(hyper_params={args.hyper_params}, backend={args.backend}, sparse_method={args.sparse_method})"
        ),
        seed=args.seed,
    )
    (result_dir / "config.yaml").write_text(yaml.safe_dump(asdict(run_config), sort_keys=False), encoding="utf-8")
    (result_dir / "run_info.json").write_text(
        json.dumps(
            {
                "hyper_params": hyper_params,
                "model_path": args.model_path,
                "tokenizer_path": tokenizer_path,
                "kvpress_repo": args.kvpress_repo,
                "row_count": len(pred_df),
                "status_counts": pred_df["status"].value_counts().to_dict(),
                "metric_average": round(sum(v["string_match"] for v in metrics.values()) / len(metrics), 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved KVPress-compatible DeltaKV result to {result_dir}")


if __name__ == "__main__":
    main()
