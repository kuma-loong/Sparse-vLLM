from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT / "src"))

from deltakv.get_chat_api import get_generate_api
from benchmark.common.ledger import git_metadata, selected_env_snapshot
from benchmark.common.paths import default_output_path


DEFAULT_PROMPTS = [
    "Answer with exactly one word: ready.",
    "Return the integer result of 17 + 25.",
    "Summarize in one short sentence: Sparse attention should preserve important context while reducing compute.",
]


def _load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    value = value.strip()
    if value.startswith("@"):
        value = Path(value[1:]).expanduser().read_text(encoding="utf-8")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must decode to an object.")
    return parsed


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return payload
    if isinstance(payload, list) and all(isinstance(item, dict) and "prompt" in item for item in payload):
        return [str(item["prompt"]) for item in payload]
    raise ValueError("--prompts_path must contain a JSON list of strings or objects with a 'prompt' field.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-vLLM sanity benchmark for quick pipeline checks.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--sparse_method", default="vanilla")
    parser.add_argument("--backend", default="hf", choices=["hf", "sparsevllm"])
    parser.add_argument("--deltakv_checkpoint_path", default=None)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--hyper_param", default=None, help="Inline JSON or @path JSON object.")
    parser.add_argument("--prompts_path", default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to benchmark/results/sanity/<method>_<time>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _load_prompts(args.prompts_path)
    infer_config: dict[str, Any] = {"max_model_len": args.max_model_len}
    infer_config.update(_load_json_arg(args.hyper_param))

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or default_output_path("sanity", f"{args.sparse_method}_{time_tag}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        **git_metadata(Path(__file__).resolve().parents[1]),
        "command": "python " + " ".join(sys.argv),
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "sparse_method": args.sparse_method,
        "backend": args.backend,
        "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
        "infer_config": infer_config,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_prompts": len(prompts),
        "env": selected_env_snapshot(),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    generate = get_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=args.deltakv_checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        sparse_method=args.sparse_method,
        cuda_device=args.cuda_device,
        backend=args.backend,
    )

    raw_records = []
    per_sample = []
    success = 0
    for idx, prompt in enumerate(prompts):
        start = time.perf_counter()
        status = "success"
        output = ""
        error_message = ""
        try:
            output = generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            if isinstance(output, list):
                output = output[0] if output else ""
            if not str(output).strip():
                status = "model_failed"
                error_message = "empty output"
        except Exception as exc:
            status = "model_failed"
            error_message = repr(exc)
        elapsed_s = time.perf_counter() - start
        if status == "success":
            success += 1

        raw_records.append({"sample_id": idx, "prompt": prompt, "raw_output": output, "status": status})
        per_sample.append(
            {
                "sample_id": idx,
                "task": "sanity",
                "status": status,
                "prompt": prompt,
                "output": output,
                "elapsed_s": elapsed_s,
                "error_message": error_message,
            }
        )

    for name, records in [
        ("raw_outputs.jsonl", raw_records),
        ("parsed_outputs.jsonl", per_sample),
        ("per_sample_results.jsonl", per_sample),
    ]:
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    aggregate = {
        "benchmark": "sanity",
        "status": "success" if success == len(prompts) else "model_failed",
        "num_samples": len(prompts),
        "success": success,
        "model_failed": len(prompts) - success,
        "success_rate": success / max(len(prompts), 1),
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **aggregate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
