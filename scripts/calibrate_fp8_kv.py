"""Measure static per-layer FP8 KV scales with the dense SparseEngine model path.

Input is a JSON file with a ``samples`` array or a JSONL file. Every sample
must contain ``prompt_token_ids`` or ``prompt``. The caller owns the input
dataset and keeps it outside Git when it is private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

def _prompts(path: Path) -> list[list[int] | str]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        rows = loaded["samples"] if isinstance(loaded, dict) else loaded
    if not isinstance(rows, list) or not rows:
        raise ValueError("Calibration input must contain a nonempty sample list.")
    prompts = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Calibration sample {index} must be an object.")
        prompt = row.get("prompt_token_ids", row.get("prompt"))
        if not isinstance(prompt, (str, list)) or not prompt:
            raise ValueError(f"Calibration sample {index} has no prompt or token IDs.")
        if isinstance(prompt, list) and (not prompt or any(type(token) is not int or token < 0 for token in prompt)):
            raise ValueError(f"Calibration sample {index} has invalid token IDs.")
        prompts.append(prompt)
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Prepared JSON or JSONL calibration samples")
    parser.add_argument("--output", required=True, help="Calibrated FP8 KV scale JSON")
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses all samples")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--safety-margin", type=float, default=1.05)
    parser.add_argument("--runtime-config", help="Additional SparseEngine config JSON file")
    args = parser.parse_args()
    if min(args.batch_size, args.max_model_len, args.max_new_tokens, args.tensor_parallel_size,
           args.expert_parallel_size) <= 0 or args.max_samples < 0:
        parser.error("batch, context, generation, and parallel sizes must be positive; max-samples >= 0")
    input_path = Path(args.input)
    prompts = _prompts(input_path)
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    if any(isinstance(prompt, list) and len(prompt) + args.max_new_tokens > args.max_model_len
           for prompt in prompts):
        raise ValueError("A tokenized calibration prompt exceeds the configured model context.")
    config = json.loads(Path(args.runtime_config).read_text(encoding="utf-8")) if args.runtime_config else {}
    if not isinstance(config, dict) or any(key in config for key in ("sparse_method", "fp8_kv_calibration")):
        raise ValueError("Runtime config must be an object and cannot override the calibration mode.")
    config.update(
        sparse_method="vanilla",
        fp8_kv_calibration=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        max_num_seqs_in_batch=args.batch_size,
        max_decoding_seqs=args.batch_size,
        max_num_seqs_in_gpu=args.batch_size,
    )
    from sparseengine import LLM, SamplingParams

    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    llm = LLM(args.model, **config)
    try:
        for start in range(0, len(prompts), args.batch_size):
            llm.generate(prompts[start:start + args.batch_size], params, use_tqdm=False)
        result = llm.export_fp8_kv_scales(args.output, safety_margin=args.safety_margin)
    finally:
        llm.exit()
    result["calibration"]["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    result["calibration"]["sample_count"] = len(prompts)
    result["calibration"]["max_new_tokens"] = args.max_new_tokens
    output = Path(args.output)
    temporary = output.with_name(f".{output.name}.manifest.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {len(result['layers'])} FP8 KV layer scales to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
