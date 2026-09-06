"""Run a matched H2O quality matrix through the existing LongBench runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from eval import TASK_HIERARCHY, scorer


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_case(path, tasks, expected_counts):
    status = json.loads((path / 'run_status.json').read_text())
    if status['status'] != 'success':
        raise RuntimeError(f'{path}: unsuccessful run: {status}')
    identities = []
    metrics = []
    for task in tasks:
        rows = read_jsonl(path / f'{task}.jsonl')
        if len(rows) != expected_counts[task]:
            raise RuntimeError(f'{path}/{task}: expected {expected_counts[task]} samples, got {len(rows)}')
        seen = set()
        for row in rows:
            source = row['source_idx']
            if row['status'] != 'success' or source in seen:
                raise RuntimeError(f'{path}/{task}: failed or duplicate sample {source}')
            seen.add(source)
            identities.append((task, source, row['prompt_tokens']))
            score = scorer(task, [row['pred']], [row['answers']], row['all_classes'])
            metrics.append(dict(dataset=task, source_idx=source, status='success', score=score))
        if seen != set(range(expected_counts[task])):
            raise RuntimeError(f'{path}/{task}: source IDs do not match the fixed selection')
    expected_total = sum(expected_counts.values())
    for filename in ('raw_outputs.jsonl', 'parsed_outputs.jsonl', 'sample_results.jsonl'):
        rows = read_jsonl(path / filename)
        actual = sorted((r['dataset'], r['source_idx'], r['prompt_tokens']) for r in rows)
        if len(rows) != expected_total or any(r['status'] != 'success' for r in rows) or actual != sorted(identities):
            raise RuntimeError(f'{path}/{filename}: incomplete or inconsistent artifacts')
    with (path / 'per_sample_metrics.jsonl').open('w') as handle:
        for row in metrics:
            handle.write(json.dumps(row) + '\n')
    result = json.loads((path / 'result.json').read_text())
    if any(task not in result for task in tasks):
        raise RuntimeError(f'{path}: missing task scores')
    return sorted(identities), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--budgets', default='2048,4096')
    parser.add_argument('--prefill-budget', type=int, default=8192)
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--tasks', default=','.join(task for tasks in TASK_HIERARCHY.values() for task in tasks))
    parser.add_argument('--cases', help='Optional comma-separated subset of case names; baseline is required for final comparison')
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    tasks = args.tasks.split(',')
    budgets = [int(x) for x in args.budgets.split(',')]
    if len(tasks) != len(set(tasks)) or not budgets or min(budgets) <= 0 or max(budgets) > args.prefill_budget:
        raise ValueError('Tasks must be unique and budgets positive and within the prefill budget')
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError('--num-samples must be positive')
    if min(args.chunk_size, args.batch_size, args.max_model_len) <= 0:
        raise ValueError('Chunk, batch and context capacities must be positive')
    common = dict(tensor_parallel_size=1, gpu_memory_utilization=.85,
                  decode_graph=False, enable_prefix_caching=False,
                  prefill_schedule_policy='all_chunked', engine_prefill_chunk_size=args.chunk_size,
                  max_num_batched_tokens=args.chunk_size * args.batch_size,
                  max_num_seqs_in_batch=args.batch_size, max_num_seqs_in_gpu=args.batch_size)
    cases = {'vanilla': dict(sparse_method='vanilla', config=common)}
    for budget in budgets:
        for reducer in ('max', 'mean'):
            cases[f'h2o_{reducer}_{budget}'] = dict(sparse_method='h2o', config=dict(
                common, h2o_head_reduction=reducer, h2o_decode_budget=budget,
                h2o_prefill_budget=args.prefill_budget, h2o_recent_ratio=.2,
                h2o_prefill_score_window=0, sparse_prefill_score_mode='probability',
            ))
    selected_cases = args.cases.split(',') if args.cases else list(cases)
    if len(selected_cases) != len(set(selected_cases)) or any(c not in cases for c in selected_cases):
        raise ValueError(f'Invalid --cases; available: {list(cases)}')
    data = {}
    for task in tasks:
        path = Path(args.data_root) / 'data' / f'{task}.jsonl'
        raw = path.read_bytes()
        count = len(read_jsonl(path))
        if args.num_samples is not None and count < args.num_samples:
            raise ValueError(f'{task}: fewer than {args.num_samples} samples')
        data[task] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                          total_rows=count, selected_rows=args.num_samples or count)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    protocol = dict(model_path=str(Path(args.model_path).resolve()), revision=revision,
                    data=data, tasks=tasks, cases=cases, max_model_len=args.max_model_len,
                    seed=42, thinking_mode='off', temperature=0., top_p=1., top_k=1,
                    generation_limits='benchmark/long_bench/config/dataset2maxlen.json',
                    truncation='shared head/tail prompt budget reserving original task generation limit')
    protocol_path = root / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Existing output root has a different protocol; choose a new output root')
    write_json(protocol_path, protocol)
    env = dict(os.environ, SPARSEVLLM_LONGBENCH_DATA_DIR=str(Path(args.data_root).resolve()), TOKENIZERS_PARALLELISM='false')
    progress = dict(status='running', revision=revision, cases={})
    write_json(root / 'matrix_status.json', progress)
    counts = {task: data[task]['selected_rows'] for task in tasks}
    try:
        for name in selected_cases:
            case = cases[name]
            path = root / name
            if not path.exists():
                # Never share a GPU with another workload. A failed case is
                # preserved and requires a new output root, not a silent retry.
                active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
                if active:
                    raise RuntimeError(f'GPU compute processes are active: {active}')
                command = [sys.executable, '-u', 'benchmark/long_bench/pred.py',
                           '--model_path', args.model_path, '--sparse_method', case['sparse_method'],
                           '--task', args.tasks, '--output_root', str(path),
                           '--max_model_len', str(args.max_model_len), '--batch_size', str(args.batch_size),
                           '--seed', '42', '--temperature', '0', '--top_p', '1', '--top_k', '1',
                           '--thinking_mode', 'off', '--hyper_param', json.dumps(case['config'])]
                if args.num_samples is not None:
                    command.extend(['--num_samples', str(args.num_samples)])
                write_json(root / f'{name}.command.json', command)
                progress['cases'][name] = 'running'
                write_json(root / 'matrix_status.json', progress)
                with (root / f'{name}.log').open('w') as log:
                    subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_case(path, tasks, counts)
            progress['cases'][name] = 'success'
            write_json(root / 'matrix_status.json', progress)
        available = [name for name in cases if (root / name / 'run_status.json').exists()]
        results, baseline_ids = {}, None
        for name in available:
            identities, result = validate_case(root / name, tasks, counts)
            if name == 'vanilla':
                baseline_ids = identities
            elif baseline_ids is not None and identities != baseline_ids:
                raise RuntimeError(f'{name}: sample IDs or prompt lengths differ from baseline')
            results[name] = result
        write_json(root / 'comparison.json', results)
        progress['status'] = 'success' if set(available) == set(cases) else 'partial'
        write_json(root / 'matrix_status.json', progress)
        print(json.dumps(progress), flush=True)
    except Exception as error:
        progress.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(root / 'matrix_status.json', progress)
        raise


if __name__ == '__main__':
    main()
