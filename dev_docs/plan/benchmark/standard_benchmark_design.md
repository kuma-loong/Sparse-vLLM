## 标准 Benchmark 设计计划

本文档用于规范 Sparse-vLLM 稀疏化算法的开发迭代评估与最终效果评估。目标不是把所有 benchmark 都跑一遍，而是建立一套分层流程：先用低成本测试确认实现没有坏，再用可控长文本任务定位稀疏化错误，最后用真实长文本任务和 KV cache 生命周期任务报告最终效果。

当前默认范围是纯文本长上下文测试。标准 quick/final 流程默认只覆盖 sanity、固定长度微基准、NIAH、LongBench、SCBench 和 MathBench。多模态 benchmark 只在明确评估视觉 KV 稀疏化或 LLaVA-OneVision 相关改动时作为可选扩展，不进入默认主流程。

## 1. 评估目标

### 1.1 直接性能收益

评估稀疏化算法在不同上下文长度下是否真正带来推理加速和显存收益。

建议固定输入长度：

| 长度 | 用途 |
| --- | --- |
| 1k, 4k | 短上下文回归，验证短序列是否正确绕过稀疏路径或没有明显退化 |
| 16k, 32k | 常用长上下文开发区间，适合日常迭代 |
| 64k, 128k | 主要长文本报告区间 |
| 256k, 512k | 极限扩展测试，只在模型和硬件支持时运行 |

必须记录的指标：

- TTFT：time to first token，重点反映 prefill 路径。
- TPOT：time per output token，重点反映 decode 路径。
- prefill tok/s。
- decode tok/s 或 new tokens/s。
- end-to-end latency。
- peak GPU memory。
- 实际 prompt tokens、output tokens。
- 稀疏化有效比例：保留 token 数、保留 block 数、稀疏选择耗时、fallback 次数。
- 状态：success、oom、timeout、parse_failed、metric_failed、model_failed。

注意：稀疏化并不应该被要求在所有长度都有收益。短序列下如果算法进入稀疏路径反而变慢，应通过阈值控制进入条件，并在报告中明确该阈值。

### 1.2 长文本真实场景加速

评估稀疏化在真实任务 prompt、真实输出长度、真实 KV cache 生命周期下是否有端到端收益。

核心问题：

- 在 LongBench / SCBench 等真实任务中，稀疏化是否降低 TTFT 和显存。
- 多轮或多请求复用时，cache 是否命中，是否避免重复 prefill。
- 算法选择开销是否抵消了 attention 计算节省。
- 加速是否只发生在某些任务类型，例如检索类快、摘要类慢。

### 1.3 正确性与质量保持

评估稀疏化后模型输出是否仍然接近 full attention。

正确性评估分两类：

- 可控正确性：NIAH、RULER、MRCR、NoLiMa 等，答案明确，便于定位稀疏化漏保留的信息。
- 真实任务质量：LongBench、LongBench-E、LongBench v2、HELMET、InfiniteBench 等，衡量真实长文本理解能力。

所有质量 benchmark 必须同时保存：

- full attention baseline。
- 当前稀疏化版本。
- 上一个稳定稀疏化版本。
- raw outputs。
- parsed outputs。
- per-sample results。
- aggregate metrics。
- run_info，包括命令、git commit、模型、数据集、seed、decode 参数、硬件、CUDA_VISIBLE_DEVICES、稀疏化参数。

### 1.4 特定功能指标

部分功能需要独立指标，不应只看总分。

Prefix cache / shared context：

- cache hit rate。
- eligible cache hit rate，只统计理论上可复用的 prefix。
- physical KV reuse rate，区分“hash 命中”和“实际 KV 复用”。
- recomputed tokens。
- evicted tokens。
- per-turn TTFT。
- multi-request 共享上下文下的吞吐。

KV 压缩 / sparse decode：

- 每层保留 token/block 数。
- 每层选择耗时。
- 被保留的 sink/recent/full-attention token 数。
- decode 阶段每步 KV view 构造耗时。
- cache metadata 内存。

视觉 KV 稀疏化，仅在明确做多模态功能时启用：

- visual token keep ratio。
- image/video QA accuracy。
- prefill/decode speed。
- peak memory。
- 每样本视频帧数和视觉 token 数。

## 2. 分层 Benchmark 矩阵

### Tier 0: Sanity 与微基准

用途：开发前后快速确认推理链路正常，性能指标可采集。

内容：

- 5 到 10 条短 prompt sanity check。
- 固定长度 synthetic prompt 性能测试。
- 长度：1k、4k、16k、32k；最终报告扩展到 64k、128k、256k、512k。
- 输出长度：建议 1、8、32 三档；1 用于 TTFT/prefill，8/32 用于 decode。

判定：

- 输出无乱码、无异常重复、无明显格式错乱。
- 稀疏化和 full attention 都能生成。
- 每条样本都有明确 status。
- 失败样本不插值、不估算、不从平均值里静默删除。

本仓库相关入口：

- `scripts/benchmarks/bench_minference_real_model.py`：已有 `benchmark/results` 中的 report 就来自这一类测试。
- 可以作为所有稀疏化算法的统一直接性能入口扩展。

### Tier 1: 可控长文本正确性

用途：定位稀疏化是否丢掉关键 token，优先用于开发迭代。

推荐 benchmark：

| Benchmark | 用途 | 适合评估 |
| --- | --- | --- |
| NIAH | 单 needle，可控长度和插入深度 | 最基础 retrieval 正确性 |
| RULER | 多类合成长上下文任务，覆盖 retrieval、multi-hop、aggregation、QA | 真实 context length 能力曲线 |
| NeedleBench | bilingual retrieval/reasoning，长度可到 1M | 极长上下文检索与推理 |
| OpenAI MRCR | 多轮共指、多 needle 区分 | 多相似目标、对话式长上下文 |
| NoLiMa | needle 与问题低词面重合 | 防止算法只在 literal match 上过拟合 |

开发子集建议：

- NIAH：16k、32k、64k，各 3 个深度，每格 3 条。
- RULER：选择 single retrieval、multi retrieval、variable tracing、aggregation 各 1 个任务。
- MRCR：优先 2-needle 128k；最终报告再考虑 4-needle 或更长。
- NoLiMa：优先 32k/64k，用于检验 semantic retrieval。

判定：

- NIAH/MRCR 这类 exact retrieval 不允许出现未解释的大量错误。
- 如果稀疏化错而 full attention 对，必须保存错误样本、needle 深度、被保留 token 分布和层级稀疏统计。
- 如果 full attention 自身也错，标记为 baseline_failed，不用于计算 sparse degradation。

本仓库相关入口：

- `benchmark/niah/test_niah.py`。
- RULER / NeedleBench / MRCR / NoLiMa 暂未内置，可作为后续标准入口接入。

### Tier 2: 真实长文本质量

用途：报告稀疏化算法对真实长上下文任务的质量影响。

推荐 benchmark：

| Benchmark | 用途 | 适合评估 |
| --- | --- | --- |
| LongBench | bilingual、多任务、已有本仓库入口 | 真实任务质量主评估 |
| LongBench-E | 按长度分桶 | 长度越长是否越掉分 |
| LongBench v2 | 更强调真实任务中的深度理解和推理 | 最终质量报告，可作为 LongBench v1 的补充 |
| HELMET | 多应用类别的综合长上下文评估 | 更全面的外部报告 |
| InfiniteBench | 平均长度超过 100K，含中英文和真实/合成任务 | 100K+ 极长上下文质量 |
| LV-Eval | 16k 到 256k 多长度级别 | 长度分层质量曲线 |

本仓库当前主入口：

```bash
python benchmark/long_bench/pred.py \
  --model <name> \
  --model_path <model_path> \
  --sparse_method <method> \
  --deltakv_checkpoint_path <checkpoint_or_none> \
  --task qasper,hotpotqa,gov_report,passage_retrieval_en \
  --num_samples 20 \
  --batch_size 1
```

开发子集建议：

| 类别 | LongBench 任务 |
| --- | --- |
| Single-doc QA | `qasper`, `narrativeqa` |
| Multi-doc QA | `hotpotqa`, `2wikimqa` |
| Summarization | `gov_report`, `qmsum` |
| Synthetic retrieval | `passage_retrieval_en`, `passage_count` |
| Code | `lcc`, `repobench-p` |

最终报告建议：

- 跑完整 LongBench 英文 + code 子集。
- 如果论文或报告需要 bilingual 结果，再加中文任务。
- 跑 LongBench-E 分桶结果，报告 `0-4k / 4-8k / 8k+` 的 quality delta。
- 如果算法目标是 100K+，增加 InfiniteBench 或 LV-Eval。

判定：

- 不只报告整体平均分，必须报告每个任务分数和类别平均。
- 重点关注检索类、代码类和摘要类是否有不同退化模式。
- 建议默认质量预算：相对 full attention 平均分下降不超过 1 到 2 分；exact retrieval 类任务应更严格。最终阈值由具体实验目标确定，并写入 run config。

### Tier 3: KV Cache 生命周期与 Prefix Cache

用途：评估稀疏化在多轮、多请求、cache 复用场景中的真实可靠性。

首选 benchmark：

| Benchmark | 用途 |
| --- | --- |
| SCBench | KV cache generation、compression、retrieval、loading 全生命周期 |
| MRCR | 多轮/多 needle 对话检索正确性 |
| 自定义 prefix-cache trace | 明确 cache 命中率、实际 KV 复用、逐轮 TTFT |
| ClawBench 类 agent workload | 可作为 agent-style 真实请求流量来源，但不是长上下文稀疏化质量主 benchmark |

SCBench 推荐优先任务：

| 任务 | 原因 |
| --- | --- |
| `scbench_kv` | key-value 精确查找，最容易暴露 cache 压缩错误 |
| `scbench_prefix_suffix` | 前后缀字符串检索 |
| `scbench_vt` | 变量追踪，多跳依赖 |
| `scbench_qa_eng` | 真实语义 QA |
| `scbench_summary_with_needles` | 全局摘要 + 局部 needle |
| `scbench_repoqa_and_kv` | 代码语义检索 + KV 查找混合 |

本仓库入口：

```bash
python benchmark/scbench/run_scbench_preprocessed.py \
  --task scbench_kv,scbench_qa_eng,scbench_summary_with_needles \
  --data_root <SCBench-preprocessed-root> \
  --model_name_or_path <model_path> \
  --attn_type <method> \
  --max_seq_length 131072 \
  --hyper_param '<json>'
```

本地数据已经处理好后，可以把数据链接到仓库忽略目录，让 benchmark 脚本在没有环境变量时直接发现：

```bash
mkdir -p benchmark/data
ln -sfn <LongBench-root> benchmark/data/LongBench
ln -sfn <SCBench-preprocessed-root> benchmark/data/SCBench-preprocessed
```

`run_scbench_preprocessed.py` 只接受带有 `prompts` 和 `ground_truth` 列的 preprocessed parquet。传入 raw snapshot 时应在数据校验阶段快速失败，不应先加载模型。

需要额外记录：

- mode：multi-turn 或 multi-request/SCDQ。
- 每轮 prompt tokens、cached tokens、uncached tokens。
- cache hit rate 和 eligible cache hit rate。
- prefix cache 是否因为 eviction、hash mismatch、block size、tensor parallel 切分而未复用。
- 每轮 TTFT，而不是只看总平均。

判定：

- 如果单请求正确、多轮错误，不能归类为普通 quality drop，必须归类为 cache lifecycle regression。
- 如果报告 prefix cache 加速，必须同时报告 hit rate 和实际 TTFT 下降，不能只报告 theoretical cached tokens。

### Tier 4: 非长文本回归与领域扩展

用途：确认稀疏化实现没有破坏普通推理或特定模态能力。

推荐：

- MathBench：GSM8K、AIME、HMMT。用于推理稳定性回归，不作为长文本主 benchmark。
- AI2D / Video-MME / StreamingBench / visual_cache：仅用于视觉 KV 稀疏化，不属于当前纯文本主流程。

本仓库入口：

```bash
python benchmark/math_bench/pred.py \
  --model <name> \
  --model_path <model_path> \
  --sparse_method <method> \
  --deltakv_checkpoint_path <checkpoint_or_none> \
  --task gsm8k,aime2024 \
  --num_samples 50 \
  --temperature 0.6 \
  --max_new_tokens 512
```

## 3. 开发迭代流程

每个功能或优化项从开发开始就必须维护一张 benchmark 数据表。它不是最终报告的附属物，而是开发过程的一部分，用于记录每一次实验尝试、失败原因、参数变化和结论。这样可以避免只保留“跑得好”的结果，也能在性能或正确性回退时快速定位是哪一次改动引入的。

当前标准总控入口：

```bash
.venv/bin/python scripts/benchmarks/run_standard_benchmark.py \
  --mode quick \
  --feature <feature_slug> \
  --objective "<what this run validates>" \
  --model_path <MODEL_PATH> \
  --primary_method <method> \
  --methods vanilla,<method>
```

运行前可加 `--dry_run` 查看命令计划。缺少 LongBench 或 SCBench 数据时，总控脚本应记录 `invalid_run`，不自动下载数据集。

建议一个功能对应一个 ledger：

```text
benchmark/results/_ledgers/<feature_slug>.jsonl
benchmark/results/_ledgers/<feature_slug>.csv
```

JSONL 作为机器可读的原始记录，CSV 作为人工查看和排序的表格。两者字段应保持一致；如果只维护一种格式，优先维护 JSONL。

每次 benchmark 无论成功、失败、OOM、timeout 或中途发现配置错误，都要追加一行记录。配置错误可以标记为 `invalid_run`，但不能直接丢弃，因为这类记录经常解释了后续结果为何不可比。

推荐字段：

| 字段 | 说明 |
| --- | --- |
| `run_id` | 唯一 id，建议 `<feature>_<date>_<short_hash>_<index>` |
| `timestamp` | 运行开始时间 |
| `feature` | 功能或优化项名称 |
| `objective` | 本次 benchmark 要验证的问题，例如 `verify 64k TTFT speedup` |
| `git_commit` | 当前 commit；未提交时记录 `dirty=true` |
| `branch` | 当前分支 |
| `code_diff_ref` | 可选，关联 patch、PR 或实验说明 |
| `benchmark` | benchmark 名称，例如 `microbench`, `niah`, `longbench`, `scbench` |
| `benchmark_tier` | Tier 0 到 Tier 4 |
| `benchmark_source` | `repo_existing`, `external_official`, `custom` |
| `script` | 实际入口脚本 |
| `command` | 完整命令 |
| `model_path` | 模型路径 |
| `tokenizer_path` | tokenizer 路径 |
| `method` | full attention、当前 sparse 方法或对比方法 |
| `method_config` | 稀疏化关键参数 JSON |
| `baseline_run_id` | 对应 full attention baseline |
| `previous_run_id` | 对应上一个稳定 sparse 版本 |
| `dataset` | 数据集名称 |
| `split` | 数据 split |
| `sample_policy` | `smoke`, `fixed_subset`, `full` |
| `sample_ids` | 固定子集 id 或其文件路径 |
| `lengths` | 输入长度列表 |
| `max_new_tokens` | 输出长度设置 |
| `decode_config` | temperature、top_p、top_k、beam 等 |
| `gpu` | GPU 型号和 CUDA_VISIBLE_DEVICES |
| `env` | 关键环境变量和依赖版本 |
| `output_dir` | 本次完整结果目录 |
| `status` | `success`, `invalid_run`, `oom`, `timeout`, `model_failed`, `metric_failed` 等 |
| `primary_metrics` | TTFT、TPOT、prefill tok/s、decode tok/s、score 等摘要 |
| `quality_delta` | 相对 full attention 的质量差异 |
| `speedup` | 相对 full attention 或上个版本的速度收益 |
| `memory_delta` | 显存变化 |
| `failure_summary` | 失败原因摘要 |
| `decision` | `keep`, `revert`, `investigate`, `rerun`, `promote_to_final` |
| `notes` | 人工备注 |

开发过程中应优先比较同一个 ledger 中的连续记录，而不是跨机器、跨模型、跨数据子集随意比较。只有当 `model_path`、数据子集、decode 参数、GPU、稀疏化入口和 benchmark 脚本版本都一致时，速度和质量差异才默认可比。

### 3.1 每次改动后的最小流程

1. Sanity prompt：5 到 10 条短请求。
2. 微基准：1k、4k、16k、32k，输出 1/8 tokens。
3. NIAH smoke：16k、32k，多个 needle 深度。
4. LongBench smoke：`qasper,hotpotqa,passage_retrieval_en`，每任务 10 到 20 条。
5. 如果改动涉及 cache manager 或 decode view，额外跑 SCBench `scbench_kv` 小样本。
6. 将以上每次运行追加到该功能的 benchmark ledger。

通过后才进入更长长度或更多任务。

### 3.2 阶段性里程碑流程

1. 微基准：1k、4k、16k、32k、64k、128k。
2. NIAH：16k、32k、64k、128k，10 个深度。
3. SCBench：`scbench_kv,scbench_qa_eng,scbench_summary_with_needles`。
4. LongBench：每类至少 1 到 2 个任务，每任务 50 条。
5. MathBench：GSM8K 小样本。

### 3.3 最终报告流程

1. 微基准完整长度矩阵，建议每个 case 至少 3 次重复，报告 mean 和单次原始记录。
2. NIAH/RULER 或 NeedleBench 完整长度曲线。
3. SCBench multi-turn 和 multi-request/SCDQ。
4. LongBench 或 LongBench-E 完整评估。
5. 如果声明 100K+ 能力，增加 InfiniteBench、LV-Eval 或 HELMET。
6. 如果声明复杂推理不退化，增加 LongBench v2 和 MathBench。
7. 如果明确声明视觉稀疏化，另行增加 Video-MME、StreamingBench、visual_cache；否则最终报告仍保持纯文本。

## 4. 对比对象

每次正式 benchmark 至少包含三组：

| 组别 | 目的 |
| --- | --- |
| full attention baseline | 正确性上限和速度下限 |
| previous stable sparse version | 判断本次迭代是否回归 |
| current sparse version | 当前结果 |

可选组：

- SnapKV、Quest、StreamingLLM、MInference 等外部方法。
- 不同稀疏比例或 block size。
- 不同 full-attention-layer 配置。
- 不同 prefix cache 开关。

比较时必须保证：

- 同一模型。
- 同一 tokenizer。
- 同一 prompt 模板。
- 同一数据顺序和样本子集。
- 同一 decode 参数。
- 同一 GPU 型号和可见设备。
- 同一 max_model_len、gpu_memory_utilization、batch size。

## 5. 数据与结果保存规范

每次运行创建独立目录：

```text
benchmark/results/<benchmark>/<model>/<method>_<length_or_task>_<time>/
```

建议文件：

```text
run_info.json
raw_outputs.jsonl
parsed_outputs.jsonl
per_sample_results.jsonl
aggregate_metrics.json
performance.jsonl
report.md
stderr.log
stdout.log
```

ledger 中的 `output_dir` 指向该目录。目录中保存详细结果，ledger 保存跨多次开发迭代的索引和摘要。不要只保留最终目录而没有 ledger，也不要只在 ledger 写摘要而丢掉原始输出。

`run_info.json` 必须包含：

- git commit。
- 命令行。
- 环境变量。
- 模型路径。
- tokenizer 路径。
- 数据集路径和 split。
- 样本数量和样本 id。
- seed。
- decoding 参数。
- 稀疏化参数。
- GPU 型号、CUDA_VISIBLE_DEVICES。
- benchmark 脚本版本。

`per_sample_results.jsonl` 每条样本至少包含：

- sample_id。
- task。
- input_tokens。
- output_tokens。
- status。
- full_attn_output 或 baseline score。
- sparse_output。
- parsed_answer。
- reference。
- metric。
- error_message。
- 稀疏化统计。

状态必须显式使用：

- `success`
- `invalid_run`
- `invalid_input`
- `model_failed`
- `parse_failed`
- `metric_failed`
- `skipped_by_policy`
- `oom`
- `timeout`

失败样本不能静默删除。aggregate metrics 应同时报告：

- all samples。
- success-only samples。
- failure counts。
- skipped counts。

### 5.1 Feature-level Ledger 示例

示例 JSONL 单行：

```json
{"run_id":"minference_refactor_20260608_a1b2c3_003","timestamp":"2026-06-08T21:30:00+08:00","feature":"minference_refactor","objective":"verify 32k and 64k TTFT after block selector change","git_commit":"a1b2c3d","dirty":true,"benchmark":"microbench","benchmark_tier":"Tier 0","benchmark_source":"repo_existing","script":"scripts/benchmarks/bench_sparse_vllm.py","command":"python scripts/benchmarks/bench_sparse_vllm.py --methods vanilla,minference_full --lengths 32768,65536 --output_len 8 ...","model_path":"/path/to/Qwen2.5-7B-Instruct-1M","method":"minference_full","method_config":{"block_size":128,"ratio":0.5},"baseline_run_id":"minference_refactor_20260608_a1b2c3_002","dataset":"synthetic","sample_policy":"fixed_subset","lengths":[32768,65536],"gpu":"A100-SXM4-80GB CUDA_VISIBLE_DEVICES=0","output_dir":"benchmark/results/minference_refactor_32k64k_003","status":"success","primary_metrics":{"ttft_s_32k":1.49,"prefill_tok_s_32k":22013.2,"peak_gb_32k":55.56},"speedup":{"ttft_vs_full_32k":0.92},"decision":"investigate","notes":"32k slower than full attention; selector overhead dominates"}
```

CSV/Markdown 视图可以只保留摘要列：

| run_id | objective | benchmark | method | lengths/tasks | status | score/delta | speedup | peak GB | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `..._003` | verify 32k/64k TTFT | microbench | minference_full | 32k,64k | success | n/a | 0.92x @32k | 55.56 | investigate |

### 5.2 Ledger 使用规则

- 每个功能开发开始时先建立 ledger，并记录第一组 full attention baseline。
- 每次修改算法阈值、block size、cache metadata、kernel、prompt 模板或 decode 参数，都必须产生新的 run_id。
- smoke run、失败 run、正式 run 都写入同一 ledger，用 `sample_policy` 和 `status` 区分。
- 最终报告只能引用 ledger 中 `decision=promote_to_final` 的记录。
- 如果某次结果不可比，必须在 `decision` 写 `rerun` 或 `invalid_run`，并在 `failure_summary` 说明原因。
- 不允许覆盖已有 run 目录；重跑同一配置也生成新 run_id，并在 notes 中关联旧 run。

## 6. 通过标准建议

具体阈值应按实验目标写入配置。默认建议如下：

### 6.1 Sanity

- 所有短 prompt 成功生成。
- 无乱码、无空输出、无无限重复。
- full attention 和 sparse 路径均可运行。

### 6.2 性能

- 16k 以下允许无收益，但不能显著变慢，除非配置明确禁用短序列稀疏化。
- 32k 及以上应报告 TTFT 或 prefill tok/s 的明确收益。
- decode-only 稀疏算法必须报告 TPOT 或 decode tok/s 收益，不能只报 TTFT。
- 显存收益必须用 peak memory 和 cache metadata overhead 一起报告。

### 6.3 正确性

- NIAH/MRCR/RULER exact retrieval：稀疏化错误必须逐样本分析。
- LongBench：报告每任务 delta 和类别平均 delta，不只报告 overall。
- 如果 full attention baseline 本身失败，应单独标记，避免把模型能力问题归因到稀疏算法。
- 如果 sparse 质量下降超过预设预算，必须附带失败样本和稀疏统计。

### 6.4 Prefix Cache

- cache hit rate、eligible cache hit rate、physical KV reuse rate 三者都要记录。
- 必须报告 per-turn TTFT。
- 如果 cache 命中但 TTFT 没下降，应定位为 cache accounting 或实际复用问题。

## 7. Benchmark 选择建议

选择 benchmark 时优先使用可靠、可复现、已有仓库入口的测试。外部 benchmark 只有在它能补足仓库现有测试缺口时才接入，并且要先固定版本、数据、metric 和样本子集。

可靠性优先级：

| 优先级 | 类型 | 例子 | 使用原则 |
| --- | --- | --- | --- |
| P0 | 仓库已有、已能产出 raw/per-sample/aggregate 的纯文本入口 | microbench、NIAH、LongBench、SCBench preprocessed、MathBench | 默认优先使用，作为开发迭代主路径 |
| P1 | 仓库已有但需要补记录字段或数据准备的入口 | 原始 SCBench | 可用于纯文本相关功能，但先补齐 run_info 和 per-sample 记录 |
| P1-vision | 仓库已有多模态入口 | visual_cache、Video-MME、StreamingBench、AI2D | 仅在视觉 KV 或多模态任务明确需要时使用 |
| P2 | 外部官方 benchmark，代码/数据/metric 清晰 | RULER、MRCR、NoLiMa、LV-Eval、InfiniteBench、HELMET、LongBench v2 | 用于最终报告或补足能力维度，接入后固定版本 |
| P3 | 探索性或 workload 型 benchmark | ClawBench 类 agent workload、自定义 prefix-cache trace | 用于观察真实请求形态，不作为主要质量结论来源 |

优先使用仓库现有 benchmark 的原因：

- 已经能跑通本项目的模型加载、稀疏方法和输出目录约定。
- 更容易和历史结果比较。
- 更容易加上稀疏化内部统计。
- 降低外部依赖、数据下载和 metric 不一致带来的噪声。

外部 benchmark 接入前必须满足：

- 有官方或可信实现。
- 数据版本可固定。
- metric 定义明确，最好不依赖非确定性 LLM judge。
- 可以保存 raw outputs、parsed outputs、per-sample results。
- 能和 full attention baseline 在同一模型、同一 prompt 模板、同一 decode 参数下对比。

### 7.1 评估稀疏化加速效果

优先级：

1. 仓库固定长度微基准：最干净地测 TTFT、TPOT、prefill tok/s、decode tok/s、peak memory。
2. 仓库 SCBench preprocessed：最适合测 KV cache 生命周期、多轮和多请求加速。
3. 仓库 NIAH：适合同时看长度扩展、位置敏感性和速度。
4. 仓库 LongBench：适合报告真实任务端到端速度，但不是纯性能 benchmark。
5. 外部 RULER / NeedleBench：用于补足更标准的长度能力曲线。

不建议用 MathBench 做长文本加速主评估。

### 7.2 评估稀疏化后是否出错

优先级：

1. 仓库 NIAH：可控、答案明确，便于快速定位错误。
2. 仓库 SCBench：尤其适合 cache lifecycle correctness。
3. 仓库 LongBench/LongBench-E：真实任务质量主评估。
4. 外部 MRCR/RULER/NoLiMa：补足多 needle、复杂合成任务和非 literal retrieval。
5. 外部 LongBench v2/HELMET/InfiniteBench/LV-Eval：最终报告或论文级补充。
6. MathBench：普通推理回归，不是长文本正确性主评估。

## 8. 外部 Benchmark 参考

以下 benchmark 可作为后续接入或最终报告参考：

| Benchmark | 参考 | 备注 |
| --- | --- | --- |
| LongBench | https://arxiv.org/abs/2308.14508 | bilingual、多任务长上下文理解 |
| LongBench v2 | https://longbench2.github.io/ | 更强调真实多任务深度理解与推理 |
| RULER | https://github.com/NVIDIA/RULER | 可配置长度和任务复杂度，适合 context length 能力曲线 |
| SCBench | https://huggingface.co/datasets/microsoft/SCBench | KV cache 生命周期、多轮、多请求 |
| InfiniteBench | https://arxiv.org/abs/2402.13718 | 100K+ 长上下文，中英文、真实和合成任务 |
| HELMET | https://github.com/princeton-nlp/HELMET | 综合长上下文评估，覆盖多应用类别 |
| NeedleBench | https://huggingface.co/papers/2407.11963 | bilingual，检索和推理，长度到 1M |
| NoLiMa | https://github.com/adobe-research/NoLiMa | 非 literal matching 的长上下文检索 |
| OpenAI MRCR | https://huggingface.co/datasets/openai/mrcr | 多 needle、多轮共指长上下文 |
| LV-Eval | https://arxiv.org/abs/2402.05136 | 16k 到 256k 分层长上下文评估 |

## 9. 待实现事项

1. 统一 benchmark runner，支持 `--benchmark`, `--method`, `--model_path`, `--lengths`, `--tasks`, `--num_samples`, `--output_dir`。
2. 实现 feature-level benchmark ledger 自动追加工具，支持 JSONL 和 CSV。
3. 为所有 runner 统一写入 `run_info.json`、`per_sample_results.jsonl`、`aggregate_metrics.json`。
4. 给微基准增加 full attention / previous sparse / current sparse 三路对比。
5. 给 NIAH 输出每层保留 token/block 统计，便于定位 needle 是否被保留。
6. 给 SCBench 增加 per-turn TTFT、cache hit、physical KV reuse 记录。
7. 接入 RULER 或 MRCR 作为 Tier 1 标准正确性测试。
8. 接入 LongBench-E 的固定开发子集和完整最终评估配置。
9. 增加失败样本归档和重跑工具，避免只看 aggregate 分数。
