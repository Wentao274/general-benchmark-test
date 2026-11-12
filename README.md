# general-benchmark-test

模型推理性能基准测试套件，支持 SGLang / vLLM 等推理框架。

## 目录结构

```
general-benchmark-test/
├── Model_Inference_Benchmark_TestStrategy.md   完整测试方案文档
├── benchmark_analysis_template.md              Markdown 报告模板
├── serve_command.sh.template                   模型服务启动命令模板（需复制为 serve_command.sh）
└── _scripts/
    ├── bench.sh                                基础 Benchmark 脚本
    ├── prefill_bench.sh                        纯 Prefill 测试脚本
    ├── decode_bench.sh                         纯 Decode 测试脚本
    ├── cache_bench.sh                          指定缓存命中率测试脚本
    ├── decode_http_sweep.py                    纯 Decode HTTP 精确测量【仅供参考，实际不使用】
    ├── collect_results.py                      日志收集 → CSV
    └── csv_to_md.py                            CSV → Markdown 报告（标题注明测试类型）
```

## 前置条件

1. 推理服务已启动并可访问
2. 复制 `serve_command.sh.template` 为 `serve_command.sh`，填写真实的模型服务启动命令
   ```bash
   cp serve_command.sh.template serve_command.sh
   ```
   > 每个测试脚本在执行前会校验 `serve_command.sh`：文件不存在或内容为空都会报错中止
3. 所有脚本默认后台执行，加 `-f` 切前台

---

## 1. 基础 Benchmark 测试

测量端到端推理性能：输入/输出/总 token 吞吐量、请求吞吐量、TTFT 与 TPOT（各含 Mean/Median/P90/P95/P99）。

> 延迟指标来源：SGLang bench 默认输出全部百分位；vLLM bench 由脚本统一传 `--metric_percentiles 90,95,99`（Mean/Median 始终输出）。

### 服务端要求

**必须关闭前缀缓存**，确保每个请求全量 prefill，结果可比可复现。

原因：本脚本用固定 seed 生成随机数据集——重复执行、以及同一轮内不同并发档（num_prompts 递增时 prompt 集合嵌套）都会产生相同请求。若服务端开启前缀缓存，会产生跨轮/轮内缓存命中，吞吐虚高且冷热轮结果不可比。

**脚本会自动实测缓存命中率并写入报告**（不是固定 0%）：

- **SGLang**：每组测试前后查询服务端 `/metrics` 的 `sglang:prefill_effective_tokens_total` 计数器（跨 rank 求和），差值计算 HiCache 三层命中率：
  - `L1命中率`（device_hit，GPU 缓存）
  - `L2命中率`（host_hit，CPU 缓存）
  - `L3命中率`（storage_hit，DFKV 存储）
  - 三层命中率之和 = 总命中率；`--cache-report` 仅统计 L1（radix cache），`/metrics` 覆盖完整 HiCache 层级
- **vLLM**：每组测试前后查询服务端 `/metrics` 的 `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` 计数器（L1 GPU 缓存）和 `vllm:external_prefix_cache_hits` 计数器（L3 external/LM Cache），差值计算两层命中率：
  - `L1命中率`（GPU prefix cache）
  - `L3命中率`（external/LM Cache，L1 miss 后查 L3；无 LM Cache 部署时为 0%）

服务端确实关闭缓存时，报告中命中率应为 `0%`；若明显大于 0%，说明服务端未按要求关闭缓存，结果不可比（显示 `N/A` 表示无法访问服务端 `/metrics`，未采集到）。

```bash
# SGLang — 加 --disable-radix-cache
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-Channel-FP8-w8a8 \
  --tp 8 --port 8080 \
  --disable-radix-cache

# vLLM — 加 --no-enable-prefix-caching
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  -tp 8 --port 8000 \
  --no-enable-prefix-caching
```

### 执行命令

```bash
# SGLang
./_scripts/bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan -P agg

# vLLM
./_scripts/bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan -P agg \
  -d "GLM-5.2 FP8 量化，8卡 TP 部署"
```

### 自定义参数

```bash
# 自定义并发数和 IO 组合
./_scripts/bench.sh -F sglang \
  -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 \
  -c 1,8,32,128 \
  -i "8192 512,32768 512" \
  -T zhangsan -P agg

# 前台执行 + 自定义报告目录
./_scripts/bench.sh -F vllm -f \
  -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 \
  -r /mnt/results/vllm_bench -s 30 -T zhangsan -P agg
```

> **测试数据集**：SGLang `random-ids` / vLLM `random`——bench 工具按 IO 组合动态生成的随机 token 序列（无真实语义，固定 seed 可复现），请求间无共享前缀（`random-prefix-len 0`），配合服务端关闭前缀缓存，结果不受缓存命中影响。

### 参数说明

| 参数 | 短写 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--concurrency` | `-c` | | `1,8,32,64,128` | 并发数列表（num_prompts = 并发数 × 并发倍数，`--concurrency-multiplier` 默认 1；各 IO 组合按输入长度限并发上限：≤64K 全档 / 128K~256K ≤64 / 512K ≤32 / 1M ≤16，超出档位自动跳过；各组合实际执行档位见文末「快速参考：默认测试矩阵」） |
| `--io-combinations` | `-i` | | `8192 512,32768 512,65536 512,131072 1024,262144 1024,512000 1024,1000000 1024` | IO 组合（逗号分隔，每组 `"输入 输出"`） |
| `--seed` | | | `123` | 随机数据集种子（固定 seed 生成随机数据集，可复现） |
| `--concurrency-multiplier` | | | `1` | 并发倍数：请求数 num_prompts = 并发数 × 此值 |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀，如 `H100`） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--pd` | `-P` | ✅ | — | PD部署模式: `agg`(非PD分离) 或 `disagg`(PD分离) |
| `--report-dir` | `-r` | | `./{framework}_reports` | 报告输出目录 |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--describe` | `-d` | | — | 模型描述信息（可选，填入报告末尾单独章节） |
| `--foreground` | `-f` | | 后台 | 前台执行 |
| `--timeout` | | | `3600` | 每组测试超时秒数（超时自动终止并跳过，继续下一组） |

### 产物

每次测试单独创建带时间戳的子目录 `{TS}`（`YYYYMMDD_HHMMSS`），重复执行不会覆盖：

- 日志：`{report-dir}/{tester}/{model_name}/{TS}/bench_bs-{concurrency}_input_len-{in}-output_len-{out}.log`
- CSV：`{report-dir}/{tester}/{model_name}/{TS}/results.csv`
- Markdown 报告：`{report-dir}/{tester}/{model_name}/{TS}/{tester}_{model_name}_{chip_type}_{framework}_{pd}_bench_{TS}.md`

---

## 2. 纯 Prefill 测试

测量 prefill 前向算力：output_len=1，0 次 decode 迭代，TTFT ≈ 端到端时间。

### 服务端要求

**必须关闭前缀缓存**，否则前缀命中会测出物理上不可能的吞吐值。

```bash
# SGLang — 加 --disable-radix-cache
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-Channel-FP8-w8a8 \
  --tp 8 --port 8080 \
  --disable-radix-cache

# vLLM — 加 --no-enable-prefix-caching
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  -tp 8 --port 8000 \
  --no-enable-prefix-caching
```

### 执行命令

```bash
# SGLang 纯 Prefill
./_scripts/prefill_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan -P agg

# vLLM 纯 Prefill
./_scripts/prefill_bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan -P agg
```

### 自定义参数

```bash
# 自定义输入长度和并发
./_scripts/prefill_bench.sh -F sglang \
  -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 \
  -c 1,8,64 -i "32768 1,65536 1" \
  -T zhangsan -P agg
```

> **测试数据集**：与基础 Benchmark 相同——SGLang `random-ids` / vLLM `random`（随机 token，动态生成，固定 seed 可复现），`random-prefix-len 0` 无共享前缀。

### 参数说明

| 参数 | 短写 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--io-combinations` | `-i` | | `65536 1,131072 1,262144 1` | IO 组合（逗号分隔，每组 `"输入 输出"`） |
| `--seed` | | | `123` | 随机数据集种子（固定 seed 生成随机数据集，可复现） |
| `--concurrency-multiplier` | | | `5` | 并发倍数：请求数 num_prompts = 并发数 × 此值 |
| `--concurrency` | `-c` | | `1,8,32,64,128` | 并发数列表（num_prompts = 并发数 × 并发倍数，`--concurrency-multiplier` 默认 5；各 IO 组合按输入长度限并发上限：≤64K 全档 / 128K~256K ≤64，超出档位自动跳过） |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--pd` | `-P` | ✅ | — | PD部署模式: `agg`(非PD分离) 或 `disagg`(PD分离) |
| `--report-dir` | `-r` | | `./{framework}_prefill_reports` | 报告输出目录 |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--describe` | `-d` | | — | 模型描述信息（可选，填入报告末尾单独章节） |
| `--foreground` | `-f` | | 后台 | 前台执行 |
| `--timeout` | | | `3600` | 每组测试超时秒数（超时自动终止并跳过，继续下一组） |

### 产物

- 日志：`{report-dir}/{tester}/{model_name}/{TS}/prefill_bs-{concurrency}_input-{in}.log`
- CSV：`{report-dir}/{tester}/{model_name}/{TS}/prefill_results.csv`
- Markdown 报告：`{report-dir}/{tester}/{model_name}/{TS}/{tester}_{model_name}_{chip_type}_{framework}_{pd}_prefill_{TS}.md`

### 关注指标

- **prefill 吞吐量** = 输入 token 吞吐量（SGLang 直接提取 `Input token throughput`；vLLM 用 `总吞吐量 - 输出吞吐量` 计算）
- **TTFT** ≈ 端到端时间（无 decode，首 token 延迟即 prefill 耗时；报告含 Mean/Median/P90/P95/P99 TTFT）
- `Output token throughput` 在 output_len=1 时等于 QPS，无参考意义

---

## 3. 纯 Decode 测试

测量 decode 迭代性能：共享前缀 + 前缀缓存命中 → prefill≈0，全部耗时是 decode。

### 服务端要求

**必须开启前缀缓存**（默认开启，不要加关闭参数）。

```bash
# SGLang — 默认开启，不要加 --disable-radix-cache
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-Channel-FP8-w8a8 \
  --tp 8 --port 8080

# vLLM — 默认开启，不要加 --no-enable-prefix-caching
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  -tp 8 --port 8000
```

### 执行命令

```bash
# SGLang 纯 Decode（使用 generated-shared-prefix 数据集）
./_scripts/decode_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 8192,32768,65536,131072,262144 \
  -c 1,8,32,64,128 \
  -T zhangsan -P agg

# vLLM 纯 Decode（使用 prefix_repetition 数据集）
./_scripts/decode_bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 8192,32768,65536,131072,262144 \
  -c 1,8,32,64,128 \
  -T zhangsan -P agg
```

### 自定义参数

```bash
# 自定义前缀、输出和并发
./_scripts/decode_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 8192,65536 -o 512 -c 1,16,128 \
  -T zhangsan -P agg

# 4 组共享前缀（组内共享、组间不同）
./_scripts/decode_bench.sh -F sglang \
  -u http://127.0.0.1:8080 -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -p 8192 -c 1,8 -g 4 -T zhangsan -P agg
```

> **测试数据集**：SGLang `generated-shared-prefix`（GSP）/ vLLM `prefix_repetition`——合成共享前缀数据集：同组所有请求共享完全相同的前缀（长度 = `--prefix-lens`），前缀之外每请求仅 1 个新 token；首个请求填充前缀缓存后，其余请求 prefill≈0。组数由 `-g` 控制（默认 1）。

### 参数说明

| 参数 | 短写 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--prefix-lens` | `-p` | | `8192,32768,65536,131072,262144` | 前缀长度列表 |
| `--output-lens` | `-o` | | —（按前缀组配对） | 输出长度列表；缺省时 ≤64K 前缀 → 512、128K~256K 前缀 → 1024，显式指定则前缀 × 输出全交叉 |
| `--concurrency` | `-c` | | `1,8,32,64,128` | 并发数列表（num_prompts = 并发数 × 并发倍数，`--concurrency-multiplier` 默认 10） |
| `--gsp-num-groups` | `-g` | | `1` | 共享前缀组数（组内请求共享同一前缀，组间不同；每组条数 = ceil(并发倍数×并发数/组数)，总请求数 = 每组条数 × 组数） |
| `--seed` | | | `123` | 随机数据集种子（固定 seed 生成随机数据集，可复现；预热与测试共享前缀依赖固定 seed） |
| `--concurrency-multiplier` | | | `10` | 并发倍数：请求总数 = 每组条数 × 组数，每组条数 = ceil(此值×并发数/组数) |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--pd` | `-P` | ✅ | — | PD部署模式: `agg`(非PD分离) 或 `disagg`(PD分离) |
| `--report-dir` | `-r` | | `./{framework}_decode_reports` | 报告输出目录 |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--describe` | `-d` | | — | 模型描述信息（可选，填入报告末尾单独章节） |
| `--foreground` | `-f` | | 后台 | 前台执行 |
| `--timeout` | | | `3600` | 每组测试超时秒数（超时自动终止并跳过，继续下一组） |

### 产物

- 日志：`{report-dir}/{tester}/{model_name}/{TS}/decode_bs-{concurrency}_prefix-{prefix}-output-{out}.log`
- CSV：`{report-dir}/{tester}/{model_name}/{TS}/decode_results.csv`
- Markdown 报告：`{report-dir}/{tester}/{model_name}/{TS}/{tester}_{model_name}_{chip_type}_{framework}_{pd}_decode_{TS}.md`

### 关注指标

- **decode 吞吐量** = `Output token throughput (tok/s)`
- **TPOT** = `Mean/Median/P90/P95/P99 TPOT (ms)`
- **TTFT** = `Mean/Median/P90/P95/P99 TTFT (ms)`（应很小，若很大说明缓存未生效）

> num_prompts = 并发倍数（`--concurrency-multiplier`，默认 10）× 并发数（`-g` 指定组数 G 时：每组条数 = ceil(并发倍数×并发/G)，总请求数 = 每组条数 × G），增加统计样本数，提高吞吐和延迟测量稳定性（bench 工具自带 warmup 预热前缀缓存）。

---

## 4. 指定缓存命中率测试

测量指定缓存命中率下的推理性能（前缀级命中）：每个请求的前缀部分命中缓存，后缀部分未命中。

### 原理（前缀级命中 + 两阶段测试）

```
每个请求的输入 = 共享前缀（命中部分）+ 唯一后缀（未命中部分）
缓存命中率 = 共享前缀长度 / 输入总长度 × 100%

阶段 1 — 预热（不计量）：
  用 concurrency=1 串行发送 G 条请求（每组 1 条，G = -g 组数，默认 1），将各组共享前缀填入缓存

阶段 2 — 测试（计量）：
  用目标并发发送 N_total = 并发倍数 × 并发 个请求（默认倍数 5）
  每组内请求共享该组前缀（命中）→ prefill ≈ 0
  每个请求有不同的后缀（未命中）→ prefill 唯一后缀部分
```

| 目标命中率 | 输入长度 | 共享前缀 (hit) | 唯一后缀 (miss) | 每请求 prefill |
|---|---|---|---|---|
| 50% | 65536 | 32768 | 32768 | 32768 tokens |
| 90% | 65536 | 58982 | 6554 | 6554 tokens |
| 95% | 65536 | 62259 | 3277 | 3277 tokens |

> 前缀级命中：每个请求都部分命中缓存，TTFT 均匀分布（不像请求级方案有双峰分布）。
> 共享前缀长度 = 输入长度 × 命中率，唯一后缀长度 = 输入长度 - 共享前缀长度。
> 上表以 64K 组为例；131072 / 262144 组按同比例计算（90% → 117964/13108、235929/26215）。

3 组 IO 组合（65536/512、131072/1024、262144/1024），并发数逐级递增；262144 组要求模型上下文 ≥ 263168。

### 服务端要求

必须**开启**前缀缓存：

| 框架 | 启动参数 |
|---|---|
| **SGLang** | 默认开启（不要加 `--disable-radix-cache`） |
| **vLLM** | 默认开启（不要加 `--no-enable-prefix-caching`） |

### 执行命令

```bash
# SGLang（默认 90% 命中率）
./_scripts/cache_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan -P agg \
  -d "GLM-5.2 FP8 量化，8卡 TP 部署"

# vLLM（指定 95% 命中率）
./_scripts/cache_bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan -P disagg \
  -R 95
```

> **测试数据集**：与纯 Decode 同族——SGLang `generated-shared-prefix`（GSP）/ vLLM `prefix_repetition`：共享前缀长度 = 输入长度 × 命中率（`-R`），每请求唯一后缀 = 输入长度 − 前缀（随机生成、各不相同，为未命中部分）。组数由 `-g` 控制（默认 1；预热每组 1 条填前缀）。

### 参数说明

| 参数 | 短写 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--pd` | `-P` | ✅ | — | PD部署模式: `agg`(非PD分离) 或 `disagg`(PD分离) |
| `--report-dir` | `-r` | | `./{framework}_cache_reports` | 报告输出目录 |
| `--concurrency` | `-c` | | `1,8,32,64,128` | 并发数列表（测试请求数 N_total = 并发数 × 并发倍数，`--concurrency-multiplier` 默认 5；另有预热 = 组数条请求，不计量） |
| `--gsp-num-groups` | `-g` | | `1` | 共享前缀组数（组内请求共享该组前缀，组间不同；预热每组 1 条，测试每组条数 = ceil(并发倍数×并发数/组数)） |
| `--seed` | | | `123` | 随机数据集种子（预热与测试共享前缀依赖固定 seed） |
| `--concurrency-multiplier` | | | `5` | 并发倍数：测试请求数 N_total = 并发数 × 此值（预热不受影响） |
| `--io-combinations` | `-i` | | `65536 512,131072 1024,262144 1024` | IO 组合（逗号分隔，每组 `"输入 输出"`；共享前缀/唯一后缀按各组输入长度计算） |
| `--cache-hit-rate` | `-R` | | `90` | 缓存命中率百分比（1~99），前缀级命中 |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--describe` | `-d` | | — | 模型描述信息（可选，填入报告末尾单独章节） |
| `--foreground` | `-f` | | 后台 | 前台执行 |
| `--timeout` | | | `3600` | 每组测试超时秒数（超时自动终止并跳过，继续下一组） |

### 产物

- 预热日志（不计量）：`{report-dir}/{tester}/{model_name}/{TS}/cache_bs-{concurrency}_hit-{rate}_input-{in}_warmup.log`
- 测试日志（计量）：`{report-dir}/{tester}/{model_name}/{TS}/cache_bs-{concurrency}_hit-{rate}_input-{in}-output-{out}.log`
- CSV：`{report-dir}/{tester}/{model_name}/{TS}/cache_results.csv`（仅收集测试日志，自动跳过预热日志）
- Markdown 报告：`{report-dir}/{tester}/{model_name}/{TS}/{tester}_{model_name}_{chip_type}_{framework}_{pd}_cache_{TS}.md`

### 关注指标

- **输入/输出/总吞吐量** = `Input/Output/Total token throughput (tok/s)`
- **请求吞吐量** = `Request throughput (req/s)`
- **TTFT** = `Mean/Median/P90/P95/P99 TTFT (ms)`（命中率越高，每请求 prefill 量越少，TTFT 越低且分布均匀）
- **TPOT** = `Mean/Median/P90/P95/P99 TPOT (ms)`
- SGLang 从 `/metrics` 提取 HiCache L1/L2/L3 三层命中率（`--cache-report` 仅覆盖 L1/radix，可对照验证）
- vLLM 从 `/metrics` 提取 L1（GPU）/L3（external/LM Cache）两层命中率；`/metrics` 不可用时 L1 回退到 `-R` 指定值，L3 显示 N/A（SGLang 各层无数据均显示 N/A）

---

## 自动产物流程

所有脚本执行完毕后会自动完成以下流程，无需手动操作：

```
bench 工具日志 → collect_results.py → CSV → csv_to_md.py → Markdown 报告
```

Markdown 报告**标题括号注明测试类型**（由各脚本自动传入：基础 Benchmark 测试 / 纯 Prefill 测试 / 纯 Decode 测试 / 指定缓存命中率测试，不显示脚本名），正文分三部分：

| 部分 | 内容来源 |
|---|---|
| 一、测试结果 | CSV 自动转换（前 4 列相同值留空合并；输入/输出长度成对显示——输入长度有值时输出长度必有值） |
| 二、模型服务启动命令 | 项目根目录 `serve_command.sh`（从模板复制） |
| 三、Benchmark 测试命令 | 脚本自动记录实际执行的 bench 工具命令 |

### 容错与超时机制

所有脚本内置三重保障，确保即使部分测试失败也能继续执行并生成报告：

| 保障层级 | 机制 | 触发条件 | 行为 |
|---|---|---|---|
| 1. 退出码检测 | `set +e` 捕获非零退出码 | bench 命令崩溃（如 Python traceback、连接拒绝） | 打印错误信息，跳过该组，继续下一组 |
| 2. 日志错误扫描 | `check_log_errors` 函数 | 退出码为 0 但日志含错误模式（`All requests failed`、`Traceback`、`ConnectionError` 等） | 标记为失败，跳过该组，继续下一组 |
| 3. 超时保护 | `timeout $TIMEOUT` 包裹 | bench 命令卡死不退出（返回码 124） | 超时终止，跳过该组，继续下一组 |

- 超时时间通过 `--timeout` 参数指定，默认 `3600` 秒（1 小时），可按需调整
- 测试结束后无论是否有失败，都会执行 `collect_results.py` + `csv_to_md.py` 生成报告
- 已成功的日志数据会被完整收集，失败组的数据不会出现在 CSV 中

### 手动收集结果

如需从已有日志手动汇总为 CSV（例如自动流程中断、或只跑了日志没生成 CSV）：

```bash
python3 _scripts/collect_results.py \
  --report-dir ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022 \
  --model-name glm-5.2-fp8 \
  --framework sglang \
  --chip-type H100 \
  --out ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/results.csv
```

脚本会递归扫描 `--report-dir` 下所有 `bench_*/*.log`、`prefill_*/*.log`、`decode_*/*.log`、`cache_*/*.log`，按文件名解析测试类型与参数，提取吞吐量、TTFT、TPOT 等指标汇总为 CSV。非测试日志（如总日志）会自动跳过。

| 参数 | 必选 | 默认值 | 说明 |
|---|---|---|---|
| `--report-dir` | ✅ | — | bench 日志根目录（递归扫描子目录） |
| `--model-name` | ✅ | — | 模型名称（填入 CSV 的模型名称列；会自动取路径最后一段并去除 `:` `\`） |
| `--out` | ✅ | — | 输出 CSV 文件路径 |
| `--framework` | | `""` | 推理框架（填入 CSV 的推理框架列，如 `sglang` / `vllm`） |
| `--chip-type` | | `""` | 芯片类型，用于 CSV 列名后缀（如 `H100` / `B200`） |

> 注意：所有参数名均使用连字符（`--report-dir`），使用下划线（`--report_dir`）会被 argparse 判定为未识别参数导致报错。

### 手动生成报告

如需从已有 CSV 重新生成报告：

```bash
python3 _scripts/csv_to_md.py \
  --csv ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/results.csv \
  --output ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/zhangsan_glm-5.2-fp8_H100_sglang_agg_bench_20260831_143022.md \
  --test-type "基础 Benchmark 测试" \
  --bench-command "python -m sglang.benchmark.serving --backend sglang-oai-chat ..."
```

---

## 快速参考：默认测试矩阵

四个脚本默认参数下实际执行的测试矩阵（`-c`/`-i`/`-p` 可自定义；bench / prefill 按输入长度限并发上限，超出档位自动跳过）：

| 脚本 | 输入/输出长度 → 并发数档位 | 组数 |
|---|---|---|
| **基础 Benchmark**<br>`bench.sh` | • `8192 512`、`32768 512`、`65536 512` → `1, 8, 32, 64, 128`<br>• `131072 1024`、`262144 1024` → `1, 8, 32, 64`<br>• `512000 1024` → `1, 8, 32`<br>• `1000000 1024` → `1, 8` | 5+5+5+4+4+3+2 = **28** |
| **纯 Prefill**<br>`prefill_bench.sh` | • `65536 1` → `1, 8, 32, 64, 128`<br>• `131072 1`、`262144 1` → `1, 8, 32, 64` | 5+4+4 = **13** |
| **纯 Decode**<br>`decode_bench.sh` | • 前缀 `8192`、`32768`、`65536`（输出 512）→ `1, 8, 32, 64, 128`<br>• 前缀 `131072`、`262144`（输出 1024）→ `1, 8, 32, 64, 128` | 5 前缀 × 5 档 = **25** |
| **指定缓存命中率**<br>`cache_bench.sh` | • `65536 512`、`131072 1024`、`262144 1024` → `1, 8, 32, 64, 128` | 3 组合 × 5 档 = **15** |

- 并发上限（超出档位自动跳过）：bench ≤64K→128、128K~256K→64、512K→32、1M→16；prefill ≤64K→128、128K~256K→64；decode / cache 无上限
- 请求数 = 并发数 × 并发倍数（倍数默认：bench 1 / prefill 5 / decode 10 / cache 5，`--concurrency-multiplier` 可自定义）；cache 另有预热 = 组数条请求，不计量

---

## 快速参考：四种测试对比

| | 基础 Benchmark | 纯 Prefill | 纯 Decode | 指定缓存命中率 |
|---|---|---|---|---|
| **脚本** | `bench.sh` | `prefill_bench.sh` | `decode_bench.sh` | `cache_bench.sh` |
| **报告标题** | （基础 Benchmark 测试） | （纯 Prefill 测试） | （纯 Decode 测试） | （指定缓存命中率测试） |
| **测试目标** | 端到端性能 | prefill 前向算力 | decode 迭代性能 | 指定前缀命中率下推理性能 |
| **output_len** | 512（≤64K 组）/ 1024（128K~1M 组） | 1（固定） | 512（≤64K 组）/ 1024（128K~256K 组） | 512（64K 组）/ 1024（128K~256K 组） |
| **前缀缓存** | **关** | **关** | **开** | **开** |
| **数据集** | `random-ids`/`random`（随机，无共享前缀） | `random-ids`/`random`（随机，无共享前缀） | `generated-shared-prefix`/`prefix_repetition`（共享前缀） | `generated-shared-prefix`/`prefix_repetition`（共享前缀） |
| **num_prompts** | = 并发数×倍数（默认 1） | = 并发数×倍数（默认 5） | = 并发数×倍数（默认 10） | N_total = 并发数×倍数（默认 5）（另预热 = 组数条请求，不计量；组数 `-g` 默认 1） |
| **--seed** | `123`（`--seed` 可配） | `123`（`--seed` 可配） | `123`（`--seed` 可配） | `123`（`--seed` 可配） |
| **核心指标** | 输入/输出/总吞吐量、TTFT、TPOT | prefill 吞吐量、TTFT | decode 吞吐量、TPOT | 输入/输出/总吞吐量、TTFT、TPOT |
| **命中率列** | 实测（关缓存时应为 0%） | 实测（关缓存时应为 ≈0%，无数据记 N/A） | 实测（缓存命中应为 ≈100%，无数据记 N/A） | 实测（SGLang `/metrics` 三层 L1/L2/L3 / vLLM `/metrics` 两层 L1/L3），vLLM L1 无数据回退 `-R`，SGLang 各层无数据显示 N/A |
| **超时保护** | `--timeout`（默认 3600s） | 同左 | 同左 | 同左 |

> 四种脚本均支持 `--timeout`（默认 3600 秒）、容错跳过、失败后继续执行并生成部分报告。详见各脚本 `--help`。

详见 [Model_Inference_Benchmark_TestStrategy.md](Model_Inference_Benchmark_TestStrategy.md) 完整测试方案文档。
