# general-benchmark-test

模型推理性能基准测试套件，支持 SGLang / vLLM 等推理框架。

## 目录结构

```
general-benchmark-test/
├── Model_Inference_Benchmark_TestStrategy.md   完整测试方案文档
├── benchmark_analysis_template.md              Markdown 报告模板
├── serve_command_template.txt                   模型服务启动命令模板（需复制为 serve_command.txt）
└── _scripts/
    ├── bench.sh                                基础 Benchmark 脚本
    ├── prefill_bench.sh                        纯 Prefill 测试脚本
    ├── decode_bench.sh                         纯 Decode 测试脚本
    ├── decode_http_sweep.py                    纯 Decode HTTP 精确测量【仅供参考，实际不使用】
    ├── collect_results.py                      日志收集 → CSV
    └── csv_to_md.py                            CSV → Markdown 报告
```

## 前置条件

1. 推理服务已启动并可访问
2. 复制 `serve_command_template.txt` 为 `serve_command.txt`，填写真实的模型服务启动命令
   ```bash
   cp serve_command_template.txt serve_command.txt
   ```
   > 每个测试脚本在执行前会校验 `serve_command.txt`：文件不存在或内容为空都会报错中止
3. 所有脚本默认后台执行，加 `-f` 切前台

---

## 1. 基础 Benchmark 测试

测量端到端推理性能：输入/输出/总 token 吞吐量、TTFT、TPOT。

### 服务端要求

无特殊要求，前缀缓存开/关均可。

### 执行命令

```bash
# SGLang
./_scripts/bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan

# vLLM
./_scripts/bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan
```

### 自定义参数

```bash
# 自定义并发数和 IO 组合
./_scripts/bench.sh -F sglang \
  -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 \
  -c 1,8,32,128 \
  -i "2048 512,8192 1024" \
  -T zhangsan

# 前台执行 + 自定义报告目录
./_scripts/bench.sh -F vllm -f \
  -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 \
  -r /mnt/results/vllm_bench -s 30 -T zhangsan
```

### 参数说明

| 参数 | 短写 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--concurrency` | `-c` | | `1,4,8,16,32,64,128` | 并发数列表 |
| `--io-combinations` | `-i` | | `2048 512,8192 1024,32768 1024,65536 1024` | IO 组合（逗号分隔，每组 `"输入 输出"`） |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀，如 `H100`） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--report-dir` | `-r` | | `./{framework}_reports` | 报告输出目录 |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--foreground` | `-f` | | 后台 | 前台执行 |

### 产物

每次测试单独创建带时间戳的子目录 `{TS}`（`YYYYMMDD_HHMMSS`），重复执行不会覆盖：

- 日志：`{report-dir}/{tester}/{model_name}/{TS}/input_len-{in}-output_len-{out}-bs-{concurrency}.log`
- CSV：`{report-dir}/{tester}/{model_name}/{TS}/results.csv`
- Markdown 报告：`{report-dir}/{tester}/{model_name}/{TS}/{tester}_{model_name}_{chip_type}_bench_{TS}.md`

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
  -n glm-5.2-fp8 -t H100 -T zhangsan

# vLLM 纯 Prefill
./_scripts/prefill_bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan
```

### 自定义参数

```bash
# 自定义输入长度和并发
./_scripts/prefill_bench.sh -F sglang \
  -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 \
  -c 1,8,64 -i "32768 1,65536 1" \
  -T zhangsan
```

### 参数说明

| 参数 | 短写 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--io-combinations` | `-i` | | `65536 1` | IO 组合（逗号分隔，每组 `"输入 输出"`） |
| `--concurrency` | `-c` | | `1,4,8,16,32,64,128` | 并发数列表 |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--report-dir` | `-r` | | `./{framework}_prefill_reports` | 报告输出目录 |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--foreground` | `-f` | | 后台 | 前台执行 |

### 产物

- 日志：`{report-dir}/{tester}/{model_name}/{TS}/prefill_input-{in}-bs-{concurrency}.log`
- CSV：`{report-dir}/{tester}/{model_name}/{TS}/prefill_results.csv`
- Markdown 报告：`{report-dir}/{tester}/{model_name}/{TS}/{tester}_{model_name}_{chip_type}_prefill_{TS}.md`

### 关注指标

- **prefill 吞吐量** = 输入 token 吞吐量（SGLang 直接提取 `Input token throughput`；vLLM 用 `总吞吐量 - 输出吞吐量` 计算）
- **TTFT** ≈ 端到端时间（无 decode，首 token 延迟即 prefill 耗时）
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
  -p 4096,32768,65536 \
  -o 1024 \
  -c 1,4,8,16,32,64,128 \
  -T zhangsan

# vLLM 纯 Decode（使用 prefix_repetition 数据集）
./_scripts/decode_bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 4096,32768,65536 \
  -o 1024 \
  -c 1,4,8,16,32,64,128 \
  -T zhangsan
```

### 自定义参数

```bash
# 自定义前缀、输出和并发
./_scripts/decode_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 8192,65536 -o 512 -c 1,16,128 \
  -T zhangsan
```

### 参数说明

| 参数 | 短写 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--prefix-lens` | `-p` | | `4096,32768,65536` | 前缀长度列表 |
| `--output-lens` | `-o` | | `1024` | 输出长度列表 |
| `--concurrency` | `-c` | | `1,4,8,16,32,64,128` | 并发数列表（num_prompts = 2×并发） |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--report-dir` | `-r` | | `./{framework}_decode_reports` | 报告输出目录 |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--foreground` | `-f` | | 后台 | 前台执行 |

### 产物

- 日志：`{report-dir}/{tester}/{model_name}/{TS}/decode_prefix-{prefix}-output-{out}-bs-{concurrency}.log`
- CSV：`{report-dir}/{tester}/{model_name}/{TS}/decode_results.csv`
- Markdown 报告：`{report-dir}/{tester}/{model_name}/{TS}/{tester}_{model_name}_{chip_type}_decode_{TS}.md`

### 关注指标

- **decode 吞吐量** = `Output token throughput (tok/s)`
- **Mean TPOT** = `Mean TPOT (ms)`
- **Mean TTFT** = `Mean TTFT (ms)`（应很小，若很大说明缓存未生效）

> num_prompts = 2 × 并发数，用于稀释首个请求的 prefill 开销（bench 工具无预热机制）。

---

## 自动产物流程

所有脚本执行完毕后会自动完成以下流程，无需手动操作：

```
bench 工具日志 → collect_results.py → CSV → csv_to_md.py → Markdown 报告
```

Markdown 报告分三部分：

| 部分 | 内容来源 |
|---|---|
| 一、测试结果 | CSV 自动转换（前 4 列相同值留空合并） |
| 二、模型服务启动命令 | 项目根目录 `serve_command.txt`（从模板复制） |
| 三、Benchmark 测试命令 | 脚本自动记录实际执行的 bench 工具命令 |

### 手动生成报告

如需从已有 CSV 重新生成报告：

```bash
python3 _scripts/csv_to_md.py \
  --csv ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/results.csv \
  --output ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/zhangsan_glm-5.2-fp8_H100_bench_20260831_143022.md \
  --bench-command "python -m sglang.bench_serving --backend sglang-oai-chat ..."
```

---

## 快速参考：三种测试对比

| | 基础 Benchmark | 纯 Prefill | 纯 Decode |
|---|---|---|---|
| **脚本** | `bench.sh` | `prefill_bench.sh` | `decode_bench.sh` |
| **测试目标** | 端到端性能 | prefill 前向算力 | decode 迭代性能 |
| **output_len** | 512/1024 | 1（固定） | 1024 |
| **前缀缓存** | 开/关均可 | **关** | **开** |
| **数据集** | `random-ids`/`random` | 同左 | `generated-shared-prefix`/`prefix_repetition` |
| **num_prompts** | = 并发数 | = 并发数 | = 2×并发数 |
| **核心指标** | 输入/输出/总吞吐量、TTFT、TPOT | prefill 吞吐量、TTFT | decode 吞吐量、TPOT |

详见 [Model_Inference_Benchmark_TestStrategy.md](Model_Inference_Benchmark_TestStrategy.md) 完整测试方案文档。
