# 推理性能 Benchmark 测试报告（基础 Benchmark 测试）

> 本报告由 `csv_to_md.py` 自动生成，请勿手动编辑。

---

## 一、测试结果

<!-- CSV_TABLE_START -->
| 模型名称 | 推理框架 | 输入长度 | 输出长度 | 前缀长度 | 并发数 | 请求数 | L1命中率 | L2命中率 | L3命中率 | 输入token吞吐量_H100 (toks/s) | 输出token吞吐量_H100 (toks/s) | 总token吞吐量_H100 (toks/s) | Mean TTFT_H100 (ms) | Median TTFT_H100 (ms) | P90 TTFT_H100 (ms) | P95 TTFT_H100 (ms) | P99 TTFT_H100 (ms) | Mean TPOT_H100 (ms) | Median TPOT_H100 (ms) | P90 TPOT_H100 (ms) | P95 TPOT_H100 (ms) | P99 TPOT_H100 (ms) | 请求吞吐量_H100 (req/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.2-fp8 | sglang | 8192 | 512 | — | 1 | 1 | 0.0% | 0.0% | 0.0% | 152.64 | 95.40 | 248.04 | 521.44 | 495.37 | 1420.03 | 1602.08 | 1820.55 | 9.25 | 9.06 | 9.53 | 9.8 | 10.27 | 0.19 |
|  |  |  |  | — | 4 | 4 | 0.0% | 0.0% | 0.0% | 588.22 | 367.64 | 955.86 | 280.33 | 266.31 | 764.49 | 862.51 | 980.12 | 8.41 | 8.24 | 8.66 | 8.91 | 9.34 | 0.72 |
|  |  |  |  | — | 8 | 8 | 0.0% | 0.0% | 0.0% | 1140.55 | 712.84 | 1853.39 | 220.44 | 209.42 | 632.06 | 713.09 | 810.33 | 8.05 | 7.89 | 8.29 | 8.53 | 8.94 | 1.39 |
|  |  |  |  | — | 16 | 16 | 0.0% | 0.0% | 0.0% | 2050.33 | 1281.46 | 3331.79 | 245.67 | 233.39 | 717.94 | 809.99 | 920.44 | 7.92 | 7.76 | 8.16 | 8.4 | 8.79 | 2.5 |
|  |  |  |  | — | 32 | 32 | 0.0% | 0.0% | 0.0% | 3360.88 | 2100.55 | 5461.43 | 340.55 | 323.52 | 998.66 | 1126.69 | 1280.33 | 7.88 | 7.72 | 8.12 | 8.35 | 8.75 | 4.1 |
|  |  |  |  | — | 64 | 64 | 0.0% | 0.0% | 0.0% | 4050.22 | 2531.39 | 6581.61 | 520.33 | 494.31 | 1544.83 | 1742.88 | 1980.55 | 8.12 | 7.96 | 8.36 | 8.61 | 9.01 | 4.94 |
|  |  |  |  | — | 128 | 128 | 0.0% | 0.0% | 0.0% | 4220.11 | 2637.57 | 6857.68 | 980.44 | 931.42 | 2746.29 | 3098.37 | 3520.88 | 8.55 | 8.38 | 8.81 | 9.06 | 9.49 | 5.15 |
|  |  | 32768 | 512 | — | 1 | 1 | 0.0% | 0.0% | 0.0% | 150.44 | 93.78 | 244.22 | 1820.55 | 1729.52 | 5085.94 | 5737.99 | 6520.44 | 9.25 | 9.06 | 9.53 | 9.8 | 10.27 | 0.18 |
|  |  |  |  | — | 4 | 4 | 0.0% | 0.0% | 0.0% | 560.22 | 350.14 | 910.36 | 980.33 | 931.31 | 2746.29 | 3098.37 | 3520.88 | 8.41 | 8.24 | 8.66 | 8.91 | 9.34 | 0.68 |
|  |  |  |  | — | 8 | 8 | 0.0% | 0.0% | 0.0% | 1080.44 | 675.28 | 1755.72 | 820.44 | 779.42 | 2324.83 | 2622.88 | 2980.55 | 8.05 | 7.89 | 8.29 | 8.53 | 8.94 | 1.32 |
|  |  |  |  | — | 16 | 16 | 0.0% | 0.0% | 0.0% | 1920.66 | 1200.41 | 3121.07 | 950.22 | 902.71 | 2691.09 | 3036.1 | 3450.11 | 7.92 | 7.76 | 8.16 | 8.4 | 8.79 | 2.34 |
|  |  |  |  | — | 32 | 32 | 0.0% | 0.0% | 0.0% | 3100.88 | 1938.05 | 5038.93 | 1280.44 | 1216.42 | 3759.86 | 4241.89 | 4820.33 | 7.88 | 7.72 | 8.12 | 8.35 | 8.75 | 3.79 |
|  |  |  |  | — | 64 | 64 | 0.0% | 0.0% | 0.0% | 3680.22 | 2300.14 | 5980.36 | 1980.55 | 1881.52 | 5866.29 | 6618.37 | 7520.88 | 8.12 | 7.96 | 8.36 | 8.61 | 9.01 | 4.49 |
|  |  |  |  | — | 128 | 128 | 0.0% | 0.0% | 0.0% | 3850.11 | 2406.32 | 6256.43 | 3520.44 | 3344.42 | 9984.43 | 11264.48 | 12800.55 | 8.55 | 8.38 | 8.81 | 9.06 | 9.49 | 4.7 |
|  |  | 65536 | 512 | — | 1 | 1 | 0.0% | 0.0% | 0.0% | 145.22 | 72.61 | 217.83 | 3520.88 | 3344.84 | 9984.43 | 11264.48 | 12800.55 | 9.25 | 9.06 | 9.53 | 9.8 | 10.27 | 0.14 |
|  |  |  |  | — | 4 | 4 | 0.0% | 0.0% | 0.0% | 540.33 | 270.17 | 810.50 | 1980.44 | 1881.42 | 5554.29 | 6266.37 | 7120.88 | 8.41 | 8.24 | 8.66 | 8.91 | 9.34 | 0.53 |
|  |  |  |  | — | 8 | 8 | 0.0% | 0.0% | 0.0% | 1020.44 | 510.22 | 1530.66 | 1620.55 | 1539.52 | 4664.66 | 5262.69 | 5980.33 | 8.05 | 7.89 | 8.29 | 8.53 | 8.94 | 1.0 |
|  |  |  |  | — | 16 | 16 | 0.0% | 0.0% | 0.0% | 1820.66 | 910.33 | 2730.99 | 1820.33 | 1729.31 | 5241.94 | 5913.99 | 6720.44 | 7.92 | 7.76 | 8.16 | 8.4 | 8.79 | 1.78 |
|  |  |  |  | — | 32 | 32 | 0.0% | 0.0% | 0.0% | 2900.88 | 1450.44 | 4351.32 | 2520.44 | 2394.42 | 7410.43 | 8360.48 | 9500.55 | 7.88 | 7.72 | 8.12 | 8.35 | 8.75 | 2.83 |
|  |  |  |  | — | 64 | 64 | 0.0% | 0.0% | 0.0% | 3450.22 | 1725.11 | 5175.33 | 3820.55 | 3629.52 | 11544.69 | 13024.77 | 14800.88 | 8.12 | 7.96 | 8.36 | 8.61 | 9.01 | 3.37 |
|  |  |  |  | — | 128 | 128 | 0.0% | 0.0% | 0.0% | 3620.11 | 1810.06 | 5430.17 | 6820.44 | 6479.42 | 19656.43 | 22176.48 | 25200.55 | 8.55 | 8.38 | 8.81 | 9.06 | 9.49 | 3.54 |

> vLLM 报告含 `L1命中率` 和 `L3命中率` 两列（无 L2），表头为：
> `| 模型名称 | 推理框架 | 输入长度 | 输出长度 | 前缀长度 | 并发数 | 请求数 | L1命中率 | L3命中率 | 输入token吞吐量_H100 (toks/s) | ... |`
<!-- CSV_TABLE_END -->

---

## 二、模型服务启动命令

> 以下内容自动从 `serve_command.sh` 读取（从 `serve_command.sh.template` 复制并填写）。

<!-- SERVE_COMMAND_START -->
```bash
# SGLang 部署命令
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  --host 0.0.0.0 \
  --port 8080 \
  --context-length 202752 \
  --max-running-requests 64 \
  --chunked-prefill-size 16384 \
  --mem-fraction-static 0.9 \
  --tp 8 \
  --dp 1 \
  --pp 1 \
  --ep 1 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```
<!-- SERVE_COMMAND_END -->

---

## 三、Benchmark 测试命令

> 以下为实际执行的 bench 工具命令，由程序自动记录（参数值用变量表示，实际运行时按测试矩阵逐组替换）。

<!-- BENCH_COMMAND_START -->
```bash
# SGLang Benchmark 命令（第1章 / 第2章纯Prefill 通用）
python -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name random-ids \
  --random-input-len $INPUT_LEN \
  --random-output-len $OUTPUT_LEN \
  --random-range-ratio 1.0 \
  --num-prompts $NUM_PROMPTS \
  --max-concurrency $CONCURRENCY \
  --seed 123

# SGLang 纯Decode 命令（第2章，generated-shared-prefix 数据集）
python -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group $NUM_PROMPTS \
  --gsp-system-prompt-len $PREFIX_LEN \
  --gsp-question-len 1 \
  --gsp-output-len $OUTPUT_LEN \
  --num-prompts $NUM_PROMPTS \
  --max-concurrency $CONCURRENCY \
  --seed 123

# SGLang 指定缓存命中率命令（第3章，前缀级命中，两阶段：预热+测试）
# 阶段1：预热（串行，填共享前缀入缓存，不计量）
python -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 1 \
  --gsp-system-prompt-len $PREFIX_LEN \
  --gsp-question-len $SUFFIX_LEN \
  --gsp-output-len 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 123
# 阶段2：测试（目标并发，计量）
python -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group $N_TOTAL \
  --gsp-system-prompt-len $PREFIX_LEN \
  --gsp-question-len $SUFFIX_LEN \
  --gsp-output-len $OUTPUT_LEN \
  --num-prompts $N_TOTAL \
  --max-concurrency $CONCURRENCY \
  --seed 123 \
  --cache-report

# vLLM Benchmark 命令（第1章 / 第2章纯Prefill 通用）
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name random \
  --random-input-len $INPUT_LEN \
  --random-output-len $OUTPUT_LEN \
  --num-prompts $NUM_PROMPTS \
  --max-concurrency $CONCURRENCY \
  --trust-remote-code \
  --temperature 0.7 \
  --random-range-ratio 0.0 \
  --random-prefix-len 0 \
  --seed 123 \
  --metric_percentiles 90,95,99 \
  --ready-check-timeout-sec 30

# vLLM 纯Decode 命令（第2章，prefix_repetition 数据集）
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len $PREFIX_LEN \
  --prefix-repetition-suffix-len 1 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len $OUTPUT_LEN \
  --num-prompts $NUM_PROMPTS \
  --max-concurrency $CONCURRENCY \
  --trust-remote-code \
  --temperature 0.7 \
  --seed 123 \
  --metric_percentiles 90,95,99 \
  --ready-check-timeout-sec 30

# vLLM 指定缓存命中率命令（第3章，前缀级命中，两阶段：预热+测试）
# 阶段1：预热（串行，填共享前缀入缓存，不计量）
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len $PREFIX_LEN \
  --prefix-repetition-suffix-len $SUFFIX_LEN \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --trust-remote-code \
  --temperature 0.7 \
  --seed 123 \
  --metric_percentiles 90,95,99 \
  --ready-check-timeout-sec 30
# 阶段2：测试（目标并发，计量）
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len $PREFIX_LEN \
  --prefix-repetition-suffix-len $SUFFIX_LEN \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len $OUTPUT_LEN \
  --num-prompts $N_TOTAL \
  --max-concurrency $CONCURRENCY \
  --trust-remote-code \
  --temperature 0.7 \
  --seed 123 \
  --metric_percentiles 90,95,99 \
  --ready-check-timeout-sec 30
```
<!-- BENCH_COMMAND_END -->

---

## 四、模型描述

<!-- DESCRIBE_START -->
GLM-5.2 FP8 量化版本，8 卡 TP 部署，context length 202752，mem-fraction-static 0.9。
<!-- DESCRIBE_END -->
