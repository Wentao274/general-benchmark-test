# 推理性能 Benchmark 测试报告

> 本报告由 `csv_to_md.py` 自动生成，请勿手动编辑。

---

## 一、测试结果

<!-- CSV_TABLE_START -->
| 模型名称 | 推理框架 | 输入长度 | 输出长度 | 并发数 | prefix长度 | 输入token吞吐量_H100 (toks/s) | 输出token吞吐量_H100 (toks/s) | 总token吞吐量_H100 (toks/s) | Mean TTFT_H100 (ms) | P99 TTFT_H100 (ms) | Mean TPOT_H100 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.2-fp8 | sglang | 8192 | 512 | 1 | 0 | 152.64 | 95.40 | 248.04 | 521.44 | 1820.55 | 9.25 |
| | | | | 4 | 0 | 588.22 | 367.64 | 955.86 | 280.33 | 980.12 | 8.41 |
| | | | | 8 | 0 | 1140.55 | 712.84 | 1853.39 | 220.44 | 810.33 | 8.05 |
| | | | | 16 | 0 | 2050.33 | 1281.46 | 3331.79 | 245.67 | 920.44 | 7.92 |
| | | | | 32 | 0 | 3360.88 | 2100.55 | 5461.43 | 340.55 | 1280.33 | 7.88 |
| | | | | 64 | 0 | 4050.22 | 2531.39 | 6581.61 | 520.33 | 1980.55 | 8.12 |
| | | | | 128 | 0 | 4220.11 | 2637.57 | 6857.68 | 980.44 | 3520.88 | 8.55 |
| | | 32768 | 512 | 1 | 0 | 150.44 | 93.78 | 244.22 | 1820.55 | 6520.44 | 9.25 |
| | | | | 4 | 0 | 560.22 | 350.14 | 910.36 | 980.33 | 3520.88 | 8.41 |
| | | | | 8 | 0 | 1080.44 | 675.28 | 1755.72 | 820.44 | 2980.55 | 8.05 |
| | | | | 16 | 0 | 1920.66 | 1200.41 | 3121.07 | 950.22 | 3450.11 | 7.92 |
| | | | | 32 | 0 | 3100.88 | 1938.05 | 5038.93 | 1280.44 | 4820.33 | 7.88 |
| | | | | 64 | 0 | 3680.22 | 2300.14 | 5980.36 | 1980.55 | 7520.88 | 8.12 |
| | | | | 128 | 0 | 3850.11 | 2406.32 | 6256.43 | 3520.44 | 12800.55 | 8.55 |
| | | 65536 | 512 | 1 | 0 | 145.22 | 72.61 | 217.83 | 3520.88 | 12800.55 | 9.25 |
| | | | | 4 | 0 | 540.33 | 270.17 | 810.50 | 1980.44 | 7120.88 | 8.41 |
| | | | | 8 | 0 | 1020.44 | 510.22 | 1530.66 | 1620.55 | 5980.33 | 8.05 |
| | | | | 16 | 0 | 1820.66 | 910.33 | 2730.99 | 1820.33 | 6720.44 | 7.92 |
| | | | | 32 | 0 | 2900.88 | 1450.44 | 4351.32 | 2520.44 | 9500.55 | 7.88 |
| | | | | 64 | 0 | 3450.22 | 1725.11 | 5175.33 | 3820.55 | 14800.88 | 8.12 |
| | | | | 128 | 0 | 3620.11 | 1810.06 | 5430.17 | 6820.44 | 25200.55 | 8.55 |
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
  --metric_percentiles 95,99 \
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
  --metric_percentiles 95,99 \
  --ready-check-timeout-sec 30
```
<!-- BENCH_COMMAND_END -->
