# 推理性能 Benchmark 测试报告

> 本报告由 `csv_to_md.py` 自动生成，请勿手动编辑。

---

## 一、测试结果

<!-- CSV_TABLE_START -->
| 模型名称 | 推理框架 | 输入长度 | 输出长度 | 并发数 | 输入token吞吐量_H100 (toks/s) | 输出token吞吐量_H100 (toks/s) | 总token吞吐量_H100 (toks/s) | Mean TTFT_H100 (ms) | P99 TTFT_H100 (ms) | Mean TPOT_H100 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.2-fp8 | sglang | 2048 | 512 | 1 | 152.79 | 95.49 | 248.28 | 321.42 | 1181.54 | 9.25 |
| | | | | 4 | 580.16 | 362.60 | 942.76 | 185.33 | 612.47 | 8.41 |
| | | | | 8 | 1120.44 | 700.28 | 1820.72 | 142.86 | 498.20 | 8.05 |
| | | | | 16 | 1985.67 | 1241.04 | 3226.71 | 128.45 | 430.15 | 7.92 |
| | | | | 32 | 3210.88 | 2005.55 | 5216.43 | 135.67 | 512.33 | 7.88 |
| | | | | 64 | 3850.12 | 2404.32 | 6254.44 | 198.44 | 710.28 | 8.12 |
| | | | | 128 | 4021.55 | 2511.22 | 6532.77 | 345.67 | 1280.44 | 8.55 |
| | | 8192 | 1024 | 1 | 152.64 | 152.64 | 305.28 | 521.44 | 1820.55 | 9.25 |
| | | | | 4 | 588.22 | 588.22 | 1176.44 | 280.33 | 980.12 | 8.41 |
| | | | | 8 | 1140.55 | 1140.55 | 2281.10 | 220.44 | 810.33 | 8.05 |
| | | | | 16 | 2050.33 | 2050.33 | 4100.66 | 245.67 | 920.44 | 7.92 |
| | | | | 32 | 3360.88 | 3360.88 | 6721.76 | 340.55 | 1280.33 | 7.88 |
| | | | | 64 | 4050.22 | 4050.22 | 8100.44 | 520.33 | 1980.55 | 8.12 |
| | | | | 128 | 4220.11 | 4220.11 | 8440.22 | 980.44 | 3520.88 | 8.55 |
| | | 32768 | 1024 | 1 | 150.44 | 150.44 | 300.88 | 1820.55 | 6520.44 | 9.25 |
| | | | | 4 | 560.22 | 560.22 | 1120.44 | 980.33 | 3520.88 | 8.41 |
| | | | | 8 | 1080.44 | 1080.44 | 2160.88 | 820.44 | 2980.55 | 8.05 |
| | | | | 16 | 1920.66 | 1920.66 | 3841.32 | 950.22 | 3450.11 | 7.92 |
| | | | | 32 | 3100.88 | 3100.88 | 6201.76 | 1280.44 | 4820.33 | 7.88 |
| | | | | 64 | 3680.22 | 3680.22 | 7360.44 | 1980.55 | 7520.88 | 8.12 |
| | | | | 128 | 3850.11 | 3850.11 | 7700.22 | 3520.44 | 12800.55 | 8.55 |
| | | 65536 | 1024 | 1 | 145.22 | 145.22 | 290.44 | 3520.88 | 12800.55 | 9.25 |
| | | | | 4 | 540.33 | 540.33 | 1080.66 | 1980.44 | 7120.88 | 8.41 |
| | | | | 8 | 1020.44 | 1020.44 | 2040.88 | 1620.55 | 5980.33 | 8.05 |
| | | | | 16 | 1820.66 | 1820.66 | 3641.32 | 1820.33 | 6720.44 | 7.92 |
| | | | | 32 | 2900.88 | 2900.88 | 5801.76 | 2520.44 | 9500.55 | 7.88 |
| | | | | 64 | 3450.22 | 3450.22 | 6900.44 | 3820.55 | 14800.88 | 8.12 |
| | | | | 128 | 3620.11 | 3620.11 | 7240.22 | 6820.44 | 25200.55 | 8.55 |
<!-- CSV_TABLE_END -->

---

## 二、模型服务启动命令

> 以下内容自动从 `serve_command.txt` 读取（从 `serve_command_template.txt` 复制并填写）。

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
python -m sglang.bench_serving \
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
python -m sglang.bench_serving \
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
