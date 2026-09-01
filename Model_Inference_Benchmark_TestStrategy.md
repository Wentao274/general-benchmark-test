# 模型推理性能 Benchmark 测试方案

> 创建：2026-08-27
> 目的：为任意推理框架（SGLang / vLLM 等）提供**可比、可复现**的推理性能基准测试方案。
> 脚本位于 `_scripts/`，产物落各自 `*_reports/`。

---

## 目录

- [第 1 章 Benchmark 基准测试](#第-1-章-benchmark-基准测试)
  - [1.1 测试目标](#11-测试目标)
  - [1.2 测试参数](#12-测试参数)
  - [1.3 测试结果表](#13-测试结果表)
  - [1.4 测试脚本（统一）](#14-测试脚本统一)
  - [1.5 使用方法](#15-使用方法)
  - [1.6 结果解析与汇总](#16-结果解析与汇总)
  - [1.7 部署建议](#17-部署建议)
- [第 2 章 纯 Prefill 与纯 Decode 测试](#第-2-章-纯-prefill-与纯-decode-测试)
  - [2.1 测试背景：PD 分离](#21-测试背景pd-分离)
  - [2.2 纯 Prefill 测试](#22-纯-prefill-测试)
  - [2.3 纯 Decode 测试](#23-纯-decode-测试)
  - [2.4 结果归因](#24-结果归因)
- [附录：文件清单](#附录文件清单)

---

## 第 1 章 Benchmark 基准测试

### 1.1 测试目标

在固定请求数（num_prompts = 并发数）、固定输入输出上下文长度组合下，并发数逐级递增，测量推理服务的端到端性能指标：输入/输出/总 token 吞吐量、TTFT（首 token 延迟）、TPOT（每 token 生成时间）。

本测试模拟**真实在线服务**的负载模式：客户端以指定并发度持续发送请求，请求总数等于并发数，测量该并发级别下的聚合吞吐与延迟。

### 1.2 测试参数

| 参数 | 值 | 说明 |
|---|---|---|
| **输入输出长度组合** | `2048 512` `8192 1024` `32768 1024` `65536 1024` | 格式为 `输入长度 输出长度`，覆盖 2K~64K 输入 |
| **并发数序列** | `1, 4, 8, 16, 32, 64, 128` | 并发数 = num_prompts（请求数与并发数相同） |
| **数据集** | `random-ids`(SGLang) / `random`(vLLM) | 随机 token，避免前缀缓存干扰 |
| **random-range-ratio** | `1.0`(SGLang) / `0.0`(vLLM) | 固定长度（SGLang 1.0 表示完全使用 random-input-len，vLLM 0.0 表示固定长度，两者均为固定长度） |
| **Seed** | `123` | 固定随机种子，保证可复现 |
| **测试间隔** | `60s` | 每组测试后等待服务恢复稳态 |

**测试矩阵**：7 个并发级别 × 4 个 IO 组合 = **28 组测试**（每个框架）。

### 1.3 测试结果表

测试完成后，将每组结果汇总至下表。`<芯片类型>` 替换为实际被测芯片型号（如 H100、B200、MTT S80 等）。

| 模型名称 | 推理框架 | 输入长度 | 输出长度 | 并发数 | prefix长度 | 输入token吞吐量\_\<芯片类型\> (toks/s) | 输出token吞吐量\_\<芯片类型\> (toks/s) | 总token吞吐量\_\<芯片类型\> (toks/s) | Mean TTFT\_\<芯片类型\> (ms) | P99 TTFT\_\<芯片类型\> (ms) | Mean TPOT\_\<芯片类型\> (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| \<模型名\> | \<框架\> | 2048 | 512 | 1 | 0 | | | | | | |
| \<模型名\> | \<框架\> | 2048 | 512 | 4 | 0 | | | | | | |
| \<模型名\> | \<框架\> | 2048 | 512 | 8 | 0 | | | | | | |
| ... | ... | ... | ... | ... | ... | | | | | | |
| \<模型名\> | \<框架\> | 65536 | 1024 | 128 | 0 | | | | | | |

**填写说明**：

大部分指标可直接从 bench 工具输出的日志中提取，`collect_results.py` 会自动完成提取和汇总。以下为各指标的来源与提取方式：

| 指标 | 提取方式 | 日志中的字段名 |
|---|---|---|
| **输入token吞吐量** | SGLang 直接提取；vLLM 用 `总吞吐量 - 输出吞吐量` 计算 | `Input token throughput (tok/s)` |
| **输出token吞吐量** | 直接提取 | `Output token throughput (tok/s)` |
| **总token吞吐量** | 直接提取 | `Total token throughput (tok/s)` |
| **Mean TTFT** | 直接提取 | `Mean TTFT (ms)` |
| **P99 TTFT** | 直接提取 | `P99 TTFT (ms)` |
| **Mean TPOT** | 直接提取 | `Mean TPOT (ms)` |

### 1.4 测试脚本（统一）

> 完整脚本：`_scripts/bench.sh`
>
> **改进点**：
> 1. 合并 SGLang 和 vLLM 两个 benchmark 脚本，通过 `-F/--framework` 参数切换
> 2. 提取模型服务名的最后一段（以 `/` `\` 分割）并清除残留分隔符（`:` `\`）作为目录名
> 3. `BASE_URL` / `MODEL_PATH` / `SERVED_MODEL_NAME` 支持命令行参数传递
> 4. 默认后台执行（自动 nohup + 日志重定向），`-f/--foreground` 切前台
> 5. 并发数和 IO 组合可通过 `-c` / `-i` 参数覆盖
> 6. 测试结束后自动调用 `collect_results.py` 生成汇总 CSV，再自动调用 `csv_to_md.py` 生成 Markdown 测试报告
> 7. 通过 `-T/--tester` 指定测试人员，报告目录增加 tester 层级（`{report-dir}/{tester}/{model_name}/{TS}/`），Markdown 报告命名为 `{tester}_{model_name}_results_report_{TS}.md`

```bash
#!/bin/bash
set -euo pipefail

# --- 默认配置 ---
FRAMEWORK=""
BASE_URL=""
MODEL_PATH=""
SERVED_MODEL_NAME=""
SEED=123
SLEEP_TIME=60
REPORT_DIR=""
CHIP_TYPE=""
TESTER=""
BACKGROUND=true

# 并发数列表（同时也是 num-prompts 的值）
DEFAULT_CONCURRENCY="1,4,8,16,32,64,128"
# 输入输出长度组合，格式 "in_len out_len"，组合之间逗号分隔
DEFAULT_IO="2048 512,8192 1024,32768 1024,65536 1024"

# --- 参数解析 ---
usage() {
  cat <<EOF
Usage: $0 -F <sglang|vllm> [OPTIONS]

必选参数:
  -F, --framework FRAMEWORK    推理框架类型: sglang 或 vllm
  -u, --base-url URL            推理服务地址
  -m, --model-path PATH         模型路径
  -n, --served-model-name NAME  服务模型名
  -t, --chip-type TYPE          芯片类型(用于CSV列名后缀,如H100/B200)
  -T, --tester NAME            测试人员(用于报告目录层级和报告命名)

可选参数:
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_reports)
  -c, --concurrency LIST        并发数列表(逗号分隔) (default: $DEFAULT_CONCURRENCY)
  -i, --io-combinations LIST    IO组合(逗号分隔,每组"in out")
                                 (default: $DEFAULT_IO)
  -s, --sleep SECONDS           每次测试间隔秒数    (default: $SLEEP_TIME)
  -f, --foreground             前台执行(默认后台)
  -h, --help                    显示帮助

示例（默认后台执行）:
  $0 -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan
  $0 -F vllm   -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 -T zhangsan
  $0 -F sglang -f -u http://... -m /path -n name -t H100 -T zhangsan              # 前台执行
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -F|--framework)           FRAMEWORK="$2";            shift 2 ;;
    -u|--base-url)            BASE_URL="$2";             shift 2 ;;
    -m|--model-path)          MODEL_PATH="$2";           shift 2 ;;
    -n|--served-model-name)   SERVED_MODEL_NAME="$2";    shift 2 ;;
    -r|--report-dir)          REPORT_DIR="$2";           shift 2 ;;
    -c|--concurrency)         CONCURRENCY_ARG="$2";      shift 2 ;;
    -i|--io-combinations)     IO_ARG="$2";              shift 2 ;;
    -s|--sleep)               SLEEP_TIME="$2";           shift 2 ;;
    -f|--foreground)          BACKGROUND=false;          shift   ;;
    -t|--chip-type)           CHIP_TYPE="$2";            shift 2 ;;
    -T|--tester)              TESTER="$2";               shift 2 ;;
    -h|--help)                usage; exit 0             ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1    ;;
  esac
done

# --- 校验框架参数 ---
if [[ -z "$FRAMEWORK" ]]; then
  echo "ERROR: 必须指定推理框架，使用 -F sglang 或 -F vllm" >&2
  echo ""
  usage
  exit 1
fi

case "$FRAMEWORK" in
  sglang|sg)
    FRAMEWORK="sglang"
    DATASET_NAME="random-ids"
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./sglang_reports"
    BENCH_LABEL="SGLang"
    LOG_PREFIX="sglang_bench"
    ;;
  vllm|vl)
    FRAMEWORK="vllm"
    DATASET_NAME="random"
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./vllm_reports"
    BENCH_LABEL="vLLM"
    LOG_PREFIX="vllm_bench"
    ;;
  *)
    echo "ERROR: 不支持的框架 '$FRAMEWORK'，请使用 sglang 或 vllm" >&2
    exit 1
    ;;
esac

# --- 校验必选参数 ---
MISSING=""
[[ -z "$BASE_URL" ]]          && MISSING+="  --base-url / -u\n"
[[ -z "$MODEL_PATH" ]]        && MISSING+="  --model-path / -m\n"
[[ -z "$SERVED_MODEL_NAME" ]] && MISSING+="  --served-model-name / -n\n"
[[ -z "$CHIP_TYPE" ]]          && MISSING+="  --chip-type / -t\n"
[[ -z "$TESTER" ]]             && MISSING+="  --tester / -T\n"
if [[ -n "$MISSING" ]]; then
  echo "ERROR: 以下必选参数未指定:" >&2
  printf "%b" "$MISSING" >&2
  echo "" >&2
  usage
  exit 1
fi

# --- 前置校验：serve_command.txt ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVE_CMD_FILE="${PROJECT_ROOT}/serve_command.txt"
if [[ ! -f "$SERVE_CMD_FILE" ]]; then
  echo "ERROR: 找不到 serve_command.txt，请先复制模板并填写模型服务启动命令：" >&2
  echo "  cp serve_command_template.txt serve_command.txt" >&2
  echo "  # 然后编辑 serve_command.txt 填写真实部署命令" >&2
  exit 1
fi
if [[ ! -s "$SERVE_CMD_FILE" ]]; then
  echo "ERROR: serve_command.txt 文件为空，请填写真实的模型服务启动命令。" >&2
  exit 1
fi

# --- 后台执行（默认）：带 --foreground 重新 nohup 自身 ---
if [[ "$BACKGROUND" == "true" ]]; then
  mkdir -p "$REPORT_DIR/${TESTER}"
  LOG_FILE="${REPORT_DIR}/${TESTER}/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
  # 重新启动自身（带 --foreground 强制前台），输出重定向到日志文件
  nohup "$0" \
    --framework "$FRAMEWORK" \
    --base-url "$BASE_URL" \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --report-dir "$REPORT_DIR" \
    --concurrency "${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}" \
    --io-combinations "${IO_ARG:-$DEFAULT_IO}" \
    --sleep "$SLEEP_TIME" \
    --chip-type "$CHIP_TYPE" \
    --tester "$TESTER" \
    --foreground \
    > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "${BENCH_LABEL} benchmark 已在后台启动 (PID: $PID)"
  echo "日志文件: $LOG_FILE"
  echo "监控进度: tail -f \"$LOG_FILE\""
  echo "停止进程: kill $PID"
  exit 0
fi

# --- 构建并发数列表 ---
CONCURRENCY_STR="${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}"
IFS=',' read -ra concurrency_list <<< "$CONCURRENCY_STR"

# --- 构建 IO 组合列表 ---
IO_STR="${IO_ARG:-$DEFAULT_IO}"
IFS=',' read -ra io_combinations <<< "$IO_STR"

# --- 提取模型服务名最后一段（以 / \ 分割），并清除残留 : \，用于目录创建 ---
safe_model_name=$(echo "$SERVED_MODEL_NAME" | sed 's/.*[\/\\]//; s/[:\\]//g')

# --- 创建报告目录 ---
RUN_DIR="$REPORT_DIR/${TESTER}/${safe_model_name}/${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "=== Starting ${BENCH_LABEL} Benchmark Suite ==="
echo "=== Framework:    $FRAMEWORK"
echo "=== Base URL:     $BASE_URL"
echo "=== Model Path:   $MODEL_PATH"
echo "=== Served Name:  $SERVED_MODEL_NAME"
echo "=== Report Dir:   $RUN_DIR"
echo "=== Concurrency:  ${concurrency_list[*]}"
echo "=== IO Combos:     ${io_combinations[*]}"
echo "=== Chip Type:    ${CHIP_TYPE:-(未指定)}"
echo "=== Tester:       $TESTER"
echo "=========================================="

# --- 运行时间戳（用于输出文件名，避免覆盖） ---
RUN_TS=$(date +%Y%m%d_%H%M%S)

# 外层循环：遍历不同的并发值
for concurrency in "${concurrency_list[@]}"; do
  num_prompts=$concurrency

  echo ""
  echo "--- Configuration: Max Concurrency=$concurrency, Num Prompts=$num_prompts ---"

  # 内层循环：遍历不同的输入/输出长度组合
  for combo in "${io_combinations[@]}"; do
    # 解析组合字符串
    input_len=$(echo "$combo" | awk '{print $1}')
    output_len=$(echo "$combo" | awk '{print $2}')

    # 构建日志文件路径
    log_file="${RUN_DIR}/input_len-${input_len}-output_len-${output_len}-bs-${num_prompts}.log"

    echo ""
    echo ">>> Running test: Input=$input_len, Output=$output_len, Concurrency=$concurrency"
    echo ">>> Log file: $log_file"

    # --- 根据框架选择对应的 bench 命令 ---
    if [[ "$FRAMEWORK" == "sglang" ]]; then
      python -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --base-url "$BASE_URL" \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio 1.0 \
        --num-prompts "$num_prompts" \
        --max-concurrency "$concurrency" \
        --seed "$SEED" \
        > "$log_file" 2>&1
    elif [[ "$FRAMEWORK" == "vllm" ]]; then
      vllm bench serve \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --base-url "$BASE_URL" \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$concurrency" \
        --trust-remote-code \
        --temperature 0.7 \
        --random-range-ratio 0.0 \
        --random-prefix-len 0 \
        --seed "$SEED" \
        --metric_percentiles 95,99 \
        --ready-check-timeout-sec 30 \
        > "$log_file" 2>&1
    fi

    # 检查命令执行状态
    if [ $? -ne 0 ]; then
      echo "!!! Error occurred. Check log file: $log_file"
    else
      echo ">>> Success. Results saved to: $log_file"
    fi

    # 每次测试后等待指定时间
    echo "--- Finished Input=$input_len, Output=$output_len. Sleeping for $SLEEP_TIME seconds... ---"
    sleep "$SLEEP_TIME"
  done

  echo "--- Completed all IO combinations for Concurrency=$concurrency ---"
done

echo ""
echo "=== All benchmark runs finished ==="
echo "=== Check results in: $RUN_DIR ==="

# --- 构建 BENCH_COMMANDS（循环模板，变量占位，用于报告第三部分） ---
CONC_LIST="${concurrency_list[*]}"
IO_QUOTED=""
for io in "${io_combinations[@]}"; do
  IO_QUOTED="${IO_QUOTED}\"$io\" "
done
IO_QUOTED="${IO_QUOTED% }"

if [[ "$FRAMEWORK" == "sglang" ]]; then
  BENCH_COMMANDS="# 并发数列表: ${CONCURRENCY_STR}
# IO 组合: ${IO_STR}
for CONCURRENCY in ${CONC_LIST}; do
  NUM_PROMPTS=\$CONCURRENCY
  for IO in ${IO_QUOTED}; do
    INPUT_LEN=\$(echo \"\$IO\" | awk '{print \$1}')
    OUTPUT_LEN=\$(echo \"\$IO\" | awk '{print \$2}')
    python -m sglang.bench_serving \\
      --backend sglang-oai-chat \\
      --base-url \"\$BASE_URL\" \\
      --model \"\$MODEL_PATH\" \\
      --served-model-name \"\$SERVED_MODEL_NAME\" \\
      --dataset-name ${DATASET_NAME} \\
      --random-input-len \$INPUT_LEN \\
      --random-output-len \$OUTPUT_LEN \\
      --random-range-ratio 1.0 \\
      --num-prompts \$NUM_PROMPTS \\
      --max-concurrency \$CONCURRENCY \\
      --seed ${SEED}
  done
done"
elif [[ "$FRAMEWORK" == "vllm" ]]; then
  BENCH_COMMANDS="# 并发数列表: ${CONCURRENCY_STR}
# IO 组合: ${IO_STR}
for CONCURRENCY in ${CONC_LIST}; do
  NUM_PROMPTS=\$CONCURRENCY
  for IO in ${IO_QUOTED}; do
    INPUT_LEN=\$(echo \"\$IO\" | awk '{print \$1}')
    OUTPUT_LEN=\$(echo \"\$IO\" | awk '{print \$2}')
    vllm bench serve \\
      --backend openai-chat \\
      --endpoint /v1/chat/completions \\
      --base-url \"\$BASE_URL\" \\
      --model \"\$MODEL_PATH\" \\
      --served-model-name \"\$SERVED_MODEL_NAME\" \\
      --dataset-name ${DATASET_NAME} \\
      --random-input-len \$INPUT_LEN \\
      --random-output-len \$OUTPUT_LEN \\
      --num-prompts \$NUM_PROMPTS \\
      --max-concurrency \$CONCURRENCY \\
      --trust-remote-code \\
      --temperature 0.7 \\
      --random-range-ratio 0.0 \\
      --random-prefix-len 0 \\
      --seed ${SEED} \\
      --metric_percentiles 95,99 \\
      --ready-check-timeout-sec 30
  done
done"
fi

# --- 自动收集结果生成 CSV ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${RUN_DIR}/results.csv"
python3 "${SCRIPT_DIR}/collect_results.py" \
  --report-dir "$RUN_DIR" \
  --model-name "$SERVED_MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --chip-type "$CHIP_TYPE" \
  --out "$CSV_FILE"

# --- 自动生成 Markdown 测试报告 ---
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_results_report_${RUN_TS}.md"
python3 "${SCRIPT_DIR}/csv_to_md.py" \
  --csv "$CSV_FILE" \
  --output "$MD_FILE" \
  --bench-command "$BENCH_COMMANDS"
```

### 1.5 使用方法

#### 基本用法（默认后台执行）

```bash
# SGLang（默认后台执行，自动 nohup + 日志重定向）
./bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 \
  -t H100 -T zhangsan

# vLLM
./bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 \
  -t H100 -T zhangsan
```

#### 前台执行

```bash
# 默认即后台执行（自动 nohup + 日志重定向）。
# 如需前台执行（输出直接到终端），加 -f / --foreground：
./bench.sh -F sglang -f -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan
./bench.sh -F vllm   -f -u http://127.0.0.1:8000 -m /path -n name -t H100 -T zhangsan

# 后台执行（默认）
./bench.sh -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan
# 输出示例：
#   SGLang benchmark 已在后台启动 (PID: 12345)
#   日志文件: ./sglang_reports/zhangsan/sglang_bench_20260827_143022.log
#   监控进度: tail -f "./sglang_reports/zhangsan/sglang_bench_20260827_143022.log"
```

#### 自定义参数

```bash
# 自定义并发数和 IO 组合
./bench.sh -F sglang \
  -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 \
  -c 1,8,32,128 \
  -i "2048 512,8192 1024" \
  -T zhangsan

# 指定报告目录和间隔时间
./bench.sh -F vllm \
  -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 \
  -r /mnt/results/vllm_bench \
  -s 30 -T zhangsan
```

#### 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|---|---|---|---|
| `--framework` | `-F` | **必选** 推理框架类型: `sglang` 或 `vllm` | — |
| `--base-url` | `-u` | **必选** 推理服务地址 | — |
| `--model-path` | `-m` | **必选** 模型路径 | — |
| `--served-model-name` | `-n` | **必选** 服务模型名（取 `/` `\` 分割后最后一段，清除 `:` `\` 作为目录名） | — |
| `--report-dir` | `-r` | 报告输出目录 | `./sglang_reports` / `./vllm_reports`（按框架自动选择） |
| `--concurrency` | `-c` | 并发数列表（逗号分隔） | `1,4,8,16,32,64,128` |
| `--io-combinations` | `-i` | IO组合（逗号分隔，每组"in out"） | `2048 512,8192 1024,...` |
| `--sleep` | `-s` | 每次测试间隔秒数 | `60` |
| `--foreground` | `-f` | 前台执行（脚本默认后台执行） | — |
| `--chip-type` | `-t` | **必选** 芯片类型（用于结果 CSV 列名后缀，如 `H100`/`B200`） | — |
| `--tester` | `-T` | **必选** 测试人员（用于报告目录层级和报告命名） | — |

### 1.6 结果解析与汇总

每个测试生成一个 `.log` 文件，包含 bench 工具的原生输出。关键指标及对应字段：

#### SGLang `bench_serving` 输出示例

```
============ Serving Benchmark Result ============
Successful requests:      128
Failed requests:            0
Benchmark duration (s):  857.88
Total input tokens:    131072
Total generated tokens: 131072
Request throughput (req/s):        0.15
Input token throughput (tok/s):   152.64      ← 输入token吞吐量
Output token throughput (tok/s):  152.64      ← 输出token吞吐量
Total token throughput (tok/s):   305.28      ← 总token吞吐量
Mean TTFT (ms):  321.42                        ← Mean TTFT
Median TTFT (ms): 306.62
P99 TTFT (ms):  1181.54
Mean TPOT (ms):  9.25                          ← Mean TPOT
Median TPOT (ms): 9.23
P99 TPOT (ms):  9.89
```

#### vLLM `bench serve` 输出示例

```
==== Serving Benchmark Result ===
Successful requests:        128
Failed requests:               0
Benchmark duration (s):   857.88
Total input tokens:        131072
Total generated tokens:    131072
Request throughput (req/s):     0.37
Output token throughput (tok/s): 95.49       ← 输出token吞吐量
Total token throughput (tok/s):  3915.14    ← 总token吞吐量

Mean TTFT (ms):  321.42                       ← Mean TTFT
P95 TTFT (ms):  982.37
P99 TTFT (ms):  1181.54                       ← P99 TTFT
Mean TPOT (ms):  9.25                         ← Mean TPOT
```

> **注意**：vLLM 的 `bench serve` 不直接输出"输入 token 吞吐量"，`collect_results.py` 会自动用 `总吞吐量 - 输出吞吐量` 计算补全。

#### 自动收集结果（测试完成后自动执行）

所有 bench 脚本在测试结束后会**自动调用** `collect_results.py` 扫描日志目录，生成汇总 CSV，随后**自动调用** `csv_to_md.py` 将 CSV 转换为 Markdown 测试报告，无需手动操作。

生成的文件位置（`{TS}` 为运行时间戳 `YYYYMMDD_HHMMSS`，避免重复执行时覆盖）：
- **Benchmark**：`{report-dir}/{tester}/{model_name}/{TS}/input_len-{IL}-output_len-{OL}-bs-{N}.log` → `results.csv` → `{tester}_{model_name}_results_report_{TS}.md`
- **纯 Prefill**：`{report-dir}/{tester}/{model_name}/{TS}/prefill_input-{IL}-bs-{N}.log` → `prefill_results.csv` → `{tester}_{model_name}_prefill_results_report_{TS}.md`
- **纯 Decode**：`{report-dir}/{tester}/{model_name}/{TS}/decode_prefix-{PL}-output-{OL}-bs-{N}.log` → `decode_results.csv` → `{tester}_{model_name}_decode_results_report_{TS}.md`

CSV 表头（`-t` 指定芯片类型后，列名带后缀）：

```
模型名称,推理框架,输入长度,输出长度,并发数,prefix长度,输入token吞吐量_<芯片类型> (toks/s),输出token吞吐量_<芯片类型> (toks/s),总token吞吐量_<芯片类型> (toks/s),Mean TTFT_<芯片类型> (ms),P99 TTFT_<芯片类型> (ms),Mean TPOT_<芯片类型> (ms)
```

CSV 示例：

```
模型名称,推理框架,输入长度,输出长度,并发数,prefix长度,输入token吞吐量_H100 (toks/s),输出token吞吐量_H100 (toks/s),总token吞吐量_H100 (toks/s),Mean TTFT_H100 (ms),P99 TTFT_H100 (ms),Mean TPOT_H100 (ms)
glm-5.2-fp8,sglang,2048,512,1,0,152.79,95.49,3915.14,321.42,1181.54,9.25
glm-5.2-fp8,sglang,8192,1024,128,0,152.64,152.64,305.28,321.42,1181.54,9.25
```

也可以**手动执行**收集脚本（如只收集已有日志）：

```bash
python3 _scripts/collect_results.py \
  --report-dir ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022 \
  --model-name glm-5.2-fp8 \
  --framework sglang \
  --chip-type H100 \
  --out ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/results.csv
```

> **注意**：vLLM 的 `bench serve` 不直接输出"输入 token 吞吐量"，
> `collect_results.py` 会自动用 `总吞吐量 - 输出吞吐量` 计算补全。

#### 自动生成 Markdown 测试报告

CSV 生成后，脚本会**自动调用** `csv_to_md.py` 将 CSV 转换为 Markdown 测试报告。报告分三部分：

| 部分 | 内容来源 | 说明 |
|---|---|---|
| **一、测试结果** | CSV 文件自动转换 | 将 CSV 数据转为 Markdown 表格 |
| **二、模型服务启动命令** | `serve_command.txt` 文件 | 需从 `serve_command_template.txt` 复制并填写真实命令；找不到则报错 |
| **三、Benchmark 测试命令** | 脚本自动生成 | 以循环语句形式合并所有参数组合，变量用占位符表示（如 `$CONCURRENCY`、`$INPUT_LEN`），而非逐条列出 |

> **`serve_command.txt`**：需从 `serve_command_template.txt` 复制并填写真实模型服务启动命令（如 `python -m sglang.launch_server ...`）。
> `csv_to_md.py` 会自动读取其内容填入报告第二部分。模板见 `benchmark_analysis_template.md`。

也可以**手动执行**生成报告：

```bash
python3 _scripts/csv_to_md.py \
  --csv ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/results.csv \
  --output ./sglang_reports/zhangsan/glm-5.2-fp8/20260831_143022/zhangsan_glm-5.2-fp8_results_report_20260831_143022.md \
  --bench-command "python -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:8080 ..."
```

### 1.7 部署建议

由于本方案测试的输入长度最大达 65536（64K），并发数最高达 128，部署推理服务时需确保模型上下文窗口和请求队列容量足够，否则会出现 OOM 或请求被拒绝的情况。

| 参数 | 说明 | 建议值 |
|---|---|---|
| **最大上下文长度** | 单次请求允许的最大 token 数（输入 + 输出） | ≥ 131072（128K） |
| **最大请求数** | 引擎内部同时排队的最大请求数 | ≥ 64 |

#### SGLang 部署命令

```bash
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

#### vLLM 部署命令

```bash
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 202752 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  -tp 8 \
  -dp 1 \
  -pp 1 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

> **注意**：以上命令为第 1 章 Benchmark 的部署示例（前缀缓存开/关均可）。
> 第 2 章纯 Prefill / 纯 Decode 的部署命令需调整前缀缓存参数，详见 [2.1.1 模型部署注意事项](#211-模型部署注意事项前缀缓存开关)。

#### 参数说明

| 参数 | SGLang | vLLM | 说明 |
|---|---|---|---|
| 模型路径 | `--model-path` | 位置参数 | 模型权重目录 |
| 模型服务名 | `--served-model-name` | `--served-model-name` | API 请求中的 model 字段 |
| 监听地址 | `--host` | `--host` | 通常 `0.0.0.0` |
| 监听端口 | `--port` | `--port` | SGLang 默认 8080，vLLM 默认 8000 |
| 最大上下文长度 | `--context-length` | `--max-model-len` | ≥ 131072（128K） |
| 最大请求数 | `--max-running-requests` | `--max-num-seqs` | ≥ 64 |
| 每批最大 token 数 | `--chunked-prefill-size` | `--max-num-batched-tokens` | 控制单次 batch/prefill chunk 的最大 token 数 |
| GPU 显存使用比例 | `--mem-fraction-static` | `--gpu-memory-utilization` | 0.9（留 10% 给 CUDA context） |
| 张量并行 | `--tp` | `--tensor-parallel-size`（`-tp`） | 按 GPU 数量设置（如 8） |
| 数据并行 | `--dp` | `--data-parallel-size`（`-dp`） | 默认 1 |
| 流水线并行 | `--pp` | `--pipeline-parallel-size`（`-pp`） | 默认 1 |
| 专家并行 | `--ep` | `--enable-expert-parallel`（`-ep`） | SGLang 设整数并行数；vLLM 布尔开关，开启后 MoE 层用 EP 替代 TP |
| 关闭前缀缓存 | `--disable-radix-cache` | `--no-enable-prefix-caching` | 纯 Prefill 必加，纯 Decode **不加**，详见第 2 章 |
| 工具调用 | `--tool-call-parser` | `--enable-auto-tool-choice` | SGLang 指定 parser 即启用工具调用；vLLM 需先加 `--enable-auto-tool-choice` |
| 工具解析器 | `--tool-call-parser` | `--tool-call-parser` | 如 `glm47`、`qwen25` 等，按模型选择 |
| 推理解析器 | `--reasoning-parser` | `--reasoning-parser` | 如 `glm45`，解析思维链标签 |

> **提示**：
> - `--tool-call-parser` 和 `--reasoning-parser` 的具体值取决于模型架构，请按实际模型选择。
> - 纯 Decode 测试部署时**不要加** `--disable-radix-cache` / `--no-enable-prefix-caching`，否则共享前缀无法命中缓存。
> - 纯 Prefill 和纯 Decode 测试需**分别部署**独立服务实例，缓存开关要求相反。

---

## 第 2 章 纯 Prefill 与纯 Decode 测试

### 2.1 测试背景：PD 分离

在 **Prefill-Decode 分离（PD 分离）** 架构中，推理服务被拆分为两类 worker：

```
┌─ Prefill Worker ──────────────────────┐
│  接收用户请求，完成 prefill 前向计算      │
│  产出 KV cache，发送给 Decode Worker     │
│  负载特征：compute-bound（算力瓶颈）      │
└────────────────────────────────────────┘
              │ KV cache 传输
              ▼
┌─ Decode Worker ───────────────────────┐
│  接收 KV cache，逐 token 自回归生成     │
│  负载特征：memory-bound（带宽瓶颈）     │
└────────────────────────────────────────┘
```

PD 分离后，Prefill 和 Decode 的性能瓶颈完全不同：
- **Prefill** 是计算密集型，瓶颈在矩阵乘法算力
- **Decode** 是访存密集型，瓶颈在 HBM 带宽和 KV cache 读取

因此，需要**分别测试纯 Prefill 和纯 Decode 的性能**，才能准确评估芯片在 PD 分离场景下的能力。

### 2.1.1 模型部署注意事项：前缀缓存开关

纯 Prefill 和纯 Decode 测试对**前缀缓存（prefix caching）**的要求**完全相反**，必须使用不同的服务实例，不能共用：

| 测试场景 | 前缀缓存 | SGLang 启动参数 | vLLM 启动参数 | 原因 |
|---|---|---|---|---|
| **纯 Prefill** | **必须关闭** | 加 `--disable-radix-cache` | 加 `--no-enable-prefix-caching` | 关闭缓存后每个请求都是完整 prefill，确保测量到真实 prefill 算力 |
| **纯 Decode** | **必须开启** | 默认（不加 `--disable-radix-cache`） | 默认（不加 `--no-enable-prefix-caching`） | 共享前缀命中缓存 → prefill≈0，确保 TTFT 仅含 decode 开销，测到纯 decode 性能 |

> **注意**：如果在纯 Decode 测试时错误地关闭了前缀缓存，共享前缀不会被缓存，每个请求都需完整 prefill，导致 TTFT 严重失真（包含 prefill 延迟），无法反映真实 decode 性能。反之，纯 Prefill 测试时若未关闭缓存，部分请求可能命中缓存导致 prefill 被跳过，测量值偏低。

#### 纯 Prefill 部署命令（前缀缓存关闭）

**SGLang**：

```bash
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
  --disable-radix-cache \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

**vLLM**：

```bash
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 202752 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  -tp 8 \
  -dp 1 \
  -pp 1 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

#### 纯 Decode 部署命令（前缀缓存开启）

**SGLang**（不加 `--disable-radix-cache`，其余参数同纯 Prefill）：

```bash
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

**vLLM**（不加 `--no-enable-prefix-caching`，其余参数同纯 Prefill）：

```bash
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 202752 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  -tp 8 \
  -dp 1 \
  -pp 1 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

> 参数说明详见 [1.7 部署建议 — 参数说明表](#17-部署建议)。

### 2.2 纯 Prefill 测试

#### 2.2.1 方法原理

> **`output_len = 1` 就是纯 prefill。**

因果 LM 的定义：读入第 i 个 token，输出第 i+1 个 token 的概率分布。"生成第一个输出 token"这个动作，物理上就是 prefill 那次前向的副产品——最后一个 prompt 位置的 logits。

| output_len | 实际发生 | 是否纯 prefill |
|---|---|---|
| **1** | 1 次 prefill forward，**0 次 decode** | ✅ 严格纯净 |
| 2 | 1 prefill + 1 decode | ❌ 已污染 |
| 128，然后"减掉 decode 时间" | chunked prefill 会交织 | ❌ 减不干净 |

**指标口径**：
- prefill 吞吐量 = 日志中的输入 token 吞吐量（SGLang 直接提取 `Input token throughput (tok/s)`；vLLM 用 `总吞吐量 - 输出吞吐量` 计算）
- `TTFT ≈ E2E`（首 token 延迟 ≈ 端到端时间，因为无 decode）
- `output throughput` 在 output_len=1 时等于 QPS，**无参考意义**

#### 2.2.2 服务端配置

> **必须关闭前缀缓存**，否则前缀命中会测出物理上不可能的吞吐值。

| 框架 | 启动参数 |
|---|---|
| **SGLang** | 启动时加 `--disable-radix-cache` |
| **vLLM** | 启动时加 `--no-enable-prefix-caching` |

```bash
# SGLang 服务端示例（关闭前缀缓存）
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-Channel-FP8-w8a8 \
  --tp 8 --port 8080 \
  --disable-radix-cache

# vLLM 服务端示例（关闭前缀缓存）
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  -tp 8 --port 8000 \
  --no-enable-prefix-caching
```

#### 2.2.3 测试参数

| 参数 | 值 | 说明 |
|---|---|---|
| **输入长度** | `65536` | 固定 64K，测量最大 prefill 算力 |
| **输出长度** | `1` | 固定 1，确保 0 次 decode |
| **并发数 / num-prompts** | `1, 4, 8, 16, 32, 64, 128` | 并发数 = 请求数 |
| **random-range-ratio** | `1.0`(SGLang) / `0.0`(vLLM) | 固定长度（SGLang 1.0 表示完全使用 random-input-len，vLLM 0.0 表示固定长度，两者均为固定长度） |
| **random-prefix-len** | `0` | 无共享前缀 |
| **Seed** | `123` | 可复现 |

#### 2.2.4 测试脚本

**方式一：bench 工具脚本**（`_scripts/prefill_bench.sh`，支持 `-F sglang|vllm` 切换框架）

```bash
# SGLang 纯 Prefill
./prefill_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan

# vLLM 纯 Prefill
./prefill_bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 -T zhangsan

# 默认后台执行，加 -f 前台
./prefill_bench.sh -F sglang -f -u http://... -m /path -n name -t H100 -T zhangsan

# 自定义输入长度和并发
./prefill_bench.sh -F sglang \
  -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 \
  -c 1,8,64 -i "32768 1,65536 1" \
  -T zhangsan
```

**参数说明**：

| 参数 | 短选项 | 必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--framework` | `-F` | ✅ | — | `sglang` 或 `vllm` |
| `--base-url` | `-u` | ✅ | — | 推理服务地址 |
| `--model-path` | `-m` | ✅ | — | 模型路径 |
| `--served-model-name` | `-n` | ✅ | — | 服务模型名 |
| `--io-combinations` | `-i` | | `65536 1` | IO 组合（逗号分隔，每组 `"输入 输出"`） |
| `--concurrency` | `-c` | | `1,4,8,16,32,64,128` | 并发数列表 |
| `--chip-type` | `-t` | ✅ | — | 芯片类型（CSV 列名后缀） |
| `--tester` | `-T` | ✅ | — | 测试人员（用于报告目录层级和报告命名） |
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--foreground` | `-f` | | 后台 | 前台执行 |
| `--report-dir` | `-r` | | `./{framework}_prefill_reports` | 报告输出目录 |

核心命令（SGLang）：

```bash
python -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name random-ids \
  --random-input-len  $input_len    \   # 65536
  --random-output-len 1             \   # 固定 1
  --random-range-ratio 1.0          \
  --num-prompts $concurrency        \
  --max-concurrency $concurrency    \
  --seed 123
```

核心命令（vLLM）：

```bash
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name random \
  --random-input-len  $input_len \
  --random-output-len 1 \
  --random-prefix-len 0 \
  --num-prompts $concurrency --max-concurrency $concurrency \
  --trust-remote-code --temperature 0.7 \
  --random-range-ratio 0.0 --seed 123 \
  --metric_percentiles 95,99 --ready-check-timeout-sec 30
```

#### 2.2.5 测试结果表

| 模型名称 | 推理框架 | 输入长度 | 输出长度 | 并发数 | prefix长度 | 输入token吞吐量\_\<芯片类型\> (toks/s) | 输出token吞吐量\_\<芯片类型\> (toks/s) | 总token吞吐量\_\<芯片类型\> (toks/s) | Mean TTFT\_\<芯片类型\> (ms) | P99 TTFT\_\<芯片类型\> (ms) | Mean TPOT\_\<芯片类型\> (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| \<模型名\> | \<框架\> | 65536 | 1 | 1 | 0 | | | | | | |
| \<模型名\> | \<框架\> | 65536 | 1 | 4 | 0 | | | | | | |
| ... | ... | ... | ... | ... | ... | | | | | | |
| \<模型名\> | \<框架\> | 65536 | 1 | 128 | 0 | | | | | | |

> **prefill 吞吐量** = 输入 token 吞吐量（SGLang 直接提取 `Input token throughput (tok/s)`；vLLM 用 `总吞吐量 - 输出吞吐量` 计算，`collect_results.py` 自动处理）
>
> **注意**：`Output token throughput` 在 output_len=1 时等于 QPS，不作为 prefill 指标。
> 只看 **输入 token 吞吐量** 和 **TTFT**。

---

### 2.3 纯 Decode 测试

#### 2.3.1 方法原理

纯 Decode 测试有两种方法，推荐使用 **方法 B**（更贴近 PD 分离 decode worker 真实负载）：

| 方法 | 原理 | 前缀缓存 | 适用场景 |
|---|---|---|---|
| **A. 最小输入** | `input_len=1` + 长输出，prefill 1 个 token 可忽略 | 不需要 | 快速验证 decode 算力 |
| **B. 共享前缀** ★ | 长前缀命中缓存（prefill≈0）+ 1 新 token + 长输出 | **必须开启** | PD 分离 decode worker 真实负载 |

**方法 B 详解**（推荐）：

```
1. 服务端开启前缀缓存
2. 预热：发送 1 个请求（prefix + 1 token + max_tokens=1）填充缓存
   → 等价于 "KV cache 已从 prefill worker 传输到 decode worker"
3. 基准测试：并发发送 N 个请求，每个 = 共享前缀 + 1 个随机新 token + 长输出
   → 前缀命中缓存，prefill 近似为零
   → 全部耗时是 decode 迭代
4. 指标（bench 工具直接提取）：
   decode 吞吐量 = 日志 Output token throughput (tok/s)
   TPOT = 日志 Mean TPOT (ms)
   TTFT = 日志 Mean TTFT (ms)（应很小，因缓存命中）
```

**为什么方法 B 更贴近 PD 分离**：decode worker 接收的是 prefill worker 已算好的 KV cache。
共享前缀命中缓存 = KV cache 已就位。之后每请求只携带 1 个新 token，前缀 prefill 跳过，
全部是 decode 迭代。这正是 decode worker 的真实工作负载。

**对称性**：
- 纯 Prefill：`output_len=1`（首 token 是 prefill 副产物，0 次 decode）
- 纯 Decode：`prefix_len>>0 + input_len=1 + output_len>>0`（前缀命中缓存，全 decode）

#### 2.3.2 服务端配置

> **必须开启前缀缓存**（与纯 Prefill 测试相反）。

| 框架 | 启动参数 |
|---|---|
| **SGLang** | 默认开启，**不要加** `--disable-radix-cache` |
| **vLLM** | 默认开启，**不要加** `--no-enable-prefix-caching` |

```bash
# SGLang 服务端示例（前缀缓存保持开启 = 默认）
python -m sglang.launch_server \
  --model-path /data1/GLM-5.2-Channel-FP8-w8a8 \
  --tp 8 --port 8080
  # 不加 --disable-radix-cache

# vLLM 服务端示例（前缀缓存保持开启 = 默认）
vllm serve /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  --served-model-name glm-5.2-fp8 \
  -tp 8 --port 8000
  # 不加 --no-enable-prefix-caching
```

#### 2.3.3 测试参数

| 参数 | 值 | 说明 |
|---|---|---|
| **前缀长度** | `4096, 32768, 65536` | 3 级，覆盖小/中/大 KV cache |
| **输出长度** | `1024` | 固定，足够 decode 迭代测量吞吐 |
| **新增 token 数** | `1` | 每请求在共享前缀后新增 1 个 token（确保 prefill≈0） |
| **并发数** | `1, 4, 8, 16, 32, 64, 128` | max-concurrency |
| **num-prompts** | `2 × 并发数` | 稀释首个请求 prefill 开销（bench 工具无预热机制） |
| **random-range-ratio** | `1.0`(SGLang) / `0.0`(vLLM) | 固定长度（SGLang 1.0 表示完全使用 random-input-len，vLLM 0.0 表示固定长度，两者均为固定长度） |
| **Seed** | `123` | 可复现 |

**方法 A（最小输入，无需前缀缓存）的替代参数**：

| 参数 | 值 |
|---|---|
| **输入长度** | `1`（固定 1） |
| **输出长度** | `128, 512, 1024, 2048, 4096` |
| **前缀长度** | `0`（无前缀） |

可复用第 1 章脚本：
```bash
./bench.sh -F sglang -u http://... -m /path -n name \
  -i "1 128,1 512,1 1024,1 2048,1 4096"
```

#### 2.3.4 测试脚本（bench 工具）

**统一脚本**（`_scripts/decode_bench.sh`，支持 `-F sglang|vllm` 切换框架）：

```bash
# SGLang 纯 Decode（使用 generated-shared-prefix 数据集，保证共享前缀）
./decode_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 4096,32768,65536 \
  -o 1024 \
  -c 1,4,8,16,32,64,128 \
  -T zhangsan

# vLLM 纯 Decode（使用 prefix_repetition 数据集，保证共享前缀）
./decode_bench.sh -F vllm \
  -u http://127.0.0.1:8000 \
  -m /data/lxl/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 4096,32768,65536 \
  -o 1024 \
  -c 1,4,8,16,32,64,128 \
  -T zhangsan

# 默认后台执行，加 -f 前台
./decode_bench.sh -F sglang -f -u http://... -m /path -n name -t H100 -T zhangsan

# 自定义前缀、输出和并发
./decode_bench.sh -F sglang \
  -u http://127.0.0.1:8080 \
  -m /data1/GLM-5.2-Channel-FP8-w8a8 \
  -n glm-5.2-fp8 -t H100 \
  -p 8192,65536 -o 512 -c 1,16,128 \
  -T zhangsan
```

**参数说明**：

| 参数 | 短选项 | 必选 | 默认值 | 说明 |
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
| `--sleep` | `-s` | | `60` | 每组测试间隔（秒） |
| `--foreground` | `-f` | | 后台 | 前台执行 |
| `--report-dir` | `-r` | | `./{framework}_decode_reports` | 报告输出目录 |

核心命令（SGLang — `generated-shared-prefix`）：

```bash
# GSP 数据集：所有请求共享同一前缀，保证缓存命中
python -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1                    \   # 1 组 → 所有请求共享同一前缀
  --gsp-prompts-per-group $num_prompts \   # = 2 × 并发数，稀释首请求 prefill
  --gsp-system-prompt-len $prefix_len   \   # 共享前缀长度
  --gsp-question-len 1                  \   # 每请求仅 1 个新 token
  --gsp-output-len $output_len           \   # 1024
  --num-prompts $num_prompts --max-concurrency $concurrency \   # num_prompts = 2 × 并发数
  --seed 123
```

核心命令（vLLM — `prefix_repetition`）：

```bash
# prefix_repetition 数据集：num-prefixes=1 → 所有请求共享同一前缀
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len $prefix_len   \   # 共享前缀长度
  --prefix-repetition-suffix-len 1              \   # 每请求仅 1 个新 token
  --prefix-repetition-num-prefixes 1           \   # 1 个前缀 → 所有请求共享
  --prefix-repetition-output-len $output_len    \   # 1024
  --num-prompts $num_prompts --max-concurrency $concurrency \   # num_prompts = 2 × 并发数
  --trust-remote-code --temperature 0.7 --seed 123 \
  --metric_percentiles 95,99 --ready-check-timeout-sec 30
```

> **数据集选择原因**：`random-ids`/`random` + `random-prefix-len` 在 SGLang 中
> 每个请求可能生成独立随机前缀，无法保证缓存命中。`generated-shared-prefix`
> 和 `prefix_repetition` 显式保证同组请求共享完全相同的前缀，首个请求填充缓存
> 后后续请求全部命中，确保 prefill≈0，测到纯 decode 性能。

> **bench 工具的局限**：bench_serving / vllm bench 并发发送所有请求，首个请求需
> prefill 完整前缀才能填充缓存，后续请求才命中。已通过 `num_prompts = 2 × 并发数`
> 稀释首个请求的 prefill 开销，但 TTFT 仍受首请求污染。若需精确测量 TTFT/TPOT，使用下方 HTTP 脚本。

#### 2.3.5 测试脚本（HTTP 精确测量）

> **说明**：本节介绍的 `decode_http_sweep.py` 为补充测量方法，实际测试中通常不使用，**仅供了解参考**。标准测试流程使用 2.3.4 节的 `decode_bench.sh`（基于 bench 工具）即可满足需求。仅当需要流式 SSE 精确测量 TTFT/TPOT、或验证 bench 工具结果时，可参考此方法。

**`_scripts/decode_http_sweep.py`** — 零第三方依赖，流式 SSE 精确测量 TTFT/TPOT。

两种模式分工：

| 模式 | 目的 | 参数组合 | 适用场景 |
|---|---|---|---|
| **batch** | 扫描不同并发下的吞吐曲线 | 21 格（3 前缀 × 1 输出 × 7 并发） | 画吞吐-并发曲线 |
| **steady** | 测单一并发的稳态极限 | 1 格（定点持续 N 秒） | 验证峰值稳定性，类似生产负载 |

核心设计：
1. **预热阶段**：先发送 1 个请求（prefix + 1 token + max_tokens=1）填充前缀缓存
2. **基准阶段**：并发发送 N 个请求（prefix + 1 随机新 token + max_tokens=output_len）
3. **流式 SSE**：精确记录首 chunk 时间（TTFT）和末 chunk 时间，计算 TPOT
4. **自动校验**：检查 `prompt_tokens == prefix_len + 1`，`completion_tokens == output_len`

```bash
# 基本用法
python3 _scripts/decode_http_sweep.py \
  --base-url http://127.0.0.1:30000 \
  --model glm-5.2-fp8 \
  --prefix-lens 4096,32768,65536 \
  --output-lens 1024 \
  --batches 1,4,8,16,32,64,128 \
  --vocab-size 151552 \
  --framework sglang --tp 8 \
  --tag h100-tp8 \
  --out results/decode_http.csv

# 稳态模式（持续 120 秒）
python3 _scripts/decode_http_sweep.py \
  --base-url http://127.0.0.1:30000 --model glm-5.2-fp8 \
  --mode steady \
  --prefix-lens 4096 --output-lens 1024 \
  --concurrency 64 --duration 120 \
  --vocab-size 151552 --framework sglang --tp 8 \
  --out results/decode_steady.csv

# 禁用流式（服务端不支持 SSE 时）
python3 _scripts/decode_http_sweep.py ... --no-stream
```

输出 CSV 字段：

```
ts,tag,layer,framework,model,tp,mode,
prefix_len,output_len,batch,repeat,concurrency,
latency_s,output_tokens,ok,fail,
decode_tok_s,ttft_ms,tpot_ms,note
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--prefix-lens` | `4096,32768,65536` | 共享前缀长度 |
| `--output-lens` | `1024` | decode 输出长度 |
| `--batches` | `1,4,8,16,32,64,128` | batch 模式并发数 |
| `--new-tokens` | `1` | 每请求新增 token 数（保持 1） |
| `--token-budget` | `196608` | 跳过 prefix + output×batch 超限格子 |
| `--repeats` | `3` | 每格重复次数，取中位数 |
| `--warmup` | `2` | 预热轮数（JIT + 缓存填充） |
| `--stream` | `True` | 流式 SSE 精确测 TTFT/TPOT |
| `--mode` | `batch` | `batch` 或 `steady` |

#### 2.3.6 测试结果表

| 模型名称 | 推理框架 | 输入长度 | 输出长度 | 并发数 | prefix长度 | 输入token吞吐量\_\<芯片类型\> (toks/s) | 输出token吞吐量\_\<芯片类型\> (toks/s) | 总token吞吐量\_\<芯片类型\> (toks/s) | Mean TTFT\_\<芯片类型\> (ms) | P99 TTFT\_\<芯片类型\> (ms) | Mean TPOT\_\<芯片类型\> (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| \<模型名\> | \<框架\> | 1 | 1024 | 1 | 4096 | | | | | | |
| \<模型名\> | \<框架\> | 1 | 1024 | 4 | 4096 | | | | | | |
| ... | ... | ... | ... | ... | ... | | | | | | |
| \<模型名\> | \<框架\> | 1 | 1024 | 128 | 65536 | | | | | | |

**指标口径**：

| 指标 | bench 工具（collect_results.py 自动提取） | HTTP 脚本（decode_http_sweep.py 自动计算） |
|---|---|---|
| **decode 吞吐量** | 日志 `Output token throughput (tok/s)` | `Σ output_tokens / wall_time` |
| **Mean TPOT** | 日志 `Mean TPOT (ms)` | `(末chunk时间 - 首chunk时间) / (output_len - 1)` |
| **Mean TTFT** | 日志 `Mean TTFT (ms)` | 流式首 chunk 延迟（ms） |

> **注意**：前缀缓存命中时 TTFT 应很小；若 TTFT 很大说明缓存未生效，需检查服务端是否开启了前缀缓存。

---

### 2.4 结果归因

#### 2.4.1 Prefill vs Decode 瓶颈对比

| 阶段 | 瓶颈类型 | 关键指标 | 优化方向 |
|---|---|---|---|
| **Prefill** | 计算密集（compute-bound） | prefill_tok_s, MFU | 算力、TP 通信、attention kernel |
| **Decode** | 访存密集（memory-bound） | decode_tok_s, TPOT | HBM 带宽、KV cache 读取效率 |

#### 2.4.2 关键判据

| 观察 | 判读 | 处置 |
|---|---|---|
| prefill 吞吐随 batch 线性增长但后趋平 | 正常，算力饱和 | 达到硬件天花板 |
| prefill 吞吐不随 batch 增长 | TP 通信瓶颈或调度问题 | 查 NCCL，降 TP 或开 DP-attention |
| decode 吞吐不随 concurrency 增长 | 已达 decode 容量上限 | 正常，decode 是 memory-bound |
| decode TTFT 异常大 | 前缀缓存未命中 | 确认服务端开启了前缀缓存 |
| decode TPOT 随 prefix_len 增大而增大 | 正常，KV cache 越长每步访存越多 | 这是 decode 的本质特征 |
| output_len=1 时 output throughput ≈ QPS | 正常，1 个 token = 1 个请求 | 只看 prefill_tok_s，不看 output throughput |

#### 2.4.3 结合硬件层测试（可选）

如需进一步归因到硬件层（区分"框架损失"与"硬件天花板"），可结合
L0 硬件层测试工具（GEMM / Attention /
HBM 带宽 / NCCL 通信），构建 roofline 上界，计算 MFU 达成率。

---

## 附录：文件清单

```
general-benchmark-test/
├── Model_Inference_Benchmark_TestStrategy.md   本文档
├── README.md                                   快速参考指南（三种测试对比、执行命令、参数说明）
├── benchmark_analysis_template.md              Markdown 报告模板
├── serve_command_template.txt                  模型服务启动命令模板（需复制为 serve_command.txt 填写真实命令）
└── _scripts/
    ├── bench.sh                                第1章 benchmark 脚本（统一，-F 指定 sglang/vllm）
    ├── prefill_bench.sh                        第2章 纯 Prefill 脚本（统一，-F 指定 sglang/vllm）
    ├── decode_bench.sh                         第2章 纯 Decode 脚本（统一，SGLang 用 GSP / vLLM 用 prefix_repetition）
    ├── decode_http_sweep.py                    第2章 纯 Decode HTTP 精确测量（零依赖，流式 SSE）【仅供参考，实际不使用】
    ├── collect_results.py                      ★ 自动收集日志结果生成 CSV（所有脚本结束后自动调用）
    └── csv_to_md.py                            ★ 将 CSV 转换为 Markdown 测试报告（自动调用）
```

### 脚本依赖

| 脚本 | 依赖 | 说明 |
|---|---|---|
| `bench.sh` | `sglang` (bench_serving) / `vllm` (bench serve) | 统一脚本，`-F` 指定框架 |
| `prefill_bench.sh` | `sglang` / `vllm` | output_len=1 变体，`-F` 指定框架 |
| `decode_bench.sh` | `sglang` / `vllm` | SGLang 用 GSP / vLLM 用 prefix_repetition，`num_prompts = 2×并发` 稀释首请求 prefill |
| `decode_http_sweep.py` | **仅 Python 标准库** | 零第三方依赖，跨框架/跨厂商可用【仅供参考，实际测试不使用】 |
| `collect_results.py` | **仅 Python 标准库** | 扫描日志提取指标生成 CSV，所有脚本结束后自动调用 |
| `csv_to_md.py` | **仅 Python 标准库** | 将 CSV 转换为 Markdown 测试报告（三部分：结果表格 + 服务启动命令 + 测试命令），自动读取 `serve_command.txt`，找不到或为空则报错 |
| `serve_command_template.txt` | — | 模板文件，执行者复制为 `serve_command.txt` 并填写真实模型服务启动命令 |
| `benchmark_analysis_template.md` | — | Markdown 报告模板，包含三部分示例（结果表格 / 服务启动命令 / 测试命令） |

### 快速参考：服务端配置对照

| 测试场景 | SGLang 启动参数 | vLLM 启动参数 | 前缀缓存 | 数据集 | num_prompts |
|---|---|---|---|---|---|
| **第1章 Benchmark** | 默认 | 默认 | 开/关均可 | `random-ids` / `random` | = 并发数 |
| **第2章 纯 Prefill** | `--disable-radix-cache` | `--no-enable-prefix-caching` | **关** | `random-ids` / `random` | = 并发数 |
| **第2章 纯 Decode** | 默认（不加 `--disable-radix-cache`） | 默认（不加 `--no-enable-prefix-caching`） | **开** | `generated-shared-prefix` / `prefix_repetition` | = 2×并发数 |
