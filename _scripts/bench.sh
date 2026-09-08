#!/bin/bash
set -euo pipefail

# ============================================================
# Benchmark 基准测试脚本（支持 SGLang / vLLM）
#
# 通过 -F/--framework 指定推理框架，自动选择对应的 bench 命令分支：
#   sglang → python -m sglang.benchmark.serving
#   vllm   → vllm bench serve
#
# 改进点:
#   1. 合并 sglang_bench.sh 和 vllm_bench.sh，一个脚本支持多框架
#   2. 提取模型服务名的最后一段（以 / \ 分割）并清除残留分隔符（: \）作为目录名
#   3. BASE_URL / MODEL_PATH / SERVED_MODEL_NAME 支持命令行参数传递
#   4. 默认后台执行（自动 nohup + 日志重定向），-f/--foreground 切前台
#   5. 并发数和 IO 组合可通过参数覆盖
#   6. 测试结束后自动调用 collect_results.py 生成汇总 CSV
# ============================================================

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
PD=""
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
  -T, --tester NAME             测试人员(用于报告目录层级和报告命名)
  -P, --pd MODE                 PD部署模式: agg(非PD分离) 或 disagg(PD分离)

可选参数:
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_reports)
  -c, --concurrency LIST        并发数列表(逗号分隔) (default: $DEFAULT_CONCURRENCY)
  -i, --io-combinations LIST    IO组合(逗号分隔,每组"in out")
                                  (default: $DEFAULT_IO)
  -s, --sleep SECONDS           每次测试间隔秒数    (default: $SLEEP_TIME)
  -f, --foreground             前台执行(默认后台)
  -h, --help                    显示帮助

示例（默认后台执行）:
  $0 -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan -P agg
  $0 -F vllm   -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 -T zhangsan -P disagg
  $0 -F sglang -f -u http://... -m /path -n name -T zhangsan -P agg              # 前台执行
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
    -P|--pd)                  PD="$2";                   shift 2 ;;
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
[[ -z "$CHIP_TYPE" ]]         && MISSING+="  --chip-type / -t\n"
[[ -z "$TESTER" ]]            && MISSING+="  --tester / -T\n"
[[ -z "$PD" ]]                 && MISSING+="  --pd / -P\n"
if [[ -n "$MISSING" ]]; then
  echo "ERROR: 以下必选参数未指定:" >&2
  printf "%b" "$MISSING" >&2
  echo "" >&2
  usage
  exit 1
fi

# --- 校验 PD 参数 ---
case "$PD" in
  agg|disagg) ;;
  *)
    echo "ERROR: --pd 只支持 agg(非PD分离) 或 disagg(PD分离)，当前值: '$PD'" >&2
    exit 1
    ;;
esac

# --- 前置校验：serve_command.sh ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVE_CMD_FILE="${PROJECT_ROOT}/serve_command.sh"
if [[ ! -f "$SERVE_CMD_FILE" ]]; then
  echo "ERROR: 找不到 serve_command.sh，请先复制模板并填写模型服务启动命令：" >&2
  echo "  cp serve_command.sh.template serve_command.sh" >&2
  echo "  # 然后编辑 serve_command.sh 填写真实部署命令" >&2
  exit 1
fi
if [[ ! -s "$SERVE_CMD_FILE" ]]; then
  echo "ERROR: serve_command.sh 文件为空，请填写真实的模型服务启动命令。" >&2
  exit 1
fi

# --- 后台执行（默认）：带 --foreground 重新 nohup 自身 ---
if [[ "$BACKGROUND" == "true" ]]; then
  mkdir -p "$REPORT_DIR/${TESTER}"
  LOG_FILE="${REPORT_DIR}/${TESTER}/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
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
    --pd "$PD" \
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

# --- 运行时间戳（作为本次测试的子目录名，避免覆盖） ---
RUN_TS=$(date +%Y%m%d_%H%M%S)

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
echo "=== PD Mode:      $PD"
echo "=========================================="

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
    log_file="${RUN_DIR}/bench_bs-${num_prompts}_input_len-${input_len}-output_len-${output_len}.log"
    jsonl_file="${RUN_DIR}/bench_bs-${num_prompts}_input_len-${input_len}-output_len-${output_len}.jsonl"

    echo ""
    echo ">>> Running test: Input=$input_len, Output=$output_len, Concurrency=$concurrency"
    echo ">>> Log file: $log_file"

    # --- 根据框架选择对应的 bench 命令 ---
    if [[ "$FRAMEWORK" == "sglang" ]]; then
      python -m sglang.benchmark.serving \
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
        --output-file "$jsonl_file" \
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
    python -m sglang.benchmark.serving \\
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
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_${CHIP_TYPE}_${FRAMEWORK}_${PD}_bench_${RUN_TS}.md"
python3 "${SCRIPT_DIR}/csv_to_md.py" \
  --csv "$CSV_FILE" \
  --output "$MD_FILE" \
  --bench-command "$BENCH_COMMANDS"
