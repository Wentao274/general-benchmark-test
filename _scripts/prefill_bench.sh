#!/bin/bash
set -euo pipefail

# ============================================================
# 纯 Prefill 基准测试脚本（支持 SGLang / vLLM）
#
# 原理：output_len=1 → 首 token 是 prefill 前向的副产品，0 次 decode 迭代
# 前置条件：服务端必须关闭前缀缓存
#   SGLang: 启动加 --disable-radix-cache
#   vLLM:   启动加 --no-enable-prefix-caching
# ============================================================

# --- 默认配置 ---
FRAMEWORK=""
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL_PATH="${MODEL_PATH:-/data1/DeepSeek-V4-Flash-INT8-Channel}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4-flash}"
SEED=123
SLEEP_TIME=60
REPORT_DIR=""
CHIP_TYPE=""
BACKGROUND=true

DEFAULT_CONCURRENCY="1,4,8,16,32,64,128"
DEFAULT_IO="65536 1"

# --- 参数解析 ---
usage() {
  cat <<EOF
Usage: $0 -F <sglang|vllm> [OPTIONS]

纯 Prefill 测试：output_len=1，input_len 递增。
前置条件：服务端必须关闭前缀缓存
  SGLang: 启动加 --disable-radix-cache
  vLLM:   启动加 --no-enable-prefix-caching

必选参数:
  -F, --framework FRAMEWORK    推理框架类型: sglang 或 vllm

可选参数:
  -u, --base-url URL            推理服务地址        (default: $BASE_URL)
  -m, --model-path PATH         模型路径            (default: $MODEL_PATH)
  -n, --served-model-name NAME  服务模型名          (default: $SERVED_MODEL_NAME)
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_prefill_reports)
  -c, --concurrency LIST        并发数列表(逗号分隔) (default: $DEFAULT_CONCURRENCY)
  -i, --io-combinations LIST    IO组合(逗号分隔,每组"in out") (default: $DEFAULT_IO)
  -s, --sleep SECONDS           每次测试间隔秒数    (default: $SLEEP_TIME)
  -f, --foreground              前台执行(默认后台)
  -t, --chip-type TYPE          芯片类型(用于CSV列名后缀,如H100/B200)
  -h, --help                    显示帮助

示例（默认后台执行）:
  $0 -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100
  $0 -F vllm   -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100
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
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./sglang_prefill_reports"
    BENCH_LABEL="SGLang"
    LOG_PREFIX="sglang_prefill"
    ;;
  vllm|vl)
    FRAMEWORK="vllm"
    DATASET_NAME="random"
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./vllm_prefill_reports"
    BENCH_LABEL="vLLM"
    LOG_PREFIX="vllm_prefill"
    ;;
  *)
    echo "ERROR: 不支持的框架 '$FRAMEWORK'，请使用 sglang 或 vllm" >&2
    exit 1
    ;;
esac

# --- 后台执行（默认） ---
if [[ "$BACKGROUND" == "true" ]]; then
  mkdir -p "$REPORT_DIR"
  LOG_FILE="${REPORT_DIR}/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
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
     --foreground \
    > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "${BENCH_LABEL} prefill benchmark 已在后台启动 (PID: $PID)"
  echo "日志文件: $LOG_FILE"
  echo "监控进度: tail -f \"$LOG_FILE\""
  echo "停止进程: kill $PID"
  exit 0
fi

# --- 构建列表 ---
CONCURRENCY_STR="${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}"
IO_STR="${IO_ARG:-$DEFAULT_IO}"
IFS=',' read -ra concurrency_list <<< "$CONCURRENCY_STR"
IFS=',' read -ra io_combinations <<< "$IO_STR"

safe_model_name=$(echo "$SERVED_MODEL_NAME" | sed 's/.*[\/\\]//; s/[:\\]//g')
mkdir -p "$REPORT_DIR/${safe_model_name}"

echo "=========================================="
echo "=== Starting ${BENCH_LABEL} Pure Prefill Benchmark ==="
echo "=== Framework:    $FRAMEWORK"
echo "=== Base URL:     $BASE_URL"
echo "=== Model Path:   $MODEL_PATH"
echo "=== Served Name:  $SERVED_MODEL_NAME"
echo "=== Report Dir:   $REPORT_DIR/${safe_model_name}"
echo "=== output_len=1 (pure prefill, 0 decode) ==="
echo "=========================================="

# --- 累积实际执行的 bench 工具命令（用于报告第三部分） ---
BENCH_COMMANDS=""

for concurrency in "${concurrency_list[@]}"; do
  num_prompts=$concurrency
  echo ""
  echo "--- Concurrency=$concurrency, Num Prompts=$num_prompts ---"

  for combo in "${io_combinations[@]}"; do
    input_len=$(echo "$combo" | awk '{print $1}')
    output_len=$(echo "$combo" | awk '{print $2}')
    log_file="${REPORT_DIR}/${safe_model_name}/prefill_input-${input_len}-bs-${num_prompts}.log"
    echo ""
    echo ">>> Prefill: Input=$input_len, Output=$output_len, Concurrency=$concurrency"
    echo ">>> Log file: $log_file"

    if [[ "$FRAMEWORK" == "sglang" ]]; then
      BENCH_CMD="python -m sglang.bench_serving \\
  --backend sglang-oai-chat \\
  --base-url \"$BASE_URL\" \\
  --model \"$MODEL_PATH\" \\
  --served-model-name \"$SERVED_MODEL_NAME\" \\
  --dataset-name \"$DATASET_NAME\" \\
  --random-input-len $input_len \\
  --random-output-len $output_len \\
  --random-range-ratio 1.0 \\
  --num-prompts $num_prompts \\
  --max-concurrency $concurrency \\
  --seed $SEED"
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
      BENCH_CMD="vllm bench serve \\
  --backend openai-chat \\
  --endpoint /v1/chat/completions \\
  --base-url \"$BASE_URL\" \\
  --model \"$MODEL_PATH\" \\
  --served-model-name \"$SERVED_MODEL_NAME\" \\
  --dataset-name \"$DATASET_NAME\" \\
  --random-input-len $input_len \\
  --random-output-len $output_len \\
  --num-prompts $num_prompts \\
  --max-concurrency $concurrency \\
  --trust-remote-code \\
  --temperature 0.7 \\
  --random-range-ratio 0.0 \\
  --random-prefix-len 0 \\
  --seed $SEED \\
  --metric_percentiles 95,99 \\
  --ready-check-timeout-sec 30"
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

    # --- 累积 bench 工具命令用于报告 ---
    BENCH_COMMANDS="${BENCH_COMMANDS}# Input=$input_len Output=$output_len Concurrency=$concurrency
${BENCH_CMD}

"

    if [ $? -ne 0 ]; then
      echo "!!! Error occurred. Check log file: $log_file"
    else
      echo ">>> Success. Results saved to: $log_file"
    fi
    echo "--- Finished Input=$input_len. Sleeping for $SLEEP_TIME seconds... ---"
    sleep "$SLEEP_TIME"
  done
  echo "--- Completed all IO combinations for Concurrency=$concurrency ---"
done

echo ""
echo "=== All prefill tests finished ==="
echo "=== Check results in: $REPORT_DIR/${safe_model_name} ==="

# --- 自动收集结果生成 CSV ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${REPORT_DIR}/${safe_model_name}/prefill_results.csv"
python3 "${SCRIPT_DIR}/collect_results.py" \
  --report-dir "$REPORT_DIR/${safe_model_name}" \
  --model-name "$SERVED_MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --chip-type "$CHIP_TYPE" \
  --out "$CSV_FILE"

# --- 自动生成 Markdown 测试报告 ---
MD_FILE="${CSV_FILE%.csv}_report.md"
python3 "${SCRIPT_DIR}/csv_to_md.py" \
  --csv "$CSV_FILE" \
  --output "$MD_FILE" \
  --bench-command "$BENCH_COMMANDS"
