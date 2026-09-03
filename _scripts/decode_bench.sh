#!/bin/bash
set -euo pipefail

# ============================================================
# 纯 Decode 基准测试脚本（支持 SGLang / vLLM）
#
# 原理：共享前缀 + 前缀缓存。prefix 命中缓存 → prefill≈0，全部耗时是 decode
# 方法：共享前缀 + 1 新 token + 长输出，需开启前缀缓存
#
# 数据集选择（显式保证共享前缀 → 缓存命中）：
#   SGLang: generated-shared-prefix (GSP)
#     --gsp-num-groups 1                    所有请求共享同一前缀
#     --gsp-system-prompt-len $prefix_len   共享前缀长度
#     --gsp-question-len 1                  每请求 1 个新 token
#     --gsp-output-len $output_len           输出长度
#   vLLM:   prefix_repetition
#     --prefix-repetition-num-prefixes 1    所有请求共享同一前缀
#     --prefix-repetition-prefix-len $prefix_len  共享前缀长度
#     --prefix-repetition-suffix-len 1      每请求 1 个新 token
#     --prefix-repetition-output-len $output_len   输出长度
#
# 前置条件：服务端必须开启前缀缓存
#   SGLang: 默认开启（不要加 --disable-radix-cache）
#   vLLM:   默认开启（不要加 --no-enable-prefix-caching）
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

# 前缀长度列表（模拟 KV cache 已就位的上下文长度）
DEFAULT_PREFIX_LENS="4096,32768,65536"
# decode 输出长度列表
DEFAULT_OUTPUT_LENS="1024"
# 并发数列表
DEFAULT_CONCURRENCY="1,4,8,16,32,64,128"

# --- 参数解析 ---
usage() {
  cat <<EOF
Usage: $0 -F <sglang|vllm> [OPTIONS]

纯 Decode 测试：共享前缀 + 前缀缓存 + 长输出。
前置条件：服务端必须开启前缀缓存
  SGLang: 默认开启（不要加 --disable-radix-cache）
  vLLM:   默认开启（不要加 --no-enable-prefix-caching）

必选参数:
  -F, --framework FRAMEWORK    推理框架类型: sglang 或 vllm
  -u, --base-url URL            推理服务地址
  -m, --model-path PATH         模型路径
  -n, --served-model-name NAME  服务模型名
  -t, --chip-type TYPE          芯片类型(用于CSV列名后缀,如H100/B200)
  -T, --tester NAME             测试人员(用于报告目录层级和报告命名)
  -P, --pd MODE                 PD部署模式: agg(非PD分离) 或 disagg(PD分离)

可选参数:
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_decode_reports)
  -p, --prefix-lens LIST        前缀长度列表(逗号分隔) (default: $DEFAULT_PREFIX_LENS)
  -o, --output-lens LIST        输出长度列表(逗号分隔) (default: $DEFAULT_OUTPUT_LENS)
  -c, --concurrency LIST        并发数列表(逗号分隔) (default: $DEFAULT_CONCURRENCY)
  -s, --sleep SECONDS           每次测试间隔秒数    (default: $SLEEP_TIME)
  -f, --foreground              前台执行(默认后台)
  -h, --help                    显示帮助

示例（默认后台执行）:
  $0 -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan -P agg
  $0 -F vllm   -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 -T zhangsan -P disagg
  $0 -F sglang -f -u http://... -p 4096,16384 -o 1024 -c 1,8,64 -T zhangsan -P agg   # 前台执行
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -F|--framework)           FRAMEWORK="$2";            shift 2 ;;
    -u|--base-url)            BASE_URL="$2";             shift 2 ;;
    -m|--model-path)          MODEL_PATH="$2";           shift 2 ;;
    -n|--served-model-name)   SERVED_MODEL_NAME="$2";    shift 2 ;;
    -r|--report-dir)          REPORT_DIR="$2";           shift 2 ;;
    -p|--prefix-lens)         PREFIX_LENS_ARG="$2";      shift 2 ;;
    -o|--output-lens)         OUTPUT_LENS_ARG="$2";      shift 2 ;;
    -c|--concurrency)         CONCURRENCY_ARG="$2";      shift 2 ;;
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
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./sglang_decode_reports"
    BENCH_LABEL="SGLang"
    LOG_PREFIX="sglang_decode"
    ;;
  vllm|vl)
    FRAMEWORK="vllm"
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./vllm_decode_reports"
    BENCH_LABEL="vLLM"
    LOG_PREFIX="vllm_decode"
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

# --- 后台执行（默认） ---
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
    --prefix-lens "${PREFIX_LENS_ARG:-$DEFAULT_PREFIX_LENS}" \
    --output-lens "${OUTPUT_LENS_ARG:-$DEFAULT_OUTPUT_LENS}" \
    --concurrency "${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}" \
    --sleep "$SLEEP_TIME" \
    --chip-type "$CHIP_TYPE" \
    --tester "$TESTER" \
    --pd "$PD" \
    --foreground \
    > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "${BENCH_LABEL} decode benchmark 已在后台启动 (PID: $PID)"
  echo "日志文件: $LOG_FILE"
  echo "监控进度: tail -f \"$LOG_FILE\""
  echo "停止进程: kill $PID"
  exit 0
fi

# --- 构建列表 ---
PREFIX_LENS_STR="${PREFIX_LENS_ARG:-$DEFAULT_PREFIX_LENS}"
OUTPUT_LENS_STR="${OUTPUT_LENS_ARG:-$DEFAULT_OUTPUT_LENS}"
CONCURRENCY_STR="${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}"
IFS=',' read -ra prefix_lens <<< "$PREFIX_LENS_STR"
IFS=',' read -ra output_lens <<< "$OUTPUT_LENS_STR"
IFS=',' read -ra concurrency_list <<< "$CONCURRENCY_STR"

safe_model_name=$(echo "$SERVED_MODEL_NAME" | sed 's/.*[\/\\]//; s/[:\\]//g')

# --- 运行时间戳（作为本次测试的子目录名，避免覆盖） ---
RUN_TS=$(date +%Y%m%d_%H%M%S)

# --- 创建报告目录 ---
RUN_DIR="$REPORT_DIR/${TESTER}/${safe_model_name}/${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "=== Starting ${BENCH_LABEL} Pure Decode Benchmark ==="
echo "=== Framework:    $FRAMEWORK"
echo "=== Base URL:     $BASE_URL"
echo "=== Model Path:   $MODEL_PATH"
echo "=== Served Name:  $SERVED_MODEL_NAME"
echo "=== Report Dir:   $RUN_DIR"
echo "=== Prefix Lens:  ${prefix_lens[*]}"
echo "=== Output Lens:   ${output_lens[*]}"
echo "=== Concurrency:  ${concurrency_list[*]}"
if [[ "$FRAMEWORK" == "sglang" ]]; then
  echo "=== Dataset:      generated-shared-prefix (GSP)"
else
  echo "=== Dataset:      prefix_repetition"
fi
echo "=== 前缀缓存:      ON (必须开启) ==="
echo "=== Tester:       $TESTER"
echo "=== PD Mode:      $PD"
echo "=========================================="

for prefix_len in "${prefix_lens[@]}"; do
  for output_len in "${output_lens[@]}"; do
    for concurrency in "${concurrency_list[@]}"; do
      # num_prompts = 2 × 并发数，稀释首个请求的 prefill 开销
      num_prompts=$((concurrency * 2))
      log_file="${RUN_DIR}/decode_prefix-${prefix_len}-output-${output_len}-bs-${concurrency}.log"
      echo ""
      echo ">>> Decode: Prefix=$prefix_len, Output=$output_len, Concurrency=$concurrency, NumPrompts=$num_prompts"
      echo ">>> Log file: $log_file"

      if [[ "$FRAMEWORK" == "sglang" ]]; then
        # SGLang: generated-shared-prefix (GSP)
        # 所有请求共享同一前缀，保证缓存命中
        python -m sglang.bench_serving \
          --backend sglang-oai-chat \
          --base-url "$BASE_URL" \
          --model "$MODEL_PATH" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --dataset-name generated-shared-prefix \
          --gsp-num-groups 1 \
          --gsp-prompts-per-group "$num_prompts" \
          --gsp-system-prompt-len "$prefix_len" \
          --gsp-question-len 1 \
          --gsp-output-len "$output_len" \
          --num-prompts "$num_prompts" \
          --max-concurrency "$concurrency" \
          --seed "$SEED" \
          > "$log_file" 2>&1
      elif [[ "$FRAMEWORK" == "vllm" ]]; then
        # vLLM: prefix_repetition
        # num-prefixes=1 → 所有请求共享同一前缀
        vllm bench serve \
          --backend openai-chat \
          --endpoint /v1/chat/completions \
          --base-url "$BASE_URL" \
          --model "$MODEL_PATH" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --dataset-name prefix_repetition \
          --prefix-repetition-prefix-len "$prefix_len" \
          --prefix-repetition-suffix-len 1 \
          --prefix-repetition-num-prefixes 1 \
          --prefix-repetition-output-len "$output_len" \
          --num-prompts "$num_prompts" \
          --max-concurrency "$concurrency" \
          --trust-remote-code \
          --temperature 0.7 \
          --seed "$SEED" \
          --metric_percentiles 95,99 \
          --ready-check-timeout-sec 30 \
          > "$log_file" 2>&1
      fi

      if [ $? -ne 0 ]; then
        echo "!!! Error occurred. Check log file: $log_file"
      else
        echo ">>> Success. Results saved to: $log_file"
      fi
      echo "--- Finished Prefix=$prefix_len, Output=$output_len. Sleeping for $SLEEP_TIME seconds... ---"
      sleep "$SLEEP_TIME"
    done
  done
done

echo ""
echo "=== All decode tests finished ==="
echo "=== Check results in: $RUN_DIR ==="

# --- 构建 BENCH_COMMANDS（循环模板，变量占位，用于报告第三部分） ---
PREFIX_LIST="${prefix_lens[*]}"
OUTPUT_LIST="${output_lens[*]}"
CONC_LIST="${concurrency_list[*]}"

if [[ "$FRAMEWORK" == "sglang" ]]; then
  BENCH_COMMANDS="# 前缀长度列表: ${PREFIX_LENS_STR}
# 输出长度列表: ${OUTPUT_LENS_STR}
# 并发数列表: ${CONCURRENCY_STR}
for PREFIX_LEN in ${PREFIX_LIST}; do
  for OUTPUT_LEN in ${OUTPUT_LIST}; do
    for CONCURRENCY in ${CONC_LIST}; do
      NUM_PROMPTS=\$((CONCURRENCY * 2))
      python -m sglang.bench_serving \\
        --backend sglang-oai-chat \\
        --base-url \"\$BASE_URL\" \\
        --model \"\$MODEL_PATH\" \\
        --served-model-name \"\$SERVED_MODEL_NAME\" \\
        --dataset-name generated-shared-prefix \\
        --gsp-num-groups 1 \\
        --gsp-prompts-per-group \$NUM_PROMPTS \\
        --gsp-system-prompt-len \$PREFIX_LEN \\
        --gsp-question-len 1 \\
        --gsp-output-len \$OUTPUT_LEN \\
        --num-prompts \$NUM_PROMPTS \\
        --max-concurrency \$CONCURRENCY \\
        --seed ${SEED}
    done
  done
done"
elif [[ "$FRAMEWORK" == "vllm" ]]; then
  BENCH_COMMANDS="# 前缀长度列表: ${PREFIX_LENS_STR}
# 输出长度列表: ${OUTPUT_LENS_STR}
# 并发数列表: ${CONCURRENCY_STR}
for PREFIX_LEN in ${PREFIX_LIST}; do
  for OUTPUT_LEN in ${OUTPUT_LIST}; do
    for CONCURRENCY in ${CONC_LIST}; do
      NUM_PROMPTS=\$((CONCURRENCY * 2))
      vllm bench serve \\
        --backend openai-chat \\
        --endpoint /v1/chat/completions \\
        --base-url \"\$BASE_URL\" \\
        --model \"\$MODEL_PATH\" \\
        --served-model-name \"\$SERVED_MODEL_NAME\" \\
        --dataset-name prefix_repetition \\
        --prefix-repetition-prefix-len \$PREFIX_LEN \\
        --prefix-repetition-suffix-len 1 \\
        --prefix-repetition-num-prefixes 1 \\
        --prefix-repetition-output-len \$OUTPUT_LEN \\
        --num-prompts \$NUM_PROMPTS \\
        --max-concurrency \$CONCURRENCY \\
        --trust-remote-code \\
        --temperature 0.7 \\
        --seed ${SEED} \\
        --metric_percentiles 95,99 \\
        --ready-check-timeout-sec 30
    done
  done
done"
fi

# --- 自动收集结果生成 CSV ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${RUN_DIR}/decode_results.csv"
python3 "${SCRIPT_DIR}/collect_results.py" \
  --report-dir "$RUN_DIR" \
  --model-name "$SERVED_MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --chip-type "$CHIP_TYPE" \
  --out "$CSV_FILE"

# --- 自动生成 Markdown 测试报告 ---
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_${CHIP_TYPE}_${FRAMEWORK}_${PD}_decode_${RUN_TS}.md"
python3 "${SCRIPT_DIR}/csv_to_md.py" \
  --csv "$CSV_FILE" \
  --output "$MD_FILE" \
  --bench-command "$BENCH_COMMANDS"
