#!/bin/bash
set -euo pipefail

# ============================================================
# 纯 Prefill 基准测试脚本（支持 SGLang / vLLM）
#
# 原理：output_len=1 → 首 token 是 prefill 前向的副产品，0 次 decode 迭代
# 前置条件：服务端必须关闭前缀缓存
#   SGLang: 启动加 --disable-radix-cache
#   vLLM:   启动加 --no-enable-prefix-caching
# 缓存验证：测试前后查询 /metrics 计数器，实测命中率应 ≈0%
#   （若 >0% 说明缓存未关干净，prefill 吞吐量被缓存命中"污染"）
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
DESCRIBE=""
BACKGROUND=true
TIMEOUT=3600

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
  -u, --base-url URL            推理服务地址
  -m, --model-path PATH         模型路径
  -n, --served-model-name NAME  服务模型名
  -t, --chip-type TYPE          芯片类型(用于CSV列名后缀,如H100/B200)
  -T, --tester NAME             测试人员(用于报告目录层级和报告命名)
  -P, --pd MODE                 PD部署模式: agg(非PD分离) 或 disagg(PD分离)

可选参数:
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_prefill_reports)
  -c, --concurrency LIST        并发数列表(逗号分隔) (default: $DEFAULT_CONCURRENCY)
  -i, --io-combinations LIST    IO组合(逗号分隔,每组"in out") (default: $DEFAULT_IO)
  -s, --sleep SECONDS           每次测试间隔秒数    (default: $SLEEP_TIME)
  -d, --describe TEXT           模型描述信息(可选,填入报告末尾)
  -f, --foreground              前台执行(默认后台)
  --timeout SECONDS             每组测试超时秒数        (default: $TIMEOUT)
  -h, --help                    显示帮助

示例（默认后台执行）:
  $0 -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan -P agg
  $0 -F vllm   -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 -T zhangsan -P disagg
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
    -d|--describe)            DESCRIBE="$2";             shift 2 ;;
    -f|--foreground)          BACKGROUND=false;          shift   ;;
    --timeout)                TIMEOUT="$2";              shift 2 ;;
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

# --- 检测可用的 Python 命令 ---
if command -v python3 &>/dev/null; then
  PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
  PYTHON_CMD="python"
else
  echo "ERROR: 找不到 python 或 python3 命令" >&2
  exit 1
fi

# --- 后台执行（默认） ---
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
    --timeout "$TIMEOUT" \
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

# --- 运行时间戳（作为本次测试的子目录名，避免覆盖） ---
RUN_TS=$(date +%Y%m%d_%H%M%S)

# --- 创建报告目录 ---
RUN_DIR="$REPORT_DIR/${TESTER}/${safe_model_name}/${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "=== Starting ${BENCH_LABEL} Pure Prefill Benchmark ==="
echo "=== Framework:    $FRAMEWORK"
echo "=== Base URL:     $BASE_URL"
echo "=== Model Path:   $MODEL_PATH"
echo "=== Served Name:  $SERVED_MODEL_NAME"
echo "=== Report Dir:   $RUN_DIR"
echo "=== output_len=1 (pure prefill, 0 decode) ==="
echo "=== Tester:       $TESTER"
echo "=== PD Mode:      $PD"
echo "=========================================="

# --- 检测日志中的错误模式（即使 bench 退出码为 0 也可能全部请求失败） ---
check_log_errors() {
  local logfile="$1"
  local patterns=(
    "All requests failed"
    "Traceback (most recent call last)"
    "ConnectionError"
    "Connection refused"
    "IndexError"
    "requests.exceptions"
    "httpx.ConnectError"
  )
  for pat in "${patterns[@]}"; do
    if grep -q "$pat" "$logfile" 2>/dev/null; then
      echo "$pat"
      return 0
    fi
  done
  return 1
}

FAIL_COUNT=0

# ============================================================
# 前缀缓存命中率实测（/metrics 计数器差值法）
#   验证服务端缓存是否已按要求关闭（预期 0%）
#   vLLM:   vllm:prefix_cache_queries / vllm:prefix_cache_hits / vllm:external_prefix_cache_hits
#           含 L1(GPU) / L3(external/LM cache) 两层命中统计
#   SGLang: sglang:prefill_effective_tokens_total (mode=input/device_hit/host_hit/storage_hit)
#           含 HiCache L1(GPU) / L2(CPU) / L3(DFKV) 全层命中统计
# ============================================================

get_vllm_prefix_cache_counters() {
  curl -s --max-time 10 "$BASE_URL/metrics" 2>/dev/null | awk '
    function val(line,   s, a) {
      s = line
      sub(/\{.*\}/, "", s)
      sub(/^[^ \t]*[ \t]*/, "", s)
      if (s == "") return 0
      split(s, a, /[ \t]+/)
      return a[1] + 0
    }
    /^vllm:(gpu_)?prefix_cache_queries(_total)?[ \t{]/ { q += val($0); found = 1 }
    /^vllm:(gpu_)?prefix_cache_hits(_total)?[ \t{]/    { h += val($0); found = 1 }
    /^vllm:external_prefix_cache_hits(_total)?[ \t{]/  { l3h += val($0) }
    END {
      if (found) printf "%d %d %d", q, h, l3h
      else print "N/A"
    }
  ' || true
}

wait_vllm_prefix_cache_counters() {
  local prev cur i
  prev="$(get_vllm_prefix_cache_counters)"
  for i in 1 2 3 4 5 6; do
    sleep 5
    cur="$(get_vllm_prefix_cache_counters)"
    [[ "$cur" == "$prev" ]] && break
    prev="$cur"
  done
  echo "$prev"
}

append_vllm_cache_report() {
  local log_file="$1" before="$2"
  local after q_b l1h_b l3h_b q_a l1h_a l3h_a
  local q_d l1h_d l3h_d total_hit rate l1_rate l3_rate
  after="$(wait_vllm_prefix_cache_counters)"
  if [[ "$before" == "N/A" || "$after" == "N/A" ]]; then
    echo ">>> [WARN] 无法从 ${BASE_URL}/metrics 获取前缀缓存计数器，本组命中率记为 N/A"
    {
      echo ""
      echo "--------------- Prefix Cache Hit Details (vLLM /metrics) ---------------"
      echo "Cache hit rate:                       N/A (prefix cache counters unavailable from server /metrics)"
    } >> "$log_file"
    return
  fi
  read -r q_b l1h_b l3h_b <<< "$before" || true
  read -r q_a l1h_a l3h_a <<< "$after" || true
  q_d=$(( q_a - q_b ))
  l1h_d=$(( l1h_a - l1h_b ))
  l3h_d=$(( l3h_a - l3h_b ))
  total_hit=$(( l1h_d + l3h_d ))
  if [[ "$q_d" -le 0 ]]; then
    {
      echo ""
      echo "--------------- Prefix Cache Hit Details (vLLM /metrics) ---------------"
      echo "Prefix cache queries (tokens):        0"
      echo "Total cache hit tokens:               0"
      echo "Total prefill tokens:                 0"
      echo "Cache hit rate:                       0.0% (server reported no prefix cache activity)"
      echo "L1 cache hit rate (device):           0.0%"
      echo "L3 cache hit rate (external):         0.0%"
    } >> "$log_file"
    return
  fi
  rate=$(awk -v h="$total_hit" -v q="$q_d" 'BEGIN { printf "%.1f", h * 100 / q }')
  l1_rate=$(awk -v h="$l1h_d" -v q="$q_d" 'BEGIN { printf "%.1f", h * 100 / q }')
  l3_rate=$(awk -v h="$l3h_d" -v q="$q_d" 'BEGIN { printf "%.1f", h * 100 / q }')
  {
    echo ""
    echo "--------------- Prefix Cache Hit Details (vLLM /metrics) ---------------"
    echo "Prefix cache queries (tokens):        ${q_d}"
    echo "Prefix cache hits (L1 GPU):           ${l1h_d}"
    echo "External prefix cache hits (L3 disk): ${l3h_d}"
    echo "Total cache hit tokens:               ${total_hit}"
    echo "Total prefill tokens:                 ${q_d}"
    echo "Cache hit rate:                       ${rate}%"
    echo "L1 cache hit rate (device):           ${l1_rate}%"
    echo "L3 cache hit rate (external):         ${l3_rate}%"
  } >> "$log_file"
  echo ">>> vLLM /metrics cache hit rate: ${rate}% (L1=${l1_rate}% L3=${l3_rate}%, hits ${total_hit} / queries ${q_d} tokens)"
}

get_sglang_cache_counters() {
  curl -s --max-time 10 "$BASE_URL/metrics" 2>/dev/null | awk '
    function val(line,   s, a) {
      s = line
      sub(/\{.*\}/, "", s)
      sub(/^[^ \t]*[ \t]*/, "", s)
      if (s == "") return 0
      split(s, a, /[ \t]+/)
      return a[1] + 0
    }
    /^sglang:prefill_effective_tokens_total\{/ && /mode="input"/        { inp += val($0); found = 1 }
    /^sglang:prefill_effective_tokens_total\{/ && /mode="device_hit"/   { dev += val($0); found = 1 }
    /^sglang:prefill_effective_tokens_total\{/ && /mode="host_hit"/     { host += val($0); found = 1 }
    /^sglang:prefill_effective_tokens_total\{/ && /mode="storage_hit"/  { stor += val($0); found = 1 }
    END {
      if (found) printf "%d %d %d %d", inp, dev, host, stor
      else print "N/A"
    }
  ' || true
}

wait_sglang_cache_counters() {
  local prev cur i
  prev="$(get_sglang_cache_counters)"
  for i in 1 2 3 4 5 6; do
    sleep 5
    cur="$(get_sglang_cache_counters)"
    [[ "$cur" == "$prev" ]] && break
    prev="$cur"
  done
  echo "$prev"
}

append_sglang_cache_report() {
  local log_file="$1" before="$2"
  local after inp_b dev_b host_b stor_b inp_a dev_a host_a stor_a
  local inp_d dev_d host_d stor_d total_hit total_all rate l1_rate l2_rate l3_rate
  after="$(wait_sglang_cache_counters)"
  if [[ "$before" == "N/A" || "$after" == "N/A" ]]; then
    echo ">>> [WARN] 无法从 ${BASE_URL}/metrics 获取 SGLang 缓存计数器，本组命中率记为 N/A"
    {
      echo ""
      echo "--------------- Prefix Cache Hit Details (SGLang /metrics) ---------------"
      echo "Cache hit rate:                       N/A (sglang prefill_effective_tokens_total unavailable from server /metrics)"
    } >> "$log_file"
    return
  fi
  read -r inp_b dev_b host_b stor_b <<< "$before" || true
  read -r inp_a dev_a host_a stor_a <<< "$after" || true
  inp_d=$(( inp_a - inp_b ))
  dev_d=$(( dev_a - dev_b ))
  host_d=$(( host_a - host_b ))
  stor_d=$(( stor_a - stor_b ))
  total_hit=$(( dev_d + host_d + stor_d ))
  total_all=$(( inp_d + total_hit ))
  if [[ "$total_all" -le 0 ]]; then
    {
      echo ""
      echo "--------------- Prefix Cache Hit Details (SGLang /metrics) ---------------"
      echo "Prefill effective tokens (input):     0"
      echo "Total cache hit tokens:               0"
      echo "Total prefill tokens:                 0"
      echo "Cache hit rate:                       0.0% (server reported no prefill activity)"
      echo "L1 cache hit rate (device):           0.0%"
      echo "L2 cache hit rate (host):             0.0%"
      echo "L3 cache hit rate (storage):          0.0%"
    } >> "$log_file"
    return
  fi
  rate=$(awk -v h="$total_hit" -v a="$total_all" 'BEGIN { printf "%.1f", h * 100 / a }')
  l1_rate=$(awk -v h="$dev_d" -v a="$total_all" 'BEGIN { printf "%.1f", h * 100 / a }')
  l2_rate=$(awk -v h="$host_d" -v a="$total_all" 'BEGIN { printf "%.1f", h * 100 / a }')
  l3_rate=$(awk -v h="$stor_d" -v a="$total_all" 'BEGIN { printf "%.1f", h * 100 / a }')
  {
    echo ""
    echo "--------------- Prefix Cache Hit Details (SGLang /metrics) ---------------"
    echo "Prefill effective tokens (input):     ${inp_d}"
    echo "Cache hit tokens (L1 device):         ${dev_d}"
    echo "Cache hit tokens (L2 host):           ${host_d}"
    echo "Cache hit tokens (L3 storage):        ${stor_d}"
    echo "Total cache hit tokens:               ${total_hit}"
    echo "Total prefill tokens:                 ${total_all}"
    echo "Cache hit rate:                       ${rate}%"
    echo "L1 cache hit rate (device):           ${l1_rate}%"
    echo "L2 cache hit rate (host):             ${l2_rate}%"
    echo "L3 cache hit rate (storage):          ${l3_rate}%"
  } >> "$log_file"
  echo ">>> SGLang /metrics cache hit rate: ${rate}% (L1=${l1_rate}% L2=${l2_rate}% L3=${l3_rate}%, hits ${total_hit} / total ${total_all} tokens)"
}

for concurrency in "${concurrency_list[@]}"; do
  num_prompts=$((concurrency * 5))
  echo ""
  echo "--- Concurrency=$concurrency, Num Prompts=$num_prompts ---"

  for combo in "${io_combinations[@]}"; do
    input_len=$(echo "$combo" | awk '{print $1}')
    output_len=$(echo "$combo" | awk '{print $2}')
    log_file="${RUN_DIR}/prefill_bs-${concurrency}_input-${input_len}.log"
    jsonl_file="${RUN_DIR}/prefill_bs-${concurrency}_input-${input_len}.jsonl"
    echo ""
    echo ">>> Prefill: Input=$input_len, Output=$output_len, Concurrency=$concurrency"
    echo ">>> Log file: $log_file"

    # --- 测试前抓取服务端前缀缓存计数器（验证缓存已关闭，预期 0%） ---
    if [[ "$FRAMEWORK" == "vllm" ]]; then
      cache_pc_before="$(wait_vllm_prefix_cache_counters)"
    elif [[ "$FRAMEWORK" == "sglang" ]]; then
      cache_pc_before="$(wait_sglang_cache_counters)"
    fi

    set +e
    if [[ "$FRAMEWORK" == "sglang" ]]; then
    timeout "$TIMEOUT" "$PYTHON_CMD" -m sglang.benchmark.serving \
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
      timeout "$TIMEOUT" vllm bench serve \
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
        --metric_percentiles 90,95,99 \
        --ready-check-timeout-sec 30 \
        > "$log_file" 2>&1
    fi
    bench_rc=$?
    set -e

    if [ $bench_rc -eq 124 ]; then
      echo "!!! TIMEOUT: test killed after ${TIMEOUT}s. Check log file: $log_file"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    elif [ $bench_rc -ne 0 ]; then
      echo "!!! Error occurred (exit code: $bench_rc). Check log file: $log_file"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    else
      error_pattern="$(check_log_errors "$log_file")" || true
      if [[ -n "$error_pattern" ]]; then
        echo "!!! Log error detected: '$error_pattern'. Check log file: $log_file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
      else
        echo ">>> Success. Results saved to: $log_file"
        # --- 实测前缀缓存命中率，验证缓存已关闭（预期 0%） ---
        if [[ "$FRAMEWORK" == "vllm" ]]; then
          append_vllm_cache_report "$log_file" "$cache_pc_before"
        elif [[ "$FRAMEWORK" == "sglang" ]]; then
          append_sglang_cache_report "$log_file" "$cache_pc_before"
        fi
      fi
    fi
    echo "--- Finished Input=$input_len. Sleeping for $SLEEP_TIME seconds... ---"
    sleep "$SLEEP_TIME"
  done
  echo "--- Completed all IO combinations for Concurrency=$concurrency ---"
done

echo ""
echo "=== All prefill tests finished ==="
echo "=== Check results in: $RUN_DIR ==="
if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "=== WARNING: $FAIL_COUNT test(s) failed. Partial results will still be collected. ==="
fi

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
  NUM_PROMPTS=\$((CONCURRENCY * 5))
  for IO in ${IO_QUOTED}; do
    INPUT_LEN=\$(echo \"\$IO\" | awk '{print \$1}')
    OUTPUT_LEN=\$(echo \"\$IO\" | awk '{print \$2}')
    $PYTHON_CMD -m sglang.benchmark.serving \\
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
  NUM_PROMPTS=\$((CONCURRENCY * 5))
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
      --metric_percentiles 90,95,99 \\
      --ready-check-timeout-sec 30
  done
done"
fi

# --- 自动收集结果生成 CSV ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${RUN_DIR}/prefill_results.csv"
"$PYTHON_CMD" "${SCRIPT_DIR}/collect_results.py" \
  --report-dir "$RUN_DIR" \
  --model-name "$SERVED_MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --chip-type "$CHIP_TYPE" \
  --out "$CSV_FILE"

# --- 自动生成 Markdown 测试报告 ---
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_${CHIP_TYPE}_${FRAMEWORK}_${PD}_prefill_${RUN_TS}.md"
if [[ -n "$DESCRIBE" ]]; then
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "纯 Prefill 测试" \
    --bench-command "$BENCH_COMMANDS" \
    --describe "$DESCRIBE"
else
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "纯 Prefill 测试" \
    --bench-command "$BENCH_COMMANDS"
fi
