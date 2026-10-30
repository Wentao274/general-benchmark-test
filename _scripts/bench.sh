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
#   7. 实测前缀缓存命中率并写入报告
#      SGLang: --cache-report (L1/radix) + /metrics 的 prefill_effective_tokens_total
#              计数器差值（含 HiCache L1/L2/L3），后者覆盖前者
#      vLLM:   测试前后查询服务端 /metrics 的 vllm:prefix_cache_* 及
#              vllm:external_prefix_cache_hits 计数器取差值（L1 GPU / L3 external）
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

# 并发数列表（同时也是 num-prompts 的值）
DEFAULT_CONCURRENCY="1,4,8,16,32,64,128"
# 输入输出长度组合，格式 "in_len out_len"，组合之间逗号分隔
DEFAULT_IO="8192 512,32768 512,65536 512"

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
  -d, --describe TEXT           模型描述信息(可选,填入报告末尾)
  -f, --foreground             前台执行(默认后台)
  --timeout SECONDS            每组测试超时秒数        (default: $TIMEOUT)
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

# --- 检测可用的 Python 命令 ---
if command -v python3 &>/dev/null; then
  PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
  PYTHON_CMD="python"
else
  echo "ERROR: 找不到 python 或 python3 命令" >&2
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
    --timeout "$TIMEOUT" \
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

# ============================================================
# 缓存命中率实测（用于校验服务端是否按要求关闭了前缀缓存）
#   SGLang: 测试前后查询服务端 /metrics 的 sglang:prefill_effective_tokens_total
#           计数器（mode=input/device_hit/host_hit/storage_hit），差值计算
#           HiCache L1/L2/L3 三层命中率（--cache-report 仅统计 L1/radix）
#   vLLM:   测试前后查询服务端 /metrics 的
#           vllm:prefix_cache_queries / vllm:prefix_cache_hits 计数器（L1 GPU）
#           及 vllm:external_prefix_cache_hits 计数器（L3 external/LM Cache），
#           差值计算 L1/L3 两层命中率
# 命中率写入 bench 日志（"Cache hit rate:" + "L1/L3 cache hit rate:" 行），
# collect_results.py 解析
# ============================================================

# --- vLLM: 从服务端 /metrics 提取前缀缓存计数器（token 数，跨 engine 求和） ---
# L1 (GPU prefix cache): vllm:prefix_cache_queries/hits_total
# L3 (external/LM cache): vllm:external_prefix_cache_hits_total（L1 miss 后查 L3）
# 输出 "<queries> <l1_hits> <l3_hits>"；/metrics 不可用或无该指标时输出 "N/A"
# 注：无 LM Cache 的部署，l3_hits 为 0（不影响 L1 统计）
get_vllm_prefix_cache_counters() {
  curl -s --max-time 10 "$BASE_URL/metrics" 2>/dev/null | awk '
    function val(line,   s, a) {
      s = line
      sub(/\{.*\}/, "", s)           # 去掉 label（防 label 值含空格）
      sub(/^[^ \t]*[ \t]*/, "", s)   # 去掉指标名，剩余为数值
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

# --- vLLM: 等待计数器稳定（服务端统计有刷新延迟），输出稳定值 "<queries> <l1_hits> <l3_hits>" ---
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

# --- vLLM: 用测试前后计数器差值计算真实命中率（L1 GPU + L3 external），附加到 bench 日志 ---
# 输出格式与 SGLang /metrics 报告对齐（"Cache hit rate:" + "L1/L3 cache hit rate:" 行），便于统一解析
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
    # 服务端未执行任何前缀缓存查询（前缀缓存已关闭时的典型表现，命中率即 0%）
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

# --- SGLang: 从服务端 /metrics 提取 prefill_effective_tokens_total 计数器（token 数，跨 rank 求和） ---
# 输出 "<input> <device_hit> <host_hit> <storage_hit>"；/metrics 不可用或无该指标时输出 "N/A"
# mode=input → 需要计算 prefill 的 token 数（缓存未命中）
# mode=device_hit → L1 (GPU) 缓存命中
# mode=host_hit   → L2 (CPU) 缓存命中
# mode=storage_hit → L3 (DFKV) 缓存命中
get_sglang_cache_counters() {
  curl -s --max-time 10 "$BASE_URL/metrics" 2>/dev/null | awk '
    function val(line,   s, a) {
      s = line
      sub(/\{.*\}/, "", s)           # 去掉 label（防 label 值含空格）
      sub(/^[^ \t]*[ \t]*/, "", s)   # 去掉指标名，剩余为数值
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

# --- SGLang: 等待计数器稳定（服务端统计有刷新延迟），输出稳定值 ---
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

# --- SGLang: 用测试前后计数器差值计算真实命中率（含 HiCache L1/L2/L3），附加到 bench 日志 ---
# 输出格式与 vLLM 的 append_vllm_cache_report 对齐（"Cache hit rate:" 行），便于统一解析
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
echo "=== Cache Stats:  实测缓存命中率（SGLang HiCache L1/L2/L3 / vLLM L1/L3，写入报告）"
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

    # --- 测试前抓取服务端前缀缓存计数器（与测试后差值计算真实命中率） ---
    if [[ "$FRAMEWORK" == "vllm" ]]; then
      vllm_pc_before="$(wait_vllm_prefix_cache_counters)"
    elif [[ "$FRAMEWORK" == "sglang" ]]; then
      sglang_pc_before="$(wait_sglang_cache_counters)"
    fi

    # --- 根据框架选择对应的 bench 命令 ---
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
        --cache-report \
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

    # 检查命令执行状态
    if [ $bench_rc -eq 124 ]; then
      echo "!!! TIMEOUT: test killed after ${TIMEOUT}s. Check log file: $log_file"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    elif [ $bench_rc -ne 0 ]; then
      echo "!!! Error occurred (exit code: $bench_rc). Check log file: $log_file"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    else
      # 退出码为 0 也可能全部请求失败，扫描日志错误模式
      error_pattern="$(check_log_errors "$log_file")" || true
      if [[ -n "$error_pattern" ]]; then
        echo "!!! Log error detected: '$error_pattern'. Check log file: $log_file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
      else
        echo ">>> Success. Results saved to: $log_file"
        # --- 计算真实前缀缓存命中率并附加到日志（供 collect_results.py 解析） ---
        # SGLang: /metrics 的 prefill_effective_tokens_total 计数器差值（含 HiCache L1/L2/L3）
        #         覆盖 --cache-report 仅统计 L1 (radix cache) 的局限性
        # vLLM:   /metrics 的 vllm:prefix_cache_* 计数器差值
        if [[ "$FRAMEWORK" == "vllm" ]]; then
          append_vllm_cache_report "$log_file" "$vllm_pc_before"
        elif [[ "$FRAMEWORK" == "sglang" ]]; then
          append_sglang_cache_report "$log_file" "$sglang_pc_before"
        fi
      fi
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
# 注: 脚本在每组测试前后查询服务端 /metrics 的 sglang:prefill_effective_tokens_total
#     计数器，差值计算真实前缀缓存命中率（含 HiCache L1/L2/L3）并写入日志
for CONCURRENCY in ${CONC_LIST}; do
  NUM_PROMPTS=\$CONCURRENCY
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
      --seed ${SEED} \\
      --cache-report
  done
done"
elif [[ "$FRAMEWORK" == "vllm" ]]; then
  BENCH_COMMANDS="# 并发数列表: ${CONCURRENCY_STR}
# IO 组合: ${IO_STR}
# 注: 脚本在每组测试前后查询服务端 /metrics 的 vllm:prefix_cache_* 计数器，
#     差值计算真实前缀缓存命中率并写入日志（报告中 L1命中率/L3命中率列为实测值）
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
      --metric_percentiles 90,95,99 \\
      --ready-check-timeout-sec 30
  done
done"
fi

# --- 自动收集结果生成 CSV ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${RUN_DIR}/results.csv"
"$PYTHON_CMD" "${SCRIPT_DIR}/collect_results.py" \
  --report-dir "$RUN_DIR" \
  --model-name "$SERVED_MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --chip-type "$CHIP_TYPE" \
  --out "$CSV_FILE"

# --- 自动生成 Markdown 测试报告 ---
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_${CHIP_TYPE}_${FRAMEWORK}_${PD}_bench_${RUN_TS}.md"
if [[ -n "$DESCRIBE" ]]; then
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "基础 Benchmark 测试" \
    --bench-command "$BENCH_COMMANDS" \
    --describe "$DESCRIBE"
else
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "基础 Benchmark 测试" \
    --bench-command "$BENCH_COMMANDS"
fi
