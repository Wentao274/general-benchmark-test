#!/bin/bash
set -euo pipefail

# ============================================================
# 指定缓存命中率基准测试脚本（支持 SGLang / vLLM）
#
# 原理：前缀级缓存命中
#   每个请求的输入 = 共享前缀（命中部分）+ 唯一后缀（未命中部分）
#   缓存命中率 = 共享前缀长度 / 输入总长度 × 100%
#
#   例如 90% 命中率：
#     输入长度 65536 = 共享前缀 58982 (90%) + 唯一后缀 6554 (10%)
#     所有请求共享同一段前缀 → 前缀部分命中缓存（prefill ≈ 0）
#     每个请求有不同的后缀 → 后缀部分未命中（prefill 10%）
#     每个请求都需要 prefill 10% 的输入，TTFT 均匀分布
#
# 两阶段测试：
#   阶段 1 — 预热（不计量）：
#     用 concurrency=1 发送 1 个请求，将共享前缀填入缓存
#   阶段 2 — 测试（计量）：
#     用目标并发发送 N_total = 5 × 并发数 个请求
#     所有请求共享同一前缀（命中），各有不同后缀（未命中）
#
# 数据集选择：
#   SGLang: generated-shared-prefix (GSP)
#     --gsp-num-groups 1               只有 1 个共享前缀组
#     --gsp-prompts-per-group N_total  每组 N_total 个请求（各有不同 question）
#     --gsp-system-prompt-len PL       共享前缀长度 = 输入 × 命中率
#     --gsp-question-len QL            每请求唯一后缀长度 = 输入 × (1-命中率)
#     --gsp-output-len 512             输出长度
#   vLLM: prefix_repetition
#     --prefix-repetition-num-prefixes 1   只有 1 个共享前缀
#     --prefix-repetition-prefix-len PL    共享前缀长度
#     --prefix-repetition-suffix-len SL    每请求唯一后缀长度（随机生成，每请求不同）
#     --prefix-repetition-output-len 512   输出长度
#
# 输入长度 = 共享前缀长度 + 唯一后缀长度 = 65536
# 输出长度 = 512
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

# 缓存命中率（百分比），默认 90
DEFAULT_CACHE_HIT_RATE=90
# 固定输入输出组合：64K 输入，512 输出
INPUT_LEN=65536
OUTPUT_LEN=512
# 并发数列表
DEFAULT_CONCURRENCY="1,4,8,16,32,64,128"

# --- 参数解析 ---
usage() {
  cat <<EOF
Usage: $0 -F <sglang|vllm> [OPTIONS]

指定缓存命中率测试（前缀级）：每个请求的前缀命中缓存，后缀未命中。
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
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_cache_reports)
  -c, --concurrency LIST        并发数列表(逗号分隔) (default: $DEFAULT_CONCURRENCY)
  -R, --cache-hit-rate PCT      缓存命中率百分比     (default: $DEFAULT_CACHE_HIT_RATE)
                                  如 90 表示每个请求 90% 前缀命中、10% 后缀未命中
                                  共享前缀长度 = 输入长度 × PCT/100
                                  唯一后缀长度 = 输入长度 × (1-PCT/100)
  -s, --sleep SECONDS           每次测试间隔秒数    (default: $SLEEP_TIME)
  -d, --describe TEXT           模型描述信息(可选,填入报告末尾)
  -f, --foreground              前台执行(默认后台)
  --timeout SECONDS             每组测试超时秒数        (default: $TIMEOUT)
  -h, --help                    显示帮助

示例（默认后台执行）:
  $0 -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan -P agg
  $0 -F vllm   -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 -T zhangsan -P disagg
  $0 -F sglang -f -u http://... -m /path -n name -T zhangsan -P agg -R 95   # 95%命中率,前台执行
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
    -R|--cache-hit-rate)      CACHE_HIT_RATE="$2";       shift 2 ;;
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
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./sglang_cache_reports"
    BENCH_LABEL="SGLang"
    LOG_PREFIX="sglang_cache"
    ;;
  vllm|vl)
    FRAMEWORK="vllm"
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./vllm_cache_reports"
    BENCH_LABEL="vLLM"
    LOG_PREFIX="vllm_cache"
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

# --- 校验缓存命中率 ---
CACHE_HIT_RATE="${CACHE_HIT_RATE:-$DEFAULT_CACHE_HIT_RATE}"
if ! [[ "$CACHE_HIT_RATE" =~ ^[0-9]+$ ]] || [[ "$CACHE_HIT_RATE" -lt 1 ]] || [[ "$CACHE_HIT_RATE" -ge 100 ]]; then
  echo "ERROR: --cache-hit-rate 必须是 1~99 的整数，当前值: '$CACHE_HIT_RATE'" >&2
  exit 1
fi

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
    --cache-hit-rate "$CACHE_HIT_RATE" \
    --sleep "$SLEEP_TIME" \
    --chip-type "$CHIP_TYPE" \
    --tester "$TESTER" \
    --pd "$PD" \
    --timeout "$TIMEOUT" \
    --foreground \
    > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "${BENCH_LABEL} cache benchmark 已在后台启动 (PID: $PID)"
  echo "日志文件: $LOG_FILE"
  echo "监控进度: tail -f \"$LOG_FILE\""
  echo "停止进程: kill $PID"
  exit 0
fi

# --- 构建并发数列表 ---
CONCURRENCY_STR="${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}"
IFS=',' read -ra concurrency_list <<< "$CONCURRENCY_STR"

# --- 提取模型服务名最后一段（以 / \ 分割），并清除残留 : \，用于目录创建 ---
safe_model_name=$(echo "$SERVED_MODEL_NAME" | sed 's/.*[\/\\]//; s/[:\\]//g')

# --- 运行时间戳（作为本次测试的子目录名，避免覆盖） ---
RUN_TS=$(date +%Y%m%d_%H%M%S)

# --- 创建报告目录 ---
RUN_DIR="$REPORT_DIR/${TESTER}/${safe_model_name}/${RUN_TS}"
mkdir -p "$RUN_DIR"

# --- 计算前缀级缓存命中参数 ---
# 共享前缀长度 = 输入长度 × 命中率 / 100（命中部分）
# 唯一后缀长度 = 输入长度 - 共享前缀长度（未命中部分）
prefix_len=$(( INPUT_LEN * CACHE_HIT_RATE / 100 ))
suffix_len=$(( INPUT_LEN - prefix_len ))

echo "=========================================="
echo "=== Starting ${BENCH_LABEL} Cache Hit Rate Benchmark ==="
echo "=== Framework:      $FRAMEWORK"
echo "=== Base URL:       $BASE_URL"
echo "=== Model Path:     $MODEL_PATH"
echo "=== Served Name:    $SERVED_MODEL_NAME"
echo "=== Report Dir:     $RUN_DIR"
echo "=== Concurrency:    ${concurrency_list[*]}"
echo "=== Input Length:   ${INPUT_LEN}"
echo "=== Output Length:  ${OUTPUT_LEN}"
echo "=== Cache Hit Rate: ${CACHE_HIT_RATE}% (prefix-level)"
echo "=== Shared Prefix:  ${prefix_len} tokens (hit)"
echo "=== Unique Suffix:  ${suffix_len} tokens (miss)"
echo "=== Chip Type:      ${CHIP_TYPE:-(未指定)}"
echo "=== Tester:         $TESTER"
echo "=== PD Mode:        $PD"
echo "=== Method:         两阶段（预热+测试），前缀级命中"
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
#   vLLM:   vllm:prefix_cache_queries / vllm:prefix_cache_hits / vllm:external_prefix_cache_hits
#           含 L1(GPU) / L3(external/LM cache) 两层命中统计
#   SGLang: sglang:prefill_effective_tokens_total (mode=input/device_hit/host_hit/storage_hit)
#           含 HiCache L1(GPU) / L2(CPU) / L3(DFKV) 全层命中统计
# 测试前后各查询一次 /metrics，差值计算真实命中率，附加到日志
# ============================================================

# --- vLLM: 从服务端 /metrics 提取前缀缓存计数器（token 数，跨 engine 求和） ---
# L1 (GPU): vllm:prefix_cache_queries/hits_total；L3 (external/LM cache): vllm:external_prefix_cache_hits_total
# 输出 "<queries> <l1_hits> <l3_hits>"；无 LM Cache 时 l3_hits=0；/metrics 不可用输出 "N/A"
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

# --- SGLang: 从服务端 /metrics 提取 prefill_effective_tokens_total 计数器（token 数，跨 rank 求和） ---
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

# --- 外层循环：遍历不同的并发值 ---
for concurrency in "${concurrency_list[@]}"; do
  # N_total = 测试阶段总请求数 = 5 × 并发数（增加统计样本，提高测量稳定性）
  N_total=$(( concurrency * 5 ))

  echo ""
  echo "--- Configuration: Concurrency=$concurrency, N_total=$N_total ---"

  # 构建日志文件路径
  log_file="${RUN_DIR}/cache_bs-${concurrency}_hit-${CACHE_HIT_RATE}_input-${INPUT_LEN}-output-${OUTPUT_LEN}.log"
  warmup_log="${RUN_DIR}/cache_bs-${concurrency}_hit-${CACHE_HIT_RATE}_warmup.log"
  jsonl_file="${RUN_DIR}/cache_bs-${concurrency}_hit-${CACHE_HIT_RATE}_input-${INPUT_LEN}-output-${OUTPUT_LEN}.jsonl"

  # ============================================================
  # 阶段 1：预热（不计量）
  # 用 concurrency=1 发送 1 个请求，将共享前缀填入缓存
  # 使用与阶段 2 相同的 seed 和 gsp-num-groups=1，确保前缀相同
  # ============================================================
  echo ""
  echo ">>> [Phase 1] Pre-warming: sending 1 request at concurrency=1 (fills shared prefix into cache)..."
  echo ">>> Warmup log: $warmup_log"

  if [[ "$FRAMEWORK" == "sglang" ]]; then
    set +e
    timeout "$TIMEOUT" "$PYTHON_CMD" -m sglang.benchmark.serving \
      --backend sglang-oai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --dataset-name generated-shared-prefix \
      --gsp-num-groups 1 \
      --gsp-prompts-per-group 1 \
      --gsp-system-prompt-len "$prefix_len" \
      --gsp-question-len "$suffix_len" \
      --gsp-output-len 1 \
      --num-prompts 1 \
      --max-concurrency 1 \
      --seed "$SEED" \
      > "$warmup_log" 2>&1
    warmup_status=$?
    set -e
  elif [[ "$FRAMEWORK" == "vllm" ]]; then
    set +e
    timeout "$TIMEOUT" vllm bench serve \
      --backend openai-chat \
      --endpoint /v1/chat/completions \
      --base-url "$BASE_URL" \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --dataset-name prefix_repetition \
      --prefix-repetition-prefix-len "$prefix_len" \
      --prefix-repetition-suffix-len "$suffix_len" \
      --prefix-repetition-num-prefixes 1 \
      --prefix-repetition-output-len 1 \
      --num-prompts 1 \
      --max-concurrency 1 \
      --trust-remote-code \
      --temperature 0.7 \
      --seed "$SEED" \
      --metric_percentiles 95,99 \
      --ready-check-timeout-sec 30 \
      > "$warmup_log" 2>&1
    warmup_status=$?
    set -e
  fi

  if [[ $warmup_status -eq 124 ]]; then
    echo "!!! [Phase 1] Warmup TIMEOUT after ${TIMEOUT}s. Check log: $warmup_log"
    echo "!!! Skipping this concurrency level."
    FAIL_COUNT=$((FAIL_COUNT + 1))
    sleep "$SLEEP_TIME"
    continue
  elif [[ $warmup_status -ne 0 ]]; then
    echo "!!! [Phase 1] Warmup failed (exit code: $warmup_status). Check log: $warmup_log"
    echo "!!! Skipping this concurrency level."
    FAIL_COUNT=$((FAIL_COUNT + 1))
    sleep "$SLEEP_TIME"
    continue
  fi
  echo ">>> [Phase 1] Warmup done. Shared prefix (${prefix_len} tokens) cached."

  # --- 测试前抓取服务端前缀缓存计数器（与测试后差值计算真实命中率） ---
  if [[ "$FRAMEWORK" == "vllm" ]]; then
    cache_pc_before="$(wait_vllm_prefix_cache_counters)"
  elif [[ "$FRAMEWORK" == "sglang" ]]; then
    cache_pc_before="$(wait_sglang_cache_counters)"
  fi

  # ============================================================
  # 阶段 2：测试（计量）
  # 用目标并发发送 N_total 个请求
  # 所有请求共享同一前缀（命中 ${CACHE_HIT_RATE}%），各有不同后缀（未命中 ${suffix_len} tokens）
  # 每个请求需 prefill ${suffix_len} tokens（唯一后缀）
  # ============================================================
  echo ""
  echo ">>> [Phase 2] Testing: sending $N_total requests at concurrency=$concurrency..."
  echo ">>> Each request: ${prefix_len} tokens hit + ${suffix_len} tokens miss = ${CACHE_HIT_RATE}% prefix hit rate"
  echo ">>> Test log: $log_file"

  if [[ "$FRAMEWORK" == "sglang" ]]; then
    set +e
    timeout "$TIMEOUT" "$PYTHON_CMD" -m sglang.benchmark.serving \
      --backend sglang-oai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --dataset-name generated-shared-prefix \
      --gsp-num-groups 1 \
      --gsp-prompts-per-group "$N_total" \
      --gsp-system-prompt-len "$prefix_len" \
      --gsp-question-len "$suffix_len" \
      --gsp-output-len "$OUTPUT_LEN" \
      --num-prompts "$N_total" \
      --max-concurrency "$concurrency" \
      --seed "$SEED" \
      --cache-report \
      --output-file "$jsonl_file" \
      > "$log_file" 2>&1
    test_status=$?
    set -e
  elif [[ "$FRAMEWORK" == "vllm" ]]; then
    set +e
    timeout "$TIMEOUT" vllm bench serve \
      --backend openai-chat \
      --endpoint /v1/chat/completions \
      --base-url "$BASE_URL" \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --dataset-name prefix_repetition \
      --prefix-repetition-prefix-len "$prefix_len" \
      --prefix-repetition-suffix-len "$suffix_len" \
      --prefix-repetition-num-prefixes 1 \
      --prefix-repetition-output-len "$OUTPUT_LEN" \
      --num-prompts "$N_total" \
      --max-concurrency "$concurrency" \
      --trust-remote-code \
      --temperature 0.7 \
      --seed "$SEED" \
      --metric_percentiles 95,99 \
      --ready-check-timeout-sec 30 \
      > "$log_file" 2>&1
    test_status=$?
    set -e
  fi

  if [[ $test_status -eq 124 ]]; then
    echo "!!! [Phase 2] Test TIMEOUT after ${TIMEOUT}s. Check log: $log_file"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  elif [[ $test_status -ne 0 ]]; then
    echo "!!! [Phase 2] Test failed (exit code: $test_status). Check log: $log_file"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    error_pattern="$(check_log_errors "$log_file")" || true
    if [[ -n "$error_pattern" ]]; then
      echo "!!! [Phase 2] Log error detected: '$error_pattern'. Check log: $log_file"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    else
      echo ">>> [Phase 2] Test done. Results saved to: $log_file"
      # --- 计算真实前缀缓存命中率并附加到日志（供 collect_results.py 解析） ---
      # SGLang: /metrics 的 prefill_effective_tokens_total 计数器差值（含 HiCache L1/L2/L3）
      #         覆盖 --cache-report 仅统计 L1 (radix cache) 的局限性
      # vLLM:   /metrics 的 vllm:prefix_cache_* 计数器差值
      if [[ "$FRAMEWORK" == "vllm" ]]; then
        append_vllm_cache_report "$log_file" "$cache_pc_before"
      elif [[ "$FRAMEWORK" == "sglang" ]]; then
        append_sglang_cache_report "$log_file" "$cache_pc_before"
      fi
    fi
  fi

  # 每次测试后等待指定时间
  echo "--- Finished Concurrency=$concurrency, HitRate=${CACHE_HIT_RATE}%. Sleeping for $SLEEP_TIME seconds... ---"
  sleep "$SLEEP_TIME"
done

echo ""
echo "=== All cache hit rate tests finished ==="
echo "=== Check results in: $RUN_DIR ==="
if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "=== WARNING: $FAIL_COUNT test(s) failed. Partial results will still be collected. ==="
fi

# --- 构建 BENCH_COMMANDS（循环模板，变量占位，用于报告第三部分） ---
CONC_LIST="${concurrency_list[*]}"

if [[ "$FRAMEWORK" == "sglang" ]]; then
  BENCH_COMMANDS="# 并发数列表: ${CONCURRENCY_STR}
# 缓存命中率: ${CACHE_HIT_RATE}% (前缀级)
# 输入长度: ${INPUT_LEN}, 输出长度: ${OUTPUT_LEN}
# 共享前缀长度: ${prefix_len} (命中), 唯一后缀长度: ${suffix_len} (未命中)
# 方法: 两阶段（预热+测试）
#   阶段1 预热: 1 个请求填共享前缀入缓存（不计量）
#   阶段2 测试: N_total = 5×并发 个请求，每个请求前缀命中+后缀未命中
for CONCURRENCY in ${CONC_LIST}; do
  N_TOTAL=\$(( CONCURRENCY * 5 ))
  # 阶段1: 预热（串行，填缓存）
    $PYTHON_CMD -m sglang.benchmark.serving \\
    --backend sglang-oai-chat \\
    --base-url \"\$BASE_URL\" \\
    --model \"\$MODEL_PATH\" \\
    --served-model-name \"\$SERVED_MODEL_NAME\" \\
    --dataset-name generated-shared-prefix \\
    --gsp-num-groups 1 \\
    --gsp-prompts-per-group 1 \\
    --gsp-system-prompt-len ${prefix_len} \\
    --gsp-question-len ${suffix_len} \\
    --gsp-output-len 1 \\
    --num-prompts 1 \\
    --max-concurrency 1 \\
    --seed ${SEED}
  # 阶段2: 测试（目标并发，计量）
    $PYTHON_CMD -m sglang.benchmark.serving \\
    --backend sglang-oai-chat \\
    --base-url \"\$BASE_URL\" \\
    --model \"\$MODEL_PATH\" \\
    --served-model-name \"\$SERVED_MODEL_NAME\" \\
    --dataset-name generated-shared-prefix \\
    --gsp-num-groups 1 \\
    --gsp-prompts-per-group \$N_TOTAL \\
    --gsp-system-prompt-len ${prefix_len} \\
    --gsp-question-len ${suffix_len} \\
    --gsp-output-len ${OUTPUT_LEN} \\
    --num-prompts \$N_TOTAL \\
    --max-concurrency \$CONCURRENCY \\
    --seed ${SEED} \\
    --cache-report
done"
elif [[ "$FRAMEWORK" == "vllm" ]]; then
  BENCH_COMMANDS="# 并发数列表: ${CONCURRENCY_STR}
# 缓存命中率: ${CACHE_HIT_RATE}% (前缀级)
# 输入长度: ${INPUT_LEN}, 输出长度: ${OUTPUT_LEN}
# 共享前缀长度: ${prefix_len} (命中), 唯一后缀长度: ${suffix_len} (未命中)
# 方法: 两阶段（预热+测试）
#   阶段1 预热: 1 个请求填共享前缀入缓存（不计量）
#   阶段2 测试: N_total = 5×并发 个请求，每个请求前缀命中+后缀未命中
for CONCURRENCY in ${CONC_LIST}; do
  N_TOTAL=\$(( CONCURRENCY * 5 ))
  # 阶段1: 预热（串行，填缓存）
  vllm bench serve \\
    --backend openai-chat \\
    --endpoint /v1/chat/completions \\
    --base-url \"\$BASE_URL\" \\
    --model \"\$MODEL_PATH\" \\
    --served-model-name \"\$SERVED_MODEL_NAME\" \\
    --dataset-name prefix_repetition \\
    --prefix-repetition-prefix-len ${prefix_len} \\
    --prefix-repetition-suffix-len ${suffix_len} \\
    --prefix-repetition-num-prefixes 1 \\
    --prefix-repetition-output-len 1 \\
    --num-prompts 1 \\
    --max-concurrency 1 \\
    --trust-remote-code \\
    --temperature 0.7 \\
    --seed ${SEED} \\
    --metric_percentiles 95,99 \\
    --ready-check-timeout-sec 30
  # 阶段2: 测试（目标并发，计量）
  vllm bench serve \\
    --backend openai-chat \\
    --endpoint /v1/chat/completions \\
    --base-url \"\$BASE_URL\" \\
    --model \"\$MODEL_PATH\" \\
    --served-model-name \"\$SERVED_MODEL_NAME\" \\
    --dataset-name prefix_repetition \\
    --prefix-repetition-prefix-len ${prefix_len} \\
    --prefix-repetition-suffix-len ${suffix_len} \\
    --prefix-repetition-num-prefixes 1 \\
    --prefix-repetition-output-len ${OUTPUT_LEN} \\
    --num-prompts \$N_TOTAL \\
    --max-concurrency \$CONCURRENCY \\
    --trust-remote-code \\
    --temperature 0.7 \\
    --seed ${SEED} \\
    --metric_percentiles 95,99 \\
    --ready-check-timeout-sec 30
done"
fi

# --- 自动收集结果生成 CSV ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${RUN_DIR}/cache_results.csv"
"$PYTHON_CMD" "${SCRIPT_DIR}/collect_results.py" \
  --report-dir "$RUN_DIR" \
  --model-name "$SERVED_MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --chip-type "$CHIP_TYPE" \
  --out "$CSV_FILE"

# --- 自动生成 Markdown 测试报告 ---
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_${CHIP_TYPE}_${FRAMEWORK}_${PD}_cache_${RUN_TS}.md"
if [[ -n "$DESCRIBE" ]]; then
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --bench-command "$BENCH_COMMANDS" \
    --describe "$DESCRIBE"
else
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --bench-command "$BENCH_COMMANDS"
fi
