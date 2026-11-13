#!/bin/bash
set -euo pipefail

# ============================================================
# 指定缓存命中率基准测试脚本（DP 均衡变体，支持 SGLang / vLLM）
#
# 原理：前缀级缓存命中。每个请求 = 共享前缀（命中部分）+ 唯一后缀（未命中部分）
#   缓存命中率 = 共享前缀长度 / 输入总长度 × 100%
# 本变体不使用上游 bench 工具，改用自研 DP 均衡客户端 dp_bench.py
# （OpenAI 兼容 /v1/completions + token id 级 prompt），适用 DP>=2 的数据并行
# 部署（组内 TP 任意）：上游工具数据集按组块顺序生成 + 滑窗提交 → 在飞窗口前缀
# 集中 → DP 路由（最短队列/缓存亲和）把请求聚到少数 rank，且客户端无法干预。
#
# 两阶段（核心 dp_bench.py，/metrics 计数器在两阶段之间快照）：
#   阶段 1 — DP 轮转预热（不计量）：
#     rotate: dp-size 轮 × 每组 1 条，逐轮轮转组序 → round_robin 分派下
#             每 rank 把 G 个前缀各缓存一遍（前缀亲和路由部署请用
#             --dp-warmup-mode affinity + --no-dp-shuffle）
#   阶段 2 — 均衡测试（计量）：
#     N_total = 并发倍数 × 并发数（按组均摊）；refill 模式（默认）层交错 + 滑窗
#     补位，持续并发请求流不中断，任意时刻每组在飞 >=1 条（rank 不空转）
#     （barrier 为可选验证口径：每批每组一条 + 批间 barrier；
#       并发<组数自动回退 refill）
#   每个请求：组前缀（命中 R%）+ 唯一后缀（未命中 1-R%）→ prefill 仅后缀部分
# ============================================================

# --- 默认配置 ---
FRAMEWORK=""
BASE_URL=""
MODEL_PATH=""
SERVED_MODEL_NAME=""
SEED=123
CONCURRENCY_MULTIPLIER=5
SLEEP_TIME=60
REPORT_DIR=""
CHIP_TYPE=""
TESTER=""
PD=""
DESCRIBE=""
BACKGROUND=true
TIMEOUT=3600
CONCURRENCY_ARG=""
IO_ARG=""

# 缓存命中率（百分比），默认 90
DEFAULT_CACHE_HIT_RATE=90
# IO 组合列表（每组 "输入 输出"）
DEFAULT_IO="65536 512,131072 1024,262144 1024"
# 并发数列表
DEFAULT_CONCURRENCY="1,8,32,64,128"
# 共享前缀组数（每组内请求共享该组前缀，组间前缀不同）
GSP_NUM_GROUPS=1
# DP 均衡参数（执行核心 dp_bench.py）
DP_SIZE=8
SEND_MODE="refill"
WARMUP_MODE="rotate"
DP_SHUFFLE=true
VOCAB_SIZE=151552

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
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_cache_dp_reports)
  -c, --concurrency LIST        并发档位。两种形式:
                                 全局: 逗号分隔单档, 如 "1,8,32,64,128" (default: $DEFAULT_CONCURRENCY)
                                 逐组: 组间逗号、组内空格, 与 -i IO组合从第一组起一一对应, 如 "1 8 32 64,1 8 32,1 8"
                                   组数=1 全局 / 组数>组合数 舍弃多余组 / 组合数>组数 未匹配组合用默认
                                   未传 -i 时只取第一组作全局
  -i, --io-combinations LIST    IO组合(逗号分隔,每组"in out") (default: $DEFAULT_IO)
  -g, --gsp-num-groups N        共享前缀组数(组内请求共享同一前缀,组间前缀不同) (default: 1)
                                  预热阶段每组发 1 条请求填前缀；测试请求总数 = 每组条数 × N，每组条数 = ceil(并发倍数×并发/N)
  --dp-size N                   服务端 DP rank 数 (default: 8)
                                    rotate 预热轮数 = N；每批/每层每组一条 + 批间(层间)打乱组序
  --dp-send-mode MODE           发送模式: refill(层交错+滑窗补位,持续并发,稳态压测,默认)
                                    或 barrier(逐批同步:每批每组一条+批间barrier,验证均衡用)
  --dp-warmup-mode MODE         预热模式: rotate(N轮×每组1条,逐轮轮转组序,round_robin 路由用)
                                    或 affinity(仅G条固定组序,前缀亲和路由用) (default: rotate)
  --no-dp-shuffle               关闭批间组序打乱(默认开启; 前缀亲和路由必须关闭)
  --vocab-size N                模型词表大小(随机 token id 上界) (default: 151552)
  --seed N                      随机数据集种子    (default: $SEED)
  --concurrency-multiplier N    并发倍数: 测试请求数 N_total = 并发数 × N（预热不受影响）
                                  (default: $CONCURRENCY_MULTIPLIER)
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
    -i|--io-combinations)     IO_ARG="$2";              shift 2 ;;
    -g|--gsp-num-groups)      GSP_NUM_GROUPS="$2";       shift 2 ;;
    --dp-size)                DP_SIZE="$2";             shift 2 ;;
    --dp-send-mode)           SEND_MODE="$2";           shift 2 ;;
    --dp-warmup-mode)         WARMUP_MODE="$2";         shift 2 ;;
    --no-dp-shuffle)          DP_SHUFFLE=false;         shift   ;;
    --vocab-size)              VOCAB_SIZE="$2";          shift 2 ;;
    --seed)                   SEED="$2";                shift 2 ;;
    --concurrency-multiplier) CONCURRENCY_MULTIPLIER="$2"; shift 2 ;;
    -R|--cache-hit-rate)      CACHE_HIT_RATE="$2";      shift 2 ;;
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
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./sglang_cache_dp_reports"
    BENCH_LABEL="SGLang"
    LOG_PREFIX="sglang_cache_dp"
    ;;
  vllm|vl)
    FRAMEWORK="vllm"
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./vllm_cache_dp_reports"
    BENCH_LABEL="vLLM"
    LOG_PREFIX="vllm_cache_dp"
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

# --- 校验共享前缀组数 ---
if ! [[ "$GSP_NUM_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --gsp-num-groups 必须为 >=1 的整数，当前值: '$GSP_NUM_GROUPS'" >&2
  exit 1
fi

# --- 校验 DP 均衡参数 ---
if ! [[ "$DP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --dp-size 必须为 >=1 的整数，当前值: '$DP_SIZE'" >&2
  exit 1
fi
case "$SEND_MODE" in
  barrier|refill) ;;
  *)
    echo "ERROR: --dp-send-mode 只支持 barrier 或 refill，当前值: '$SEND_MODE'" >&2
    exit 1
    ;;
esac
case "$WARMUP_MODE" in
  rotate|affinity) ;;
  *)
    echo "ERROR: --dp-warmup-mode 只支持 rotate 或 affinity，当前值: '$WARMUP_MODE'" >&2
    exit 1
    ;;
esac
if ! [[ "$VOCAB_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vocab-size 必须为 >=1 的整数，当前值: '$VOCAB_SIZE'" >&2
  exit 1
fi
if [[ "$WARMUP_MODE" == "affinity" && "$DP_SHUFFLE" == "true" ]]; then
  echo "ERROR: --dp-warmup-mode affinity（前缀亲和路由）必须配合 --no-dp-shuffle，否则批间打乱组序会破坏前缀→rank 亲和导致缓存 miss" >&2
  exit 1
fi


# --- 校验 seed 与并发倍数 ---
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --seed 必须为非负整数，当前值: '$SEED'" >&2
  exit 1
fi
if ! [[ "$CONCURRENCY_MULTIPLIER" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --concurrency-multiplier 必须为 >=1 的整数，当前值: '$CONCURRENCY_MULTIPLIER'" >&2
  exit 1
fi

# --- 校验并发档位（-c 支持全局逗号列表或逐组指定：组间逗号、组内空格） ---
__conc_raw="${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}"
if [[ "$__conc_raw" == *[[:space:]]* ]]; then
  IFS=',' read -ra __conc_groups <<< "$__conc_raw"
  for __grp in "${__conc_groups[@]}"; do
    if [[ -z "$__grp" ]]; then
      echo "ERROR: --concurrency 存在空的并发组（检查连续逗号），当前值: '$__conc_raw'" >&2
      exit 1
    fi
    for __lvl in $__grp; do
      if ! [[ "$__lvl" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --concurrency 并发档位 '$__lvl' 必须为 >=1 的整数，当前值: '$__conc_raw'" >&2
        exit 1
      fi
    done
  done
else
  IFS=',' read -ra __conc_levels <<< "$__conc_raw"
  for __lvl in "${__conc_levels[@]}"; do
    if [[ -z "$__lvl" ]] || ! [[ "$__lvl" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: --concurrency 并发档位必须为 >=1 的整数（逗号分隔），当前值: '$__conc_raw'" >&2
      exit 1
    fi
  done
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
  # 传递给 wrapper 自身：仅显式关闭时传 --no-dp-shuffle（--shuffle 是 dp_bench.py 的旗标）
  DP_SHUFFLE_ARGS=""
  [[ "$DP_SHUFFLE" == "false" ]] && DP_SHUFFLE_ARGS="--no-dp-shuffle"
  nohup "$0" \
    --framework "$FRAMEWORK" \
    --base-url "$BASE_URL" \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --report-dir "$REPORT_DIR" \
    --concurrency "${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}" \
    --io-combinations "${IO_ARG:-$DEFAULT_IO}" \
    --cache-hit-rate "$CACHE_HIT_RATE" \
    --gsp-num-groups "$GSP_NUM_GROUPS" \
    --dp-size "$DP_SIZE" \
    --dp-send-mode "$SEND_MODE" \
    --dp-warmup-mode "$WARMUP_MODE" \
    $DP_SHUFFLE_ARGS \
    --seed "$SEED" \
    --concurrency-multiplier "$CONCURRENCY_MULTIPLIER" \
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

# --- 构建并发档位与 IO 组合映射 ---
# -c 无空格: 全局档位列表（逗号分隔单档，向后兼容）
# -c 含空格: 并发组（组间逗号、组内空格），与 IO 组合从第一组起一一对应：
#   组数=1 → 全局适用；组数<组合数 → 未匹配组合用默认档位；组数>组合数 → 舍弃多余组
#   未显式传 -i 时 → 只取第一组作为全局
CONCURRENCY_STR="${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}"
IO_STR="${IO_ARG:-$DEFAULT_IO}"
IFS=',' read -ra io_combinations <<< "$IO_STR"

n_combos=${#io_combinations[@]}
CONC_GLOBAL_MODE=true
CONC_MAP_DESC="全局 ${CONCURRENCY_STR}"
declare -a CONC_STRS=()
if [[ "$CONCURRENCY_STR" == *[[:space:]]* ]]; then
  IFS=',' read -ra __conc_groups <<< "$CONCURRENCY_STR"
  n_groups=${#__conc_groups[@]}
  if [[ -z "$IO_ARG" ]] || [[ "$n_groups" -eq 1 ]]; then
    for ((ci=0; ci<n_combos; ci++)); do
      CONC_STRS[$ci]="${__conc_groups[0]}"
    done
    if [[ -z "$IO_ARG" ]]; then
      CONC_MAP_DESC="全局 ${__conc_groups[0]}（未传 -i，取第一组）"
    else
      CONC_MAP_DESC="全局 ${__conc_groups[0]}（单组）"
    fi
  else
    CONC_GLOBAL_MODE=false
    CONC_MAP_DESC="按 IO 组合一一对应（组数>组合数舍弃多余，未匹配组合用默认 ${DEFAULT_CONCURRENCY}）"
    for ((ci=0; ci<n_combos; ci++)); do
      if [[ "$ci" -lt "$n_groups" ]]; then
        CONC_STRS[$ci]="${__conc_groups[$ci]}"
      else
        CONC_STRS[$ci]="$(echo "$DEFAULT_CONCURRENCY" | tr ',' ' ')"
      fi
    done
  fi
else
  __conc_flat="$(echo "$CONCURRENCY_STR" | tr ',' ' ')"
  for ((ci=0; ci<n_combos; ci++)); do
    CONC_STRS[$ci]="$__conc_flat"
  done
fi

# --- 提取模型服务名最后一段（以 / \ 分割），并清除残留 : \，用于目录创建 ---
safe_model_name=$(echo "$SERVED_MODEL_NAME" | sed 's/.*[\/\\]//; s/[:\\]//g')

# --- 运行时间戳（作为本次测试的子目录名，避免覆盖） ---
RUN_TS=$(date +%Y%m%d_%H%M%S)

# --- 创建报告目录 ---
RUN_DIR="$REPORT_DIR/${TESTER}/${safe_model_name}/${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "=== Starting ${BENCH_LABEL} Cache Hit Rate Benchmark (DP balanced) ==="
echo "=== Framework:      $FRAMEWORK"
echo "=== Base URL:       $BASE_URL"
echo "=== Model Path:     $MODEL_PATH"
echo "=== Served Name:    $SERVED_MODEL_NAME"
echo "=== Report Dir:     $RUN_DIR"
if [[ "$CONC_GLOBAL_MODE" == "true" ]]; then
  echo "=== Concurrency:    ${CONC_MAP_DESC} (global)"
else
  echo "=== Concurrency:    per-IO groups:"
  for ((ci=0; ci<n_combos; ci++)); do
    echo "===   ${io_combinations[$ci]} -> ${CONC_STRS[$ci]}"
  done
fi
echo "=== IO Combinations: ${IO_STR}"
echo "=== Cache Hit Rate: ${CACHE_HIT_RATE}% (prefix-level)"
echo "=== GSP Groups:     $GSP_NUM_GROUPS (per group shares one prefix)"
echo "=== DP Size:        $DP_SIZE"
echo "=== DP Send Mode:   $SEND_MODE"
echo "=== DP Shuffle:     $([[ "$DP_SHUFFLE" == "true" ]] && echo "on" || echo "off")"
echo "=== DP Warmup:      $WARMUP_MODE (rotate=${DP_SIZE}轮×每组1条 / affinity=G条)"
echo "=== Seed:           $SEED"
echo "=== Conc Multiplier: $CONCURRENCY_MULTIPLIER"
echo "=== Chip Type:      ${CHIP_TYPE:-(未指定)}"
echo "=== Tester:         $TESTER"
echo "=== PD Mode:        $PD"
echo "=== Method:         两阶段（DP 轮转预热+均衡测试），前缀级命中"
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

# --- 外层循环：遍历 IO 组合；内层循环：遍历该组合对应的并发档位（--concurrency 支持逐组指定） ---
for combo_idx in "${!io_combinations[@]}"; do
  combo="${io_combinations[$combo_idx]}"
  input_len=$(echo "$combo" | awk '{print $1}')
  output_len=$(echo "$combo" | awk '{print $2}')
  # 前缀级命中参数（按该组合输入长度计算）
  prefix_len=$(( input_len * CACHE_HIT_RATE / 100 ))
  suffix_len=$(( input_len - prefix_len ))

  # 该 IO 组合对应的并发档位（-c 逐组映射，见构建区解析）
  IFS=' ' read -ra combo_concurrency_list <<< "${CONC_STRS[$combo_idx]}"

  for concurrency in "${combo_concurrency_list[@]}"; do
    # N_total = 测试阶段总请求数 = 并发倍数（默认 5）× 并发数（增加统计样本，提高测量稳定性）
    # GSP_NUM_GROUPS > 1 时按组均摊（向上取整），实际总数 = 每组条数 × 组数
    base_total=$(( concurrency * CONCURRENCY_MULTIPLIER ))
    per_group_prompts=$(( (base_total + GSP_NUM_GROUPS - 1) / GSP_NUM_GROUPS ))
    N_total=$(( per_group_prompts * GSP_NUM_GROUPS ))

    echo ""
    echo "--- Configuration: Concurrency=$concurrency, Input=${input_len}, Output=${output_len}, N_total=$N_total ---"

    # 构建日志文件路径
    log_file="${RUN_DIR}/cache_bs-${concurrency}_hit-${CACHE_HIT_RATE}_input-${input_len}-output-${output_len}.log"
    warmup_log="${RUN_DIR}/cache_bs-${concurrency}_hit-${CACHE_HIT_RATE}_input-${input_len}_warmup.log"
    jsonl_file="${RUN_DIR}/cache_bs-${concurrency}_hit-${CACHE_HIT_RATE}_input-${input_len}-output-${output_len}.jsonl"
    warmup_jsonl="${RUN_DIR}/cache_bs-${concurrency}_hit-${CACHE_HIT_RATE}_input-${input_len}_warmup.jsonl"

    # ============================================================
    # 阶段 1：预热（不计量）
    # rotate: DP_SIZE 轮 × 每组 1 条（逐轮轮转组序）；affinity: G 条固定组序
    # 使用与阶段 2 相同的 seed / 组数 / 前缀长度，确保前缀相同
    # ============================================================
    echo ""
    if [[ "$WARMUP_MODE" == "rotate" ]]; then
      echo ">>> [Phase 1] DP warmup: rotate ${DP_SIZE} round(s) x $GSP_NUM_GROUPS group(s) per round, fills all prefixes into every DP rank cache..."
    else
      echo ">>> [Phase 1] DP warmup: affinity mode, $GSP_NUM_GROUPS request(s) fixed group order (prefix-affinity routing)..."
    fi
    echo ">>> Warmup log: $warmup_log"

    DP_SHUFFLE_ARGS="--shuffle"
    [[ "$DP_SHUFFLE" == "false" ]] && DP_SHUFFLE_ARGS="--no-shuffle"
    set +e
    timeout "$TIMEOUT" "$PYTHON_CMD" "${SCRIPT_DIR}/dp_bench.py" \
      --phase warmup \
      --base-url "$BASE_URL" \
      --model "$SERVED_MODEL_NAME" \
      --gsp-num-groups "$GSP_NUM_GROUPS" \
      --dp-size "$DP_SIZE" \
      --prefix-len "$prefix_len" \
      --suffix-len "$suffix_len" \
      --output-len 1 \
      --warmup-mode "$WARMUP_MODE" \
      $DP_SHUFFLE_ARGS \
      --seed "$SEED" \
      --vocab-size "$VOCAB_SIZE" \
      --timeout "$TIMEOUT" \
      > "$warmup_log" 2>&1
    warmup_status=$?
    set -e

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
    echo ">>> [Phase 1] DP warmup done. $GSP_NUM_GROUPS prefix(es) (${prefix_len} tokens each) cached on every DP rank."

    # --- 测试前抓取服务端前缀缓存计数器（与测试后差值计算真实命中率） ---
    if [[ "$FRAMEWORK" == "vllm" ]]; then
      cache_pc_before="$(wait_vllm_prefix_cache_counters)"
    elif [[ "$FRAMEWORK" == "sglang" ]]; then
      cache_pc_before="$(wait_sglang_cache_counters)"
    fi

    # ============================================================
    # 阶段 2：测试（计量）
    # 用目标并发发送 N_total 个请求
    # 每组内请求共享该组前缀（命中 ${CACHE_HIT_RATE}%），各有不同后缀（未命中 ${suffix_len} tokens）
    # 每个请求需 prefill ${suffix_len} tokens（唯一后缀）
    # ============================================================
    echo ""
    echo ">>> [Phase 2] Testing: sending $N_total requests at concurrency=$concurrency..."
    echo ">>> Each request: ${prefix_len} tokens hit + ${suffix_len} tokens miss = ${CACHE_HIT_RATE}% prefix hit rate"
    echo ">>> Test log: $log_file"

    set +e
    timeout "$TIMEOUT" "$PYTHON_CMD" "${SCRIPT_DIR}/dp_bench.py" \
      --phase test \
      --base-url "$BASE_URL" \
      --model "$SERVED_MODEL_NAME" \
      --gsp-num-groups "$GSP_NUM_GROUPS" \
      --dp-size "$DP_SIZE" \
      --prefix-len "$prefix_len" \
      --suffix-len "$suffix_len" \
      --output-len "$output_len" \
      --num-prompts "$N_total" \
      --concurrency "$concurrency" \
      --send-mode "$SEND_MODE" \
      --warmup-mode "$WARMUP_MODE" \
      $DP_SHUFFLE_ARGS \
      --seed "$SEED" \
      --vocab-size "$VOCAB_SIZE" \
      --timeout "$TIMEOUT" \
      > "$log_file" 2>&1
    test_status=$?
    set -e

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
    echo "--- Finished Concurrency=$concurrency, Input=${input_len}, HitRate=${CACHE_HIT_RATE}%. Sleeping for $SLEEP_TIME seconds... ---"
    sleep "$SLEEP_TIME"
  done
done

echo ""
echo "=== All cache hit rate tests finished ==="
echo "=== Check results in: $RUN_DIR ==="
if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "=== WARNING: $FAIL_COUNT test(s) failed. Partial results will still be collected. ==="
fi

# --- 构建 BENCH_COMMANDS（循环模板，变量占位，用于报告第三部分） ---
IO_QUOTED=""
for io in "${io_combinations[@]}"; do
  IO_QUOTED="${IO_QUOTED}\"$io\" "
done
IO_QUOTED="${IO_QUOTED% }"

# 记录模板中的并发档位解析块（全局模式直接赋值；逐组模式用 case 匹配 IO 组合）
CONC_CASE_BLOCK=""
if [[ "$CONC_GLOBAL_MODE" == "true" ]]; then
  CONC_CASE_BLOCK="  CONC_LIST=\"$(echo "${CONC_STRS[0]}" | tr ',' ' ')\""
else
  CONC_CASE_BLOCK="  case \"\$IO\" in"
  for ((ci=0; ci<n_combos; ci++)); do
    CONC_CASE_BLOCK="${CONC_CASE_BLOCK}
    \"${io_combinations[$ci]}\") CONC_LIST=\"${CONC_STRS[$ci]}\" ;;"
  done
  CONC_CASE_BLOCK="${CONC_CASE_BLOCK}
    *) CONC_LIST=\"$(echo "$DEFAULT_CONCURRENCY" | tr ',' ' ')\" ;;
  esac"
fi

BENCH_COMMANDS="# 并发数映射: ${CONC_MAP_DESC}
# IO 组合: ${IO_STR}
# 缓存命中率: ${CACHE_HIT_RATE}% (前缀级)
# 共享前缀组数: ${GSP_NUM_GROUPS}（组内请求共享该组前缀，组间不同）
# DP 均衡: dp-size=${DP_SIZE}, send-mode=${SEND_MODE}, warmup=${WARMUP_MODE}, shuffle=${DP_SHUFFLE}
# 执行核心: dp_bench.py（阶段1 DP 轮转预热 + 阶段2 每批每组一条/barrier/批间打乱组序）
# 并发倍数: ${CONCURRENCY_MULTIPLIER}（测试 N_total = 并发数 × 并发倍数；rotate 预热 = dp-size 轮 × 每组 1 条）
# 共享前缀长度 = 输入长度 × ${CACHE_HIT_RATE} / 100（命中），其余为唯一后缀（未命中）
for IO in ${IO_QUOTED}; do
  INPUT_LEN=\$(echo \"\$IO\" | awk '{print \$1}')
  OUTPUT_LEN=\$(echo \"\$IO\" | awk '{print \$2}')
  PREFIX_LEN=\$(( INPUT_LEN * ${CACHE_HIT_RATE} / 100 ))
  SUFFIX_LEN=\$(( INPUT_LEN - PREFIX_LEN ))
${CONC_CASE_BLOCK}
  for CONCURRENCY in \$CONC_LIST; do
    PER_GROUP=\$(( (CONCURRENCY * ${CONCURRENCY_MULTIPLIER} + ${GSP_NUM_GROUPS} - 1) / ${GSP_NUM_GROUPS} ))
    N_TOTAL=\$((PER_GROUP * ${GSP_NUM_GROUPS}))
    \$PYTHON_CMD ${SCRIPT_DIR}/dp_bench.py \\
      --phase warmup --base-url \"\$BASE_URL\" --model \"\$SERVED_MODEL_NAME\" \\
      --gsp-num-groups ${GSP_NUM_GROUPS} --dp-size ${DP_SIZE} \\
      --prefix-len \$PREFIX_LEN --suffix-len \$SUFFIX_LEN \\
      --warmup-mode ${WARMUP_MODE} --seed ${SEED} --vocab-size ${VOCAB_SIZE}
    \$PYTHON_CMD ${SCRIPT_DIR}/dp_bench.py \\
      --phase test --base-url \"\$BASE_URL\" --model \"\$SERVED_MODEL_NAME\" \\
      --gsp-num-groups ${GSP_NUM_GROUPS} --dp-size ${DP_SIZE} \\
      --prefix-len \$PREFIX_LEN --suffix-len \$SUFFIX_LEN --output-len \$OUTPUT_LEN \\
      --num-prompts \$N_TOTAL --concurrency \$CONCURRENCY \\
      --send-mode ${SEND_MODE} --warmup-mode ${WARMUP_MODE} \\
      --seed ${SEED} --vocab-size ${VOCAB_SIZE}
  done
done"
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
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_${CHIP_TYPE}_${FRAMEWORK}_${PD}_cache_dp_${RUN_TS}.md"
if [[ -n "$DESCRIBE" ]]; then
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "指定缓存命中率测试（DP 均衡）" \
    --bench-command "$BENCH_COMMANDS" \
    --describe "$DESCRIBE"
else
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "指定缓存命中率测试（DP 均衡）" \
    --bench-command "$BENCH_COMMANDS"
fi
