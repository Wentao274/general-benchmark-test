#!/bin/bash
set -euo pipefail

# ============================================================
# 纯 Decode 基准测试脚本（DP 均衡变体，支持 SGLang / vLLM）
#
# 原理：共享前缀 + 前缀缓存。prefix 命中缓存 → prefill≈0，全部耗时是 decode
# 本变体不使用上游 bench 工具，改用自研 DP 均衡客户端 dp_bench.py（OpenAI 兼容
# /v1/completions + token id 级 prompt），适用 DP>=2 的数据并行部署（组内 TP 任意）：
#   上游工具的数据集按组块顺序生成 + 滑窗提交 → 在飞窗口前缀集中 → DP 路由
#   （最短队列/缓存亲和）把请求聚到少数 rank，DP 严重不均衡且客户端无法干预。
#
# dp_bench.py 发送协议（客户端可控的全部要素）：
#   每请求 = 组 g 共享前缀（seed+g 确定性生成）+ 每请求唯一后缀（1 个新 token）
#   阶段1 轮转预热（不计量）：dp-size 轮 × 每组 1 条，逐轮轮转组序 →
#          round_robin 分派下每 rank 把 G 个前缀各缓存一遍
#          （前缀亲和路由部署请用 --dp-warmup-mode affinity + --no-dp-shuffle）
#   阶段2 均衡测试（计量）：refill 模式（默认）层交错 + 滑窗补位，完成一个立刻
#          补一个，持续并发请求流不中断，任意时刻每组在飞 >=1 条（rank 不空转）
#          （barrier 模式为可选验证口径：每批每组一条 + 批间 barrier；
#            并发<组数自动回退 refill）
#   /metrics 计数器在两阶段之间快照 → 实测命中率只统计测试阶段
#
# 前置条件：服务端必须开启前缀缓存
#   SGLang: 默认开启（不要加 --disable-radix-cache）
#   vLLM:   默认开启（不要加 --no-enable-prefix-caching）
# 缓存验证：测试前后查询 /metrics 计数器，实测命中率应 ≈100%
#   （若远低于 100% 说明缓存未生效或路由与预热模式不匹配，decode 结果被 prefill 开销污染）
# ============================================================

# --- 默认配置 ---
FRAMEWORK=""
BASE_URL=""
MODEL_PATH=""
SERVED_MODEL_NAME=""
SEED=123
CONCURRENCY_MULTIPLIER=10
SLEEP_TIME=60
REPORT_DIR=""
CHIP_TYPE=""
TESTER=""
PD=""
DESCRIBE=""
BACKGROUND=true
TIMEOUT=3600
CONCURRENCY_ARG=""
PREFIX_LENS_ARG=""
OUTPUT_LENS_ARG=""

# 前缀长度列表（模拟 KV cache 已就位的上下文长度）
DEFAULT_PREFIX_LENS="8192,32768,65536,131072,262144"
# decode 输出长度：未指定 -o 时按前缀组配对（≤64K → 512，128K~256K → 1024）
DEFAULT_OUTPUT_LENS=""
OUTPUT_LENS_ARG=""
# 并发数列表
DEFAULT_CONCURRENCY="1,8,32,64,128"
# 共享前缀组数（每组内请求共享同一前缀，组间前缀不同）
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
  -r, --report-dir DIR          报告输出目录        (default: ./{framework}_decode_dp_reports)
  -p, --prefix-lens LIST        前缀长度列表(逗号分隔) (default: $DEFAULT_PREFIX_LENS)
  -o, --output-lens LIST        输出长度列表(逗号分隔;缺省按前缀组配对: <=64K->512, 128K~256K->1024)
  -c, --concurrency LIST        并发档位。两种形式:
                                 全局: 逗号分隔单档, 如 "1,8,32,64,128" (default: $DEFAULT_CONCURRENCY)
                                 逐组: 组间逗号、组内空格, 与 -p 前缀组从第一组起一一对应, 如 "1 8 32 64,1 8 32,1 8"
                                   组数=1 全局 / 组数>前缀组数 舍弃多余组 / 前缀组数>组数 未匹配前缀组用默认
                                   未传 -p 时只取第一组作全局
  -g, --gsp-num-groups N        共享前缀组数(组内请求共享同一前缀,组间前缀不同) (default: 1)
                                  请求总数 = 每组条数 × N，每组条数 = ceil(并发倍数×并发/N)
  --dp-size N                   服务端 DP rank 数 (default: 8)
                                    rotate 预热轮数 = N；每批/每层每组一条 + 批间(层间)打乱组序
  --dp-send-mode MODE           发送模式: refill(层交错+滑窗补位,持续并发,稳态压测,默认)
                                    或 barrier(逐批同步:每批每组一条+批间barrier,验证均衡用)
  --dp-warmup-mode MODE         预热模式: rotate(N轮×每组1条,逐轮轮转组序,round_robin 路由用)
                                    或 affinity(仅G条固定组序,前缀亲和路由用) (default: rotate)
  --no-dp-shuffle               关闭批间组序打乱(默认开启; 前缀亲和路由必须关闭)
  --vocab-size N                模型词表大小(随机 token id 上界) (default: 151552)
  --seed N                      随机数据集种子    (default: $SEED)
  --concurrency-multiplier N    并发倍数: 请求总数 = 每组条数 × 组数，每组条数 = ceil(N×并发/组数)
                                  (default: $CONCURRENCY_MULTIPLIER)
  -s, --sleep SECONDS           每次测试间隔秒数    (default: $SLEEP_TIME)
  -d, --describe TEXT           模型描述信息(可选,填入报告末尾)
  -f, --foreground              前台执行(默认后台)
  --timeout SECONDS             每组测试超时秒数        (default: $TIMEOUT)
  -h, --help                    显示帮助

示例（默认后台执行）:
  $0 -F sglang -u http://127.0.0.1:8080 -m /data/model -n model-name -t H100 -T zhangsan -P agg
  $0 -F vllm   -u http://127.0.0.1:8000 -m /data/model -n model-name -t H100 -T zhangsan -P disagg
  $0 -F sglang -f -u http://... -p 8192,16384 -o 512 -c 1,8,64 -T zhangsan -P agg   # 前台执行
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
    -g|--gsp-num-groups)      GSP_NUM_GROUPS="$2";       shift 2 ;;
    --dp-size)                DP_SIZE="$2";             shift 2 ;;
    --dp-send-mode)           SEND_MODE="$2";           shift 2 ;;
    --dp-warmup-mode)         WARMUP_MODE="$2";         shift 2 ;;
    --no-dp-shuffle)          DP_SHUFFLE=false;         shift   ;;
    --vocab-size)              VOCAB_SIZE="$2";          shift 2 ;;
    --seed)                   SEED="$2";                shift 2 ;;
    --concurrency-multiplier) CONCURRENCY_MULTIPLIER="$2"; shift 2 ;;
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
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./sglang_decode_dp_reports"
    BENCH_LABEL="SGLang"
    LOG_PREFIX="sglang_decode_dp"
    ;;
  vllm|vl)
    FRAMEWORK="vllm"
    [[ -z "$REPORT_DIR" ]] && REPORT_DIR="./vllm_decode_dp_reports"
    BENCH_LABEL="vLLM"
    LOG_PREFIX="vllm_decode_dp"
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

# --- 后台执行（默认） ---
if [[ "$BACKGROUND" == "true" ]]; then
  mkdir -p "$REPORT_DIR/${TESTER}"
  LOG_FILE="${REPORT_DIR}/${TESTER}/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
  # 重新启动自身（带 --foreground 强制前台），输出重定向到日志文件
  # 传递给 wrapper 自身：仅显式关闭时传 --no-dp-shuffle（--shuffle 是 dp_bench.py 的旗标）
  DP_SHUFFLE_ARGS=""
  [[ "$DP_SHUFFLE" == "false" ]] && DP_SHUFFLE_ARGS="--no-dp-shuffle"
  nohup "$0" \
    --framework "$FRAMEWORK" \
    --base-url "$BASE_URL" \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --report-dir "$REPORT_DIR" \
    --prefix-lens "${PREFIX_LENS_ARG:-$DEFAULT_PREFIX_LENS}" \
    --output-lens "${OUTPUT_LENS_ARG:-$DEFAULT_OUTPUT_LENS}" \
    --concurrency "${CONCURRENCY_ARG:-$DEFAULT_CONCURRENCY}" \
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

# --- 构建并发档位与前缀组映射 ---
# -c 无空格: 全局档位列表（逗号分隔单档，向后兼容）
# -c 含空格: 并发组（组间逗号、组内空格），与 -p 前缀组从第一组起一一对应：
#   组数=1 → 全局适用；组数<前缀组数 → 未匹配前缀组用默认档位；组数>前缀组数 → 舍弃多余组
#   未显式传 -p 时 → 只取第一组作为全局
n_combos=${#prefix_lens[@]}
CONC_GLOBAL_MODE=true
CONC_MAP_DESC="全局 ${CONCURRENCY_STR}"
declare -a CONC_STRS=()
if [[ "$CONCURRENCY_STR" == *[[:space:]]* ]]; then
  IFS=',' read -ra __conc_groups <<< "$CONCURRENCY_STR"
  n_groups=${#__conc_groups[@]}
  if [[ -z "$PREFIX_LENS_ARG" ]] || [[ "$n_groups" -eq 1 ]]; then
    for ((ci=0; ci<n_combos; ci++)); do
      CONC_STRS[$ci]="${__conc_groups[0]}"
    done
    if [[ -z "$PREFIX_LENS_ARG" ]]; then
      CONC_MAP_DESC="全局 ${__conc_groups[0]}（未传 -p，取第一组）"
    else
      CONC_MAP_DESC="全局 ${__conc_groups[0]}（单组）"
    fi
  else
    CONC_GLOBAL_MODE=false
    CONC_MAP_DESC="按前缀组一一对应（组数>前缀组数舍弃多余，未匹配前缀组用默认 ${DEFAULT_CONCURRENCY}）"
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

safe_model_name=$(echo "$SERVED_MODEL_NAME" | sed 's/.*[\/\\]//; s/[:\\]//g')

# --- 运行时间戳（作为本次测试的子目录名，避免覆盖） ---
RUN_TS=$(date +%Y%m%d_%H%M%S)

# --- 创建报告目录 ---
RUN_DIR="$REPORT_DIR/${TESTER}/${safe_model_name}/${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "=== Starting ${BENCH_LABEL} Pure Decode Benchmark (DP balanced) ==="
echo "=== Framework:    $FRAMEWORK"
echo "=== Base URL:     $BASE_URL"
echo "=== Model Path:   $MODEL_PATH"
echo "=== Served Name:  $SERVED_MODEL_NAME"
echo "=== Report Dir:   $RUN_DIR"
echo "=== Prefix Lens:  ${prefix_lens[*]}"
if [[ -n "$OUTPUT_LENS_ARG" ]]; then
  echo "=== Output Lens:   ${output_lens[*]}"
else
  echo "=== Output Lens:   512 (prefix<=64K) / 1024 (prefix>64K) [paired per prefix]"
fi
if [[ "$CONC_GLOBAL_MODE" == "true" ]]; then
  echo "=== Concurrency:  ${CONC_MAP_DESC} (global)"
else
  echo "=== Concurrency:  per-prefix groups:"
  for ((ci=0; ci<n_combos; ci++)); do
    echo "===   prefix ${prefix_lens[$ci]} -> ${CONC_STRS[$ci]}"
  done
fi
echo "=== GSP Groups:   $GSP_NUM_GROUPS (per group shares one prefix)"
echo "=== DP Size:      $DP_SIZE"
echo "=== DP Send Mode: $SEND_MODE"
echo "=== DP Shuffle:   $([[ "$DP_SHUFFLE" == "true" ]] && echo "on" || echo "off")"
echo "=== DP Warmup:    $WARMUP_MODE (rotate=${DP_SIZE}轮×每组1条 / affinity=G条)"
echo "=== Seed:          $SEED"
echo "=== Conc Multiplier: $CONCURRENCY_MULTIPLIER"
echo "=== Client:       dp_bench.py (DP 均衡客户端, /v1/completions token-id prompt)"
echo "=== 前缀缓存:      ON (必须开启) ==="
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
#   验证服务端缓存命中是否生效（预期 ~100%）
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

for prefix_idx in "${!prefix_lens[@]}"; do
  prefix_len="${prefix_lens[$prefix_idx]}"
  if [[ -n "$OUTPUT_LENS_ARG" ]]; then
    # 用户显式指定 -o：前缀 × 输出全交叉
    eff_output_lens=("${output_lens[@]}")
  else
    # 缺省：输出长度按前缀组配对（≤64K → 512，128K~256K → 1024）
    if [[ "$prefix_len" -le 65536 ]]; then
      eff_output_lens=(512)
    else
      eff_output_lens=(1024)
    fi
  fi
  for output_len in "${eff_output_lens[@]}"; do
    # 该前缀组对应的并发档位（-c 逐组映射，见构建区解析）
    IFS=' ' read -ra prefix_concurrency_list <<< "${CONC_STRS[$prefix_idx]}"
    for concurrency in "${prefix_concurrency_list[@]}"; do
      # num_prompts = 并发倍数（默认 10）× 并发数，增加统计样本数，提高测量稳定性
      # GSP_NUM_GROUPS > 1 时按组均摊（向上取整），实际总数 = 每组条数 × 组数
      base_prompts=$((concurrency * CONCURRENCY_MULTIPLIER))
      per_group_prompts=$(( (base_prompts + GSP_NUM_GROUPS - 1) / GSP_NUM_GROUPS ))
      num_prompts=$((per_group_prompts * GSP_NUM_GROUPS))
      log_file="${RUN_DIR}/decode_bs-${concurrency}_prefix-${prefix_len}-output-${output_len}.log"
      warmup_log="${RUN_DIR}/decode_bs-${concurrency}_prefix-${prefix_len}-output-${output_len}_warmup.log"
      echo ""
      echo ">>> Decode: Prefix=$prefix_len, Output=$output_len, Concurrency=$concurrency, Groups=$GSP_NUM_GROUPS, NumPrompts=$num_prompts, DPSize=$DP_SIZE, SendMode=$SEND_MODE"
      echo ">>> Log file: $log_file"

      # ============================================================
      # DP 均衡两阶段（核心：dp_bench.py）
      #   阶段1 轮转预热（不计量）：dp-size 轮 × 每组 1 条 → 每 rank 缓存全部 G 个前缀
      #   阶段2 均衡测试（计量）：每批每组一条 + 批 barrier + 批间打乱组序
      # /metrics 计数器在两阶段之间快照 → 实测命中率只统计测试阶段
      # ============================================================
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
        --suffix-len 1 \
        --output-len 1 \
        --warmup-mode "$WARMUP_MODE" \
        $DP_SHUFFLE_ARGS \
        --seed "$SEED" \
        --vocab-size "$VOCAB_SIZE" \
        --timeout "$TIMEOUT" \
        > "$warmup_log" 2>&1
      warmup_rc=$?
      set -e

      if [ $warmup_rc -eq 124 ]; then
        echo "!!! [Phase 1] DP warmup TIMEOUT after ${TIMEOUT}s. Check log: $warmup_log"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "--- Finished Prefix=$prefix_len, Output=$output_len. Sleeping for $SLEEP_TIME seconds... ---"
        sleep "$SLEEP_TIME"
        continue
      elif [ $warmup_rc -ne 0 ]; then
        echo "!!! [Phase 1] DP warmup failed (exit code: $warmup_rc). Check log: $warmup_log"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "--- Finished Prefix=$prefix_len, Output=$output_len. Sleeping for $SLEEP_TIME seconds... ---"
        sleep "$SLEEP_TIME"
        continue
      fi
      echo ">>> [Phase 1] DP warmup done ($WARMUP_MODE): $GSP_NUM_GROUPS group(s), all prefixes cached on every DP rank."

      # --- 测试前抓取服务端前缀缓存计数器（验证缓存命中，预期 ~100%） ---
      if [[ "$FRAMEWORK" == "vllm" ]]; then
        cache_pc_before="$(wait_vllm_prefix_cache_counters)"
      elif [[ "$FRAMEWORK" == "sglang" ]]; then
        cache_pc_before="$(wait_sglang_cache_counters)"
      fi

      echo ">>> [Phase 2] DP-balanced testing: $num_prompts requests, groups=$GSP_NUM_GROUPS, send_mode=$SEND_MODE, shuffle=$DP_SHUFFLE"

      set +e
      timeout "$TIMEOUT" "$PYTHON_CMD" "${SCRIPT_DIR}/dp_bench.py" \
        --phase test \
        --base-url "$BASE_URL" \
        --model "$SERVED_MODEL_NAME" \
        --gsp-num-groups "$GSP_NUM_GROUPS" \
        --dp-size "$DP_SIZE" \
        --prefix-len "$prefix_len" \
        --suffix-len 1 \
        --output-len "$output_len" \
        --num-prompts "$num_prompts" \
        --concurrency "$concurrency" \
        --send-mode "$SEND_MODE" \
        --warmup-mode "$WARMUP_MODE" \
        $DP_SHUFFLE_ARGS \
        --seed "$SEED" \
        --vocab-size "$VOCAB_SIZE" \
        --timeout "$TIMEOUT" \
        > "$log_file" 2>&1
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
          # --- 实测前缀缓存命中率，验证缓存命中生效（预期 ~100%） ---
          if [[ "$FRAMEWORK" == "vllm" ]]; then
            append_vllm_cache_report "$log_file" "$cache_pc_before"
          elif [[ "$FRAMEWORK" == "sglang" ]]; then
            append_sglang_cache_report "$log_file" "$cache_pc_before"
          fi
        fi
      fi
      echo "--- Finished Prefix=$prefix_len, Output=$output_len. Sleeping for $SLEEP_TIME seconds... ---"
      sleep "$SLEEP_TIME"
    done
  done
done

echo ""
echo "=== All decode tests finished ==="
echo "=== Check results in: $RUN_DIR ==="
if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "=== WARNING: $FAIL_COUNT test(s) failed. Partial results will still be collected. ==="
fi

# --- 构建 BENCH_COMMANDS（循环模板，变量占位，用于报告第三部分） ---
PREFIX_LIST="${prefix_lens[*]}"
OUTPUT_LIST="${output_lens[*]}"

# 记录模板中的并发档位解析块（全局模式直接赋值；逐组模式用 case 匹配前缀组）
CONC_CASE_BLOCK=""
if [[ "$CONC_GLOBAL_MODE" == "true" ]]; then
  CONC_CASE_BLOCK="  CONC_LIST=\"$(echo "${CONC_STRS[0]}" | tr ',' ' ')\""
else
  CONC_CASE_BLOCK="  case \"\$PREFIX_LEN\" in"
  for ((ci=0; ci<n_combos; ci++)); do
    CONC_CASE_BLOCK="${CONC_CASE_BLOCK}
    ${prefix_lens[$ci]}) CONC_LIST=\"${CONC_STRS[$ci]}\" ;;"
  done
  CONC_CASE_BLOCK="${CONC_CASE_BLOCK}
    *) CONC_LIST=\"$(echo "$DEFAULT_CONCURRENCY" | tr ',' ' ')\" ;;
  esac"
fi
if [[ -n "$OUTPUT_LENS_ARG" ]]; then
  OUT_HDR="# 输出长度列表: ${OUTPUT_LENS_STR}"
  OUT_OPEN="  for OUTPUT_LEN in ${OUTPUT_LIST}; do"
  OUT_CLOSE="  done"
else
  OUT_HDR="# 输出长度: 512（≤64K 前缀组）/ 1024（128K~256K 前缀组），按前缀组配对"
  OUT_OPEN="  # 输出长度按前缀组配对：≤64K → 512，128K~256K → 1024
  if [[ \"\$PREFIX_LEN\" -le 65536 ]]; then OUTPUT_LEN=512; else OUTPUT_LEN=1024; fi"
  OUT_CLOSE=""
fi

BENCH_COMMANDS="# 前缀长度列表: ${PREFIX_LENS_STR}
${OUT_HDR}
# 并发数映射: ${CONC_MAP_DESC}
# 共享前缀组数: ${GSP_NUM_GROUPS}（组内请求共享同一前缀，组间不同）
# DP 均衡: dp-size=${DP_SIZE}, send-mode=${SEND_MODE}, warmup=${WARMUP_MODE}, shuffle=${DP_SHUFFLE}
# 执行核心: dp_bench.py（阶段1 轮转预热 + 阶段2 每批每组一条/barrier/批间打乱组序）
# 并发倍数: ${CONCURRENCY_MULTIPLIER}（请求总数 = 每组条数 × 组数，每组条数 = ceil(并发倍数×并发/组数)）
for PREFIX_LEN in ${PREFIX_LIST}; do
${OUT_OPEN}
${CONC_CASE_BLOCK}
    for CONCURRENCY in \$CONC_LIST; do
      PER_GROUP=\$(( (CONCURRENCY * ${CONCURRENCY_MULTIPLIER} + ${GSP_NUM_GROUPS} - 1) / ${GSP_NUM_GROUPS} ))
      NUM_PROMPTS=\$((PER_GROUP * ${GSP_NUM_GROUPS}))
      \$PYTHON_CMD ${SCRIPT_DIR}/dp_bench.py \\
        --phase warmup --base-url \"\$BASE_URL\" --model \"\$SERVED_MODEL_NAME\" \\
        --gsp-num-groups ${GSP_NUM_GROUPS} --dp-size ${DP_SIZE} \\
        --prefix-len \$PREFIX_LEN --suffix-len 1 --warmup-mode ${WARMUP_MODE} \\
        --seed ${SEED} --vocab-size ${VOCAB_SIZE}
      \$PYTHON_CMD ${SCRIPT_DIR}/dp_bench.py \\
        --phase test --base-url \"\$BASE_URL\" --model \"\$SERVED_MODEL_NAME\" \\
        --gsp-num-groups ${GSP_NUM_GROUPS} --dp-size ${DP_SIZE} \\
        --prefix-len \$PREFIX_LEN --suffix-len 1 --output-len \$OUTPUT_LEN \\
        --num-prompts \$NUM_PROMPTS --concurrency \$CONCURRENCY \\
        --send-mode ${SEND_MODE} --warmup-mode ${WARMUP_MODE} \\
        --seed ${SEED} --vocab-size ${VOCAB_SIZE}
    done
${OUT_CLOSE}
done"
# --- 自动收集结果生成 CSV ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${RUN_DIR}/decode_results.csv"
"$PYTHON_CMD" "${SCRIPT_DIR}/collect_results.py" \
  --report-dir "$RUN_DIR" \
  --model-name "$SERVED_MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --chip-type "$CHIP_TYPE" \
  --out "$CSV_FILE"

# --- 自动生成 Markdown 测试报告 ---
MD_FILE="${RUN_DIR}/${TESTER}_${safe_model_name}_${CHIP_TYPE}_${FRAMEWORK}_${PD}_decode_dp_${RUN_TS}.md"
if [[ -n "$DESCRIBE" ]]; then
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "纯 Decode 测试（DP 均衡）" \
    --bench-command "$BENCH_COMMANDS" \
    --describe "$DESCRIBE"
else
  "$PYTHON_CMD" "${SCRIPT_DIR}/csv_to_md.py" \
    --csv "$CSV_FILE" \
    --output "$MD_FILE" \
    --test-type "纯 Decode 测试（DP 均衡）" \
    --bench-command "$BENCH_COMMANDS"
fi
