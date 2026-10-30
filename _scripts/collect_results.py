#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_results.py — 扫描 bench 日志目录，提取指标汇总为 CSV

支持四种测试类型的日志文件名：
  benchmark:  bench_bs-{N}_input_len-{IL}-output_len-{OL}.log        (缓存命中率=日志实测值, 无则 N/A)
  prefill:    prefill_bs-{N}_input-{IL}.log                          (缓存命中率=日志实测值, 无则 N/A)
  decode:     decode_bs-{N}_prefix-{PL}-output-{OL}.log           (缓存命中率=日志实测值, 无则 N/A)
  cache:      cache_bs-{N}_hit-{HR}_input-{IL}-output-{OL}.log     (缓存命中率=实测优先, 无则回退 HR)

四种测试日志中的实测命中率来源（bench.sh / cache_bench.sh / prefill_bench.sh / decode_bench.sh 自动生成）：
  SGLang: --cache-report 输出 L1 (radix cache) 命中率 → "Cache Hit Details" 统计段；
          脚本追加 /metrics 的 prefill_effective_tokens_total 计数器差值
          （含 HiCache L1/L2/L3）→ "Prefix Cache Hit Details (SGLang /metrics)" 统计段
  vLLM:   脚本在测试前后查询服务端 /metrics 的 vllm:prefix_cache_* 计数器（L1 GPU）
          及 vllm:external_prefix_cache_hits（L3 external/LM Cache），
          差值计算后附加 "Prefix Cache Hit Details (vLLM /metrics)" 统计段到日志
  两者均输出 "Cache hit rate: X.X%" 行，取日志中最后一个匹配（/metrics 值覆盖 --cache-report）。
  prefill/decode 无 --cache-report，仅有 /metrics 统计段（如服务端支持）。

CSV 列根据框架条件化生成：
  SGLang: L1命中率 / L2命中率 / L3命中率（HiCache 分层，L1=device/GPU, L2=host/CPU, L3=storage/DFKV）
  vLLM:   L1命中率 / L3命中率（L1-GPU prefix cache, L3-external/LM Cache；无 L2）

前缀长度列（所有测试类型统一包含）：
  decode: 文件名 prefix-{PL}（共享前缀，每请求另有 1 个新 token）
  cache:  输入长度 × 命中率 / 100（与 cache_bench.sh 公式一致）
  benchmark/prefill: 无共享前缀概念，显示 "—"
  decode 的输入长度列 = 前缀 + 1（每请求实际输入 token 数）。
  行按 输入长度 → 输出长度 → 并发数 数值升序排列。

  用法:
  python3 collect_results.py --report-dir ./sglang_reports/model_name/YYYYMMDD_HHMMSS \
      --model-name glm-5.2-fp8 --framework sglang --chip-type H100 \
      --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# ---------------------------------------------------------------- CSV 列定义

def csv_columns(chip_type, framework=""):
    """根据芯片类型和框架生成 CSV 表头列名。

    SGLang: L1命中率/L2命中率/L3命中率（HiCache 分层，含 L1-GPU/L2-CPU/L3-DFKV）
    vLLM:   L1命中率/L3命中率（L1-GPU prefix cache, L3-external/LM Cache；无 L2）
    """
    ct = chip_type or "芯片类型"
    base = [
        "模型名称",
        "推理框架",
        "输入长度",
        "输出长度",
        "前缀长度",
        "并发数",
    ]
    if framework == "sglang":
        cache_cols = ["L1命中率", "L2命中率", "L3命中率"]
    else:
        cache_cols = ["L1命中率", "L3命中率"]
    perf_cols = [
        f"输入token吞吐量_{ct} (toks/s)",
        f"输出token吞吐量_{ct} (toks/s)",
        f"总token吞吐量_{ct} (toks/s)",
        f"Mean TTFT_{ct} (ms)",
        f"Median TTFT_{ct} (ms)",
        f"P90 TTFT_{ct} (ms)",
        f"P95 TTFT_{ct} (ms)",
        f"P99 TTFT_{ct} (ms)",
        f"Mean TPOT_{ct} (ms)",
        f"Median TPOT_{ct} (ms)",
        f"P90 TPOT_{ct} (ms)",
        f"P95 TPOT_{ct} (ms)",
        f"P99 TPOT_{ct} (ms)",
        f"请求吞吐量_{ct} (req/s)",
    ]
    return base + cache_cols + perf_cols


# ---------------------------------------------------------------- 文件名解析

def parse_filename(fname):
    """从日志文件名提取 (test_type, input_len, output_len, prefix_len, concurrency, cache_hit_rate_pct)。

    返回 None 表示无法识别的文件名。
    prefix_len：decode 取文件名 prefix-{PL}；cache = 输入长度 × 命中率 / 100（与脚本公式一致）；
    benchmark/prefill 无共享前缀概念，返回 None（表格中显示 "—"）。
    decode 的输入长度 = 前缀 + 1（每请求 1 个新 token）。
    cache_hit_rate_pct 为 Prefix Cache 命中率百分比整数（0~100）；
    benchmark 类型返回 None，表示命中率从日志内容实测（见 parse_log_content）。
    """
    # benchmark: bench_bs-{N}_input_len-{IL}-output_len-{OL}.log
    m = re.match(r'bench_bs-(\d+)_input_len-(\d+)-output_len-(\d+)\.log$', fname)
    if m:
        return "benchmark", int(m.group(2)), int(m.group(3)), None, int(m.group(1)), None

    # prefill: prefill_bs-{N}_input-{IL}.log
    m = re.match(r'prefill_bs-(\d+)_input-(\d+)\.log$', fname)
    if m:
        return "prefill", int(m.group(2)), 1, None, int(m.group(1)), 0

    # decode: decode_bs-{N}_prefix-{PL}-output-{OL}.log（输入 = 共享前缀 + 1 个新 token）
    m = re.match(r'decode_bs-(\d+)_prefix-(\d+)-output-(\d+)\.log$', fname)
    if m:
        prefix_len = int(m.group(2))
        return "decode", prefix_len + 1, int(m.group(3)), prefix_len, int(m.group(1)), 100

    # cache: cache_bs-{N}_hit-{HR}_input-{IL}-output-{OL}.log（前缀 = IL × HR / 100）
    m = re.match(r'cache_bs-(\d+)_hit-(\d+)_input-(\d+)-output-(\d+)\.log$', fname)
    if m:
        input_len, hit_rate = int(m.group(3)), int(m.group(2))
        return "cache", input_len, int(m.group(4)), input_len * hit_rate // 100, int(m.group(1)), hit_rate

    return None


# ---------------------------------------------------------------- 日志内容解析

# 正则：同时匹配英文和中文标签
# SGLang 英文: "Input token throughput (tok/s):   152.64"
# vLLM   英文: "Output token throughput (tok/s):  95.49"
# vLLM   中文: "输出 token 吞吐量 (tok/s):       95.49"
PATTERNS = {
    "input_throughput": re.compile(
        r'(?:Input token throughput|输入\s*token\s*吞吐量)\s*[\（(]tok/s[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "output_throughput": re.compile(
        r'(?:Output token throughput|输出\s*token\s*吞吐量)\s*[\（(]tok/s[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "total_throughput": re.compile(
        r'(?:Total token throughput|总\s*token\s*吞吐量)\s*[\（(]tok/s[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "request_throughput": re.compile(
        r'(?:Request throughput|请求\s*吞吐量)\s*[\（(]req/s[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "mean_ttft": re.compile(
        r'(?:Mean\s*TTFT|平均\s*TTFT)\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "median_ttft": re.compile(
        r'(?:Median\s*TTFT|中位数\s*TTFT)\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "p90_ttft": re.compile(
        r'P90\s*TTFT\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "p95_ttft": re.compile(
        r'P95\s*TTFT\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "p99_ttft": re.compile(
        r'P99\s*TTFT\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "mean_tpot": re.compile(
        r'(?:Mean\s*TPOT|平均\s*TPOT)\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "median_tpot": re.compile(
        r'(?:Median\s*TPOT|中位数\s*TPOT)\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "p90_tpot": re.compile(
        r'P90\s*TPOT\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "p95_tpot": re.compile(
        r'P95\s*TPOT\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "p99_tpot": re.compile(
        r'P99\s*TPOT\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    # vLLM fallback: 总吞吐量 - 输出吞吐量
    "total_input_tokens": re.compile(
        r'(?:Total input tokens|总输入\s*tokens?)\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "benchmark_duration": re.compile(
        r'(?:Benchmark duration|测试持续时间)\s*[\（(]s[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    # 前缀缓存命中率实测值（取最后一个匹配：/metrics 差值覆盖 --cache-report 的 L1 值）
    "cache_hit_rate": re.compile(
        r'Cache hit rate:\s*([\d.]+)\s*%',
        re.IGNORECASE),
    # HiCache 分层命中率（L1=device/GPU, L2=host/CPU, L3=storage 或 external）
    # SGLang L3=(storage/DFKV)，vLLM L3=(external/LM Cache)
    "l1_hit_rate": re.compile(
        r'L1 cache hit rate \(device\):\s*([\d.]+)\s*%',
        re.IGNORECASE),
    "l2_hit_rate": re.compile(
        r'L2 cache hit rate \(host\):\s*([\d.]+)\s*%',
        re.IGNORECASE),
    "l3_hit_rate": re.compile(
        r'L3 cache hit rate \([^)]+\):\s*([\d.]+)\s*%',
        re.IGNORECASE),
}


def parse_log_content(filepath):
    """解析日志文件内容，返回指标字典。"""
    metrics = {
        "input_throughput": "",
        "output_throughput": "",
        "total_throughput": "",
        "request_throughput": "",
        "mean_ttft": "",
        "median_ttft": "",
        "p90_ttft": "",
        "p95_ttft": "",
        "p99_ttft": "",
        "mean_tpot": "",
        "median_tpot": "",
        "p90_tpot": "",
        "p95_tpot": "",
        "p99_tpot": "",
        "cache_hit_rate": None,
        "l1_hit_rate": None,
        "l2_hit_rate": None,
        "l3_hit_rate": None,
    }
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print("[WARN] 无法读取 %s: %s" % (filepath, e), file=sys.stderr)
        return metrics

    for key, pat in PATTERNS.items():
        if key in ("cache_hit_rate", "l1_hit_rate", "l2_hit_rate", "l3_hit_rate"):
            # 取最后一个匹配：/metrics 差值统计附加在日志末尾，覆盖 --cache-report 的 L1 值
            matches = pat.findall(text)
            if matches:
                metrics[key] = float(matches[-1])
            continue
        m = pat.search(text)
        if m:
            val = float(m.group(1))
            if key == "total_input_tokens":
                metrics["_total_input_tokens"] = val
            elif key == "benchmark_duration":
                metrics["_benchmark_duration"] = val
            elif key in metrics:
                metrics[key] = val

    # vLLM: 没有 "Input token throughput" 时，用 总吞吐量 - 输出吞吐量 计算
    if not metrics["input_throughput"]:
        otp = metrics["output_throughput"]
        ttp = metrics["total_throughput"]
        if isinstance(otp, (int, float)) and isinstance(ttp, (int, float)):
            metrics["input_throughput"] = round(ttp - otp, 2)

    # 没有 "Total token throughput" 时，用 input + output 计算
    if not metrics["total_throughput"]:
        itp = metrics["input_throughput"]
        otp = metrics["output_throughput"]
        if isinstance(itp, (int, float)) and isinstance(otp, (int, float)):
            metrics["total_throughput"] = round(itp + otp, 2)

    return metrics


# ---------------------------------------------------------------- 主逻辑

def scan_dir(report_dir, model_name, framework):
    """递归扫描目录，返回 (rows, skipped) 。"""
    rows = []
    skipped = 0
    for root, dirs, files in os.walk(report_dir):
        for fname in sorted(files):
            if not fname.endswith(".log"):
                continue
            parsed = parse_filename(fname)
            if parsed is None:
                # 跳过非测试日志（如 sglang_bench_时间戳.log）
                skipped += 1
                continue
            test_type, input_len, output_len, prefix_len, concurrency, cache_hit_rate = parsed
            filepath = os.path.join(root, fname)
            metrics = parse_log_content(filepath)
            measured = metrics["cache_hit_rate"]
            if cache_hit_rate is None:
                # benchmark 类型：取日志中实测的前缀缓存命中率，旧日志无数据时记 N/A
                hit_rate = f"{measured:.1f}%" if measured is not None else "N/A"
            elif test_type == "cache":
                # cache 类型：优先取日志实测命中率，无数据时回退到文件名中的
                # 指定值（-R 为用户指定的测试输入参数，回退显示设计值）
                if measured is not None:
                    hit_rate = f"{measured:.1f}% (spec={cache_hit_rate}%)"
                else:
                    hit_rate = f"{cache_hit_rate}%"
            else:
                # prefill/decode 类型：取日志实测命中率（验证缓存状态），
                # 无数据时记 N/A——期望值 0%/100% 是待验证目标而非测量结果，
                # 不作为回退值（避免伪造验证通过）
                hit_rate = f"{measured:.1f}% (spec={cache_hit_rate}%)" if measured is not None else "N/A"
            # 前缀长度：decode/cache 有效；benchmark/prefill 无共享前缀概念，显示 "—"
            prefix_display = prefix_len if prefix_len is not None else "—"
            row = {
                "模型名称": model_name,
                "推理框架": framework,
                "输入长度": input_len,
                "输出长度": output_len,
                "前缀长度": prefix_display,
                "并发数": concurrency,
                "输入token吞吐量": metrics["input_throughput"],
                "输出token吞吐量": metrics["output_throughput"],
                "总token吞吐量": metrics["total_throughput"],
                "Mean TTFT": metrics["mean_ttft"],
                "Median TTFT": metrics["median_ttft"],
                "P90 TTFT": metrics["p90_ttft"],
                "P95 TTFT": metrics["p95_ttft"],
                "P99 TTFT": metrics["p99_ttft"],
                "Mean TPOT": metrics["mean_tpot"],
                "Median TPOT": metrics["median_tpot"],
                "P90 TPOT": metrics["p90_tpot"],
                "P95 TPOT": metrics["p95_tpot"],
                "P99 TPOT": metrics["p99_tpot"],
                "请求吞吐量": metrics["request_throughput"],
            }
            if framework == "sglang":
                row["L1命中率"] = f"{metrics['l1_hit_rate']:.1f}%" if metrics["l1_hit_rate"] is not None else "N/A"
                row["L2命中率"] = f"{metrics['l2_hit_rate']:.1f}%" if metrics["l2_hit_rate"] is not None else "N/A"
                row["L3命中率"] = f"{metrics['l3_hit_rate']:.1f}%" if metrics["l3_hit_rate"] is not None else "N/A"
            else:
                row["L1命中率"] = f"{metrics['l1_hit_rate']:.1f}%" if metrics["l1_hit_rate"] is not None else hit_rate
                row["L3命中率"] = f"{metrics['l3_hit_rate']:.1f}%" if metrics["l3_hit_rate"] is not None else "N/A"
            rows.append(row)
    return rows, skipped


def main():
    ap = argparse.ArgumentParser(
        description="扫描 bench 日志目录，提取指标汇总为 CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--report-dir", required=True,
                    help="bench 日志根目录（会递归扫描子目录）")
    ap.add_argument("--model-name", required=True,
                    help="模型名称（填入 CSV 的模型名称列）")
    ap.add_argument("--chip-type", default="",
                    help="芯片类型，用于 CSV 列名后缀（如 H100 / B200）")
    ap.add_argument("--framework", default="",
                    help="推理框架（填入 CSV 的推理框架列，如 sglang / vllm）")
    ap.add_argument("--out", required=True,
                    help="输出 CSV 文件路径")
    args = ap.parse_args()

    # 提取模型名最后一段（以 / \ 分割），并清除残留 : \，与 bench 脚本逻辑一致
    safe_model_name = re.sub(r'.*[/\\]', '', args.model_name)
    safe_model_name = re.sub(r'[:\\]', '', safe_model_name)

    if not os.path.isdir(args.report_dir):
        print("[ERROR] 报告目录不存在: %s" % args.report_dir, file=sys.stderr)
        sys.exit(1)

    rows, skipped = scan_dir(args.report_dir, safe_model_name, args.framework)
    if not rows:
        print("[WARN] 未在 %s 中找到任何测试日志" % args.report_dir, file=sys.stderr)
        sys.exit(0)

    # 排序：按输入长度 → 输出长度 → 并发数（数值升序；decode 按前缀递增分组）
    rows.sort(key=lambda r: (r["输入长度"], r["输出长度"], r["并发数"]))

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    cols = csv_columns(args.chip_type, args.framework)
    if args.framework == "sglang":
        cache_cols = ["L1命中率", "L2命中率", "L3命中率"]
    else:
        cache_cols = ["L1命中率", "L3命中率"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            row_vals = [
                r["模型名称"],
                r["推理框架"],
                r["输入长度"],
                r["输出长度"],
                r["前缀长度"],
                r["并发数"],
            ]
            for ck in cache_cols:
                row_vals.append(r.get(ck, ""))
            row_vals.extend([
                r["输入token吞吐量"],
                r["输出token吞吐量"],
                r["总token吞吐量"],
                r["Mean TTFT"],
                r["Median TTFT"],
                r["P90 TTFT"],
                r["P95 TTFT"],
                r["P99 TTFT"],
                r["Mean TPOT"],
                r["Median TPOT"],
                r["P90 TPOT"],
                r["P95 TPOT"],
                r["P99 TPOT"],
                r["请求吞吐量"],
            ])
            w.writerow(row_vals)

    print("已收集 %d 条测试结果 → %s" % (len(rows), args.out), file=sys.stderr)
    if skipped:
        print("跳过 %d 个非测试日志文件（如总日志）" % skipped, file=sys.stderr)
    print("\nCSV 预览:", file=sys.stderr)
    with open(args.out, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i > 5:
                print("  ...", file=sys.stderr)
                break
            print("  " + line.rstrip(), file=sys.stderr)


if __name__ == "__main__":
    main()
