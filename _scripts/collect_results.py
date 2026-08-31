#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_results.py — 扫描 bench 日志目录，提取指标汇总为 CSV

支持三种测试类型的日志文件名：
  benchmark:  input_len-{IL}-output_len-{OL}-bs-{N}.log
  prefill:    prefill_input-{IL}-bs-{N}.log              (output_len=1)
  decode:     decode_prefix-{PL}-output-{OL}-bs-{N}.log  (input_len=1)

支持两种框架的输出格式（SGLang / vLLM），自动匹配英文和中文标签。

用法:
  python3 collect_results.py --report-dir ./sglang_reports/model_name \
      --model-name dsv4-flash --framework sglang --chip-type H100 \
      --out results.csv

  # 自动检测目录下所有子目录
  python3 collect_results.py --report-dir ./sglang_reports \
      --model-name dsv4-flash --framework sglang --chip-type H100 \
      --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# ---------------------------------------------------------------- CSV 列定义

def csv_columns(chip_type):
    """根据芯片类型生成 CSV 表头列名。"""
    ct = chip_type or "芯片类型"
    return [
        "模型名称",
        "推理框架",
        "输入长度",
        "输出长度",
        "并发数",
        f"输入token吞吐量_{ct} (toks/s)",
        f"输出token吞吐量_{ct} (toks/s)",
        f"总token吞吐量_{ct} (toks/s)",
        f"Mean TTFT_{ct} (ms)",
        f"P99 TTFT_{ct} (ms)",
        f"Mean TPOT_{ct} (ms)",
    ]


# ---------------------------------------------------------------- 文件名解析

def parse_filename(fname):
    """从日志文件名提取 (test_type, input_len, output_len, concurrency)。

    返回 None 表示无法识别的文件名。
    """
    # benchmark: input_len-{IL}-output_len-{OL}-bs-{N}.log
    m = re.match(r'input_len-(\d+)-output_len-(\d+)-bs-(\d+)\.log$', fname)
    if m:
        return "benchmark", int(m.group(1)), int(m.group(2)), int(m.group(3))

    # prefill: prefill_input-{IL}-bs-{N}.log
    m = re.match(r'prefill_input-(\d+)-bs-(\d+)\.log$', fname)
    if m:
        return "prefill", int(m.group(1)), 1, int(m.group(2))

    # decode: decode_prefix-{PL}-output-{OL}-bs-{N}.log
    m = re.match(r'decode_prefix-(\d+)-output-(\d+)-bs-(\d+)\.log$', fname)
    if m:
        return "decode", int(m.group(1)), int(m.group(2)), int(m.group(3))

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
    "mean_ttft": re.compile(
        r'(?:Mean\s*TTFT|平均\s*TTFT)\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "p99_ttft": re.compile(
        r'P99\s*TTFT\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "mean_tpot": re.compile(
        r'(?:Mean\s*TPOT|平均\s*TPOT)\s*[\（(]ms[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
    # vLLM fallback: 总吞吐量 - 输出吞吐量
    "total_input_tokens": re.compile(
        r'(?:Total input tokens|总输入\s*tokens?)\s*:\s*([\d.]+)',
        re.IGNORECASE),
    "benchmark_duration": re.compile(
        r'(?:Benchmark duration|测试持续时间)\s*[\（(]s[\）)]\s*:\s*([\d.]+)',
        re.IGNORECASE),
}


def parse_log_content(filepath):
    """解析日志文件内容，返回指标字典。"""
    metrics = {
        "input_throughput": "",
        "output_throughput": "",
        "total_throughput": "",
        "mean_ttft": "",
        "p99_ttft": "",
        "mean_tpot": "",
    }
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print("[WARN] 无法读取 %s: %s" % (filepath, e), file=sys.stderr)
        return metrics

    for key, pat in PATTERNS.items():
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
            test_type, input_len, output_len, concurrency = parsed
            filepath = os.path.join(root, fname)
            metrics = parse_log_content(filepath)
            rows.append({
                "模型名称": model_name,
                "推理框架": framework,
                "输入长度": input_len,
                "输出长度": output_len,
                "并发数": concurrency,
                "输入token吞吐量": metrics["input_throughput"],
                "输出token吞吐量": metrics["output_throughput"],
                "总token吞吐量": metrics["total_throughput"],
                "Mean TTFT": metrics["mean_ttft"],
                "P99 TTFT": metrics["p99_ttft"],
                "Mean TPOT": metrics["mean_tpot"],
            })
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

    # 排序：按输入长度 → 输出长度 → 并发数
    rows.sort(key=lambda r: (r["输入长度"], r["输出长度"], r["并发数"]))

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    cols = csv_columns(args.chip_type)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["模型名称"],
                r["推理框架"],
                r["输入长度"],
                r["输出长度"],
                r["并发数"],
                r["输入token吞吐量"],
                r["输出token吞吐量"],
                r["总token吞吐量"],
                r["Mean TTFT"],
                r["P99 TTFT"],
                r["Mean TPOT"],
            ])

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
