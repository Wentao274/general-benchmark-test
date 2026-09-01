#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""csv_to_md.py -- 将 collect_results.py 生成的 CSV 转换为 Markdown 测试报告

报告分三部分：
  1. 测试结果表格（从 CSV 转换）
  2. 模型服务启动命令（从 serve_command.txt 读取）
  3. Benchmark 测试命令（从 --bench-command 参数传入）

使用前需复制 serve_command_template.txt 为 serve_command.txt 并填写真实部署命令。

用法:
  python3 csv_to_md.py \
      --csv results.csv \
      --output report.md \
      --bench-command "bench.sh -F sglang -u http://... -m /path -n name -t H100" \
      --serve-command-file serve_command.txt

  如果不指定 --output，则输出到 CSV 同目录下的 {csv_basename}_report.md
  如果不指定 --serve-command-file，则默认使用脚本同目录下的 ../serve_command.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


def csv_to_markdown_table(csv_path: str) -> str:
    """读取 CSV 文件，转换为 Markdown 表格字符串。

    对前 4 列（模型名称、推理框架、输入长度、输出长度）做留空合并：
    当某列值与上一行相同时，该单元格留空，实现视觉合并效果。
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return "| (无数据) |\n|---|"

    header = rows[0]
    data_rows = rows[1:]

    # 前 6 列做留空合并（模型名称、推理框架、输入长度、输出长度、并发数、prefix长度）
    MERGE_COLS = min(6, len(header))
    prev = [""] * MERGE_COLS

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in data_rows:
        padded = list(row) + [""] * (len(header) - len(row))
        display = list(padded)
        for c in range(MERGE_COLS):
            if padded[c] == prev[c]:
                display[c] = ""
            else:
                prev[c] = padded[c]
        lines.append("| " + " | ".join(display) + " |")

    return "\n".join(lines)


def read_serve_command(serve_cmd_path: str) -> str:
    """读取 serve_command.txt 文件内容。"""
    if not os.path.isfile(serve_cmd_path):
        print("[ERROR] 找不到模型服务启动命令文件: %s" % serve_cmd_path, file=sys.stderr)
        print("        请复制 serve_command_template.txt 为 serve_command.txt，"
              "并填写真实的模型服务启动命令。", file=sys.stderr)
        sys.exit(1)

    with open(serve_cmd_path, encoding="utf-8-sig") as f:
        content = f.read().strip()

    if not content:
        print("[ERROR] serve_command.txt 文件为空，请填写模型服务启动命令: %s"
              % serve_cmd_path, file=sys.stderr)
        sys.exit(1)

    # 如果内容不以 ```bash 开头，则包裹为 bash 代码块
    if not content.startswith("```"):
        content = "```bash\n" + content + "\n```"

    return content


def generate_report(csv_path: str, serve_cmd_path: str,
                    bench_command: str, output_path: str) -> None:
    """生成完整的 Markdown 报告。"""

    # --- 第一部分：测试结果表格 ---
    md_table = csv_to_markdown_table(csv_path)

    # --- 第二部分：模型服务启动命令 ---
    serve_content = read_serve_command(serve_cmd_path)

    # --- 第三部分：Benchmark 测试命令 ---
    bench_block = "```bash\n" + bench_command + "\n```"

    # --- 组装报告 ---
    report = []
    report.append("# 推理性能 Benchmark 测试报告\n")
    report.append("> 本报告由 `csv_to_md.py` 自动生成，请勿手动编辑。\n")
    report.append("---\n")

    # 第一部分
    report.append("## 一、测试结果\n")
    report.append("<!-- CSV_TABLE_START -->")
    report.append(md_table)
    report.append("<!-- CSV_TABLE_END -->\n")
    report.append("---\n")

    # 第二部分
    report.append("## 二、模型服务启动命令\n")
    report.append("> 以下内容自动从 `serve_command.txt` 读取。\n")
    report.append("<!-- SERVE_COMMAND_START -->")
    report.append(serve_content)
    report.append("<!-- SERVE_COMMAND_END -->\n")
    report.append("---\n")

    # 第三部分
    report.append("## 三、Benchmark 测试命令\n")
    report.append("> 以下为实际执行的测试脚本命令，由程序自动记录。\n")
    report.append("<!-- BENCH_COMMAND_START -->")
    report.append(bench_block)
    report.append("<!-- BENCH_COMMAND_END -->\n")

    # --- 写入文件 ---
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("Markdown 报告已生成: %s" % output_path, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="将 CSV 测试结果转换为 Markdown 测试报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="collect_results.py 生成的 CSV 文件路径")
    ap.add_argument("--output", default="",
                    help="输出 Markdown 文件路径（默认: CSV 同目录下 {basename}_report.md）")
    ap.add_argument("--bench-command", required=True,
                    help="实际执行的 benchmark 测试命令")
    ap.add_argument("--serve-command-file", default="",
                    help="模型服务启动命令文件路径（默认: 脚本上级目录的 serve_command.txt）")

    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print("[ERROR] CSV 文件不存在: %s" % args.csv, file=sys.stderr)
        sys.exit(1)

    # 确定 serve_command.txt 路径
    serve_cmd_path = args.serve_command_file
    if not serve_cmd_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 默认在 _scripts/ 的上级目录（项目根目录）查找
        serve_cmd_path = os.path.join(os.path.dirname(script_dir), "serve_command.txt")

    # 确定输出路径
    output_path = args.output
    if not output_path:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        csv_basename = os.path.splitext(os.path.basename(args.csv))[0]
        output_path = os.path.join(csv_dir, csv_basename + "_report.md")

    generate_report(args.csv, serve_cmd_path, args.bench_command, output_path)


if __name__ == "__main__":
    main()
