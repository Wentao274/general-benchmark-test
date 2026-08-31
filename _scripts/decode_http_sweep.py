#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""decode_http_sweep.py — 纯 decode 吞吐扫描（任意 OpenAI 兼容端点，零第三方依赖）

为什么"共享前缀 + 前缀缓存"就是纯 decode:
    PD 分离架构中，decode worker 从 prefill worker 接收已填充的 KV cache，然后只做 decode。
    用一个长共享前缀（prefix_len 个 token）预热前缀缓存，等价于"KV cache 已就位"。
    之后每个请求只携带 prefix + 1 个新 token，前缀命中缓存 → prefill 近似为零，
    全部耗时是 decode 迭代。这就是 decode worker 的真实工作负载。

    对称地，纯 prefill 用 output_len=1（首 token 是 prefill 副产物，0 次 decode）；
    纯 decode 用 prefix_len>>0 + input_len=1 + output_len>>0（prefill 命中缓存，全 decode）。

为什么用 token id 而不是文本 prompt:
    与 prefill_http_sweep.py 一致——长度精确、跨框架可比、绕开 chat template。
    共享前缀用固定随机种子生成，所有请求复用同一段 token id 序列，保证缓存命中。

为什么用 streaming:
    decode 的核心指标是 TPOT（每 token 生成时间）。非流式只能拿到总时间，
    无法分离 TTFT（首 token 延迟≈缓存命中+首步 decode）与后续 decode 步。
    流式 SSE 可以精确记录首 chunk 和末 chunk 时间，从而：
      TTFT = 首 chunk 时间 - 请求发出时间
      TPOT = (末 chunk 时间 - 首 chunk 时间) / (output_len - 1)
    服务端不返回 usage 时，用 max_tokens 作为 completion_tokens 的回退值。

两种模式:
    --mode batch    一次并发打 B 发，计整批墙钟 -> 定形状 decode 吞吐
    --mode steady   维持 concurrency 在飞行中持续 duration 秒 -> 稳态产能

前置条件:
    服务端必须开启前缀缓存：
      SGLang: 不要加 --disable-radix-cache（默认开启）
      vLLM:   不要加 --no-enable-prefix-caching（默认开启）

用法:
    python3 decode_http_sweep.py \\
        --base-url http://127.0.0.1:30000 --model my-model \\
        --prefix-lens 4096,32768,65536 \\
        --output-lens 1024 \\
        --batches 1,4,8,16,32,64,128 \\
        --vocab-size 151552 --framework sglang --tp 8 \\
        --tag h100-tp8 --out results/decode_http.csv
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

CSV_FIELDS = [
    "ts", "tag", "layer", "framework", "model", "tp", "mode",
    "prefix_len", "output_len", "batch", "repeat", "concurrency",
    "latency_s", "output_tokens", "ok", "fail",
    "decode_tok_s", "ttft_ms", "tpot_ms", "note",
]

_tl = threading.local()


# ---------------------------------------------------------------- HTTP 底座

def _new_conn(url, timeout: float):
    if url.scheme == "https":
        import ssl
        return http.client.HTTPSConnection(
            url.hostname, url.port or 443, timeout=timeout,
            context=ssl._create_unverified_context())
    return http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)


def _conn(url, timeout):
    c = getattr(_tl, "conn", None)
    if c is None:
        c = _new_conn(url, timeout)
        _tl.conn = c
    return c


def _drop_conn():
    c = getattr(_tl, "conn", None)
    if c is not None:
        try:
            c.close()
        except Exception:
            pass
        _tl.conn = None


# ---------------------------------------------------------------- 流式请求

def stream_completion(url, path, headers, payload, timeout):
    """流式 completion 请求。

    返回 (ok, total_s, prompt_tokens, completion_tokens, err, ttft_s, decode_time_s)

    ttft_s        = 首 content chunk 时间（含 prefill/缓存命中 + 首步 decode）
    decode_time_s = (末 chunk - 首 chunk)，即纯 decode 迭代总耗时
    total_s       = 请求发出到 [DONE] 的墙钟
    """
    body = json.dumps(payload)
    t0 = time.perf_counter()
    for attempt in (0, 1):
        try:
            c = _conn(url, timeout)
            c.request("POST", path, body=body, headers=headers)
            resp = c.getresponse()
            if resp.status != 200:
                raw = resp.read()
                _drop_conn()
                dt = time.perf_counter() - t0
                return (False, dt, 0, 0,
                        "HTTP %d: %s" % (resp.status, raw[:200].decode("utf-8", "replace")),
                        0.0, 0.0)

            ttft = 0.0
            first_chunk_time = 0.0
            last_chunk_time = 0.0
            completion_tokens = 0
            prompt_tokens = 0
            got_first = False
            buf = b""

            while True:
                chunk = resp.read1(8192)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line or line.startswith(b":"):
                        continue
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].lstrip(b" ")
                    if data == b"[DONE]":
                        buf = b""
                        break
                    try:
                        obj = json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        text = choices[0].get("text", "") or ""
                        if text and not got_first:
                            got_first = True
                            first_chunk_time = time.perf_counter()
                            ttft = first_chunk_time - t0
                        if text:
                            last_chunk_time = time.perf_counter()
                    usage = obj.get("usage")
                    if usage:
                        completion_tokens = int(usage.get("completion_tokens", 0))
                        prompt_tokens = int(usage.get("prompt_tokens", 0))

            total = time.perf_counter() - t0
            if not got_first:
                first_chunk_time = time.perf_counter()
                ttft = total
            decode_time = max(0.0, last_chunk_time - first_chunk_time)
            return (True, total, prompt_tokens, completion_tokens, "",
                    ttft, decode_time)
        except Exception as e:
            _drop_conn()
            if attempt == 1:
                dt = time.perf_counter() - t0
                return (False, dt, 0, 0, "%s: %s" % (type(e).__name__, e),
                        0.0, 0.0)
    return (False, 0.0, 0, 0, "unreachable", 0.0, 0.0)


def nonstream_completion(url, path, headers, payload, timeout):
    """非流式回退（当服务端不支持 streaming 时使用）。

    返回格式同 stream_completion，但 decode_time 用 (total - ttft) 估算，
    其中 ttft 不可得，故 decode_time = total（含首步），tpot 含首步偏差。
    """
    body = json.dumps(payload)
    t0 = time.perf_counter()
    for attempt in (0, 1):
        try:
            c = _conn(url, timeout)
            c.request("POST", path, body=body, headers=headers)
            resp = c.getresponse()
            raw = resp.read()
            dt = time.perf_counter() - t0
            if resp.status != 200:
                _drop_conn()
                return (False, dt, 0, 0,
                        "HTTP %d: %s" % (resp.status, raw[:200].decode("utf-8", "replace")),
                        0.0, 0.0)
            data = json.loads(raw)
            usage = data.get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            return (True, dt, pt, ct, "", 0.0, dt)
        except Exception as e:
            _drop_conn()
            if attempt == 1:
                dt = time.perf_counter() - t0
                return (False, dt, 0, 0, "%s: %s" % (type(e).__name__, e),
                        0.0, 0.0)
    return (False, 0.0, 0, 0, "unreachable", 0.0, 0.0)


# ---------------------------------------------------------------- prompt 构造

def make_prefix(rng, length, vocab_size):
    """生成固定共享前缀。所有请求复用同一段 token id → 前缀缓存命中。"""
    lo, hi = 100, max(200, vocab_size - 1000)
    return [rng.randrange(lo, hi) for _ in range(length)]


def make_prompt_with_prefix(prefix, rng, vocab_size, new_tokens=1):
    """prefix + 随机新 token，每请求新 token 不同以避免整体缓存命中。"""
    lo, hi = 100, max(200, vocab_size - 1000)
    new = [rng.randrange(lo, hi) for _ in range(new_tokens)]
    return prefix + new


# ---------------------------------------------------------------- 指标探针

def fetch_cache_metrics(base_url):
    """返回 (命中率字符串, 是否非零)。"""
    try:
        u = urlparse(base_url)
        c = _new_conn(u, 10)
        c.request("GET", "/metrics")
        txt = c.getresponse().read().decode("utf-8", "replace")
        c.close()
    except Exception:
        return None, None
    keys = ("sglang:cache_hit_rate", "vllm:gpu_prefix_cache_hit_rate",
            "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
    hits = []
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in keys:
            if line.startswith(k):
                hits.append(line.strip())
    return ("; ".join(hits) if hits else None), None


# ---------------------------------------------------------------- batch 模式

def run_batch_cell(args, url, path, headers, pool, prefix, batch,
                   prefix_len, output_len, repeat, seed):
    """一次并发打 batch 发共享前缀请求，计整批墙钟。"""
    rng = random.Random(seed)
    payloads = []
    for _ in range(batch):
        p = {
            "model": args.model,
            "prompt": make_prompt_with_prefix(prefix, rng, args.vocab_size,
                                               args.new_tokens),
            "max_tokens": output_len,
            "temperature": 0.0,
            "stream": args.stream,
            "ignore_eos": True,
        }
        if args.stream:
            p["stream_options"] = {"include_usage": True}
        payloads.append(p)

    t0 = time.perf_counter()
    if args.stream:
        results = list(pool.map(
            lambda pl: stream_completion(url, path, headers, pl, args.timeout),
            payloads))
    else:
        results = list(pool.map(
            lambda pl: nonstream_completion(url, path, headers, pl, args.timeout),
            payloads))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r[0]]
    bad = [r for r in results if not r[0]]
    out_tokens = sum(r[3] for r in ok) or batch * output_len
    ttfts = [r[5] for r in ok if r[5] > 0]
    decode_times = [r[6] for r in ok if r[6] > 0]
    tpots = []
    for r in ok:
        ct = r[3] or output_len
        if ct > 1 and r[6] > 0:
            tpots.append(r[6] / (ct - 1) * 1000)

    notes = []
    if bad:
        notes.append(bad[0][4][:120])
    ptoks_set = set(r[2] for r in ok if r[2] > 0)
    if ptoks_set and max(ptoks_set) != prefix_len + args.new_tokens:
        notes.append("prompt_tokens=%s != %d" % (
            sorted(ptoks_set), prefix_len + args.new_tokens))
    ctoks_set = set(r[3] for r in ok if r[3] > 0)
    if ctoks_set and max(ctoks_set) != output_len:
        notes.append("completion_tokens=%s != %d (decode 被截断)" % (
            sorted(ctoks_set), output_len))

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag, "layer": "L2-http-decode", "framework": args.framework,
        "model": args.model, "tp": args.tp, "mode": "batch",
        "prefix_len": prefix_len, "output_len": output_len,
        "batch": batch, "repeat": repeat, "concurrency": batch,
        "latency_s": round(wall, 5), "output_tokens": out_tokens,
        "ok": len(ok), "fail": len(bad),
        "decode_tok_s": round(out_tokens / wall, 2) if wall > 0 else 0,
        "ttft_ms": round(statistics.median(ttfts) * 1000, 3) if ttfts else "",
        "tpot_ms": round(statistics.median(tpots), 3) if tpots else "",
        "note": " | ".join(notes),
    }


# ---------------------------------------------------------------- steady 模式

def run_steady_cell(args, url, path, headers, prefix,
                    prefix_len, output_len, concurrency, repeat, seed, duration):
    deadline = time.perf_counter() + duration
    lock = threading.Lock()
    stat = {"ok": 0, "fail": 0, "tokens": 0,
            "ttfts": [], "tpots": [], "err": ""}

    def worker(wid):
        rng = random.Random(seed * 100003 + wid)
        while time.perf_counter() < deadline:
            payload = {
                "model": args.model,
                "prompt": make_prompt_with_prefix(prefix, rng, args.vocab_size,
                                                   args.new_tokens),
                "max_tokens": output_len,
                "temperature": 0.0,
                "stream": args.stream,
                "ignore_eos": True,
            }
            if args.stream:
                payload["stream_options"] = {"include_usage": True}
            if args.stream:
                good, dt, ptok, ctok, err, ttft, dtime = stream_completion(
                    url, path, headers, payload, args.timeout)
            else:
                good, dt, ptok, ctok, err, ttft, dtime = nonstream_completion(
                    url, path, headers, payload, args.timeout)
            with lock:
                if good:
                    stat["ok"] += 1
                    stat["tokens"] += ctok or output_len
                    if ttft > 0:
                        stat["ttfts"].append(ttft)
                    ct = ctok or output_len
                    if ct > 1 and dtime > 0:
                        stat["tpots"].append(dtime / (ct - 1) * 1000)
                else:
                    stat["fail"] += 1
                    if not stat["err"]:
                        stat["err"] = err[:120]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    wall = time.perf_counter() - t0

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag, "layer": "L2-http-decode", "framework": args.framework,
        "model": args.model, "tp": args.tp, "mode": "steady",
        "prefix_len": prefix_len, "output_len": output_len,
        "batch": "", "repeat": repeat, "concurrency": concurrency,
        "latency_s": round(wall, 3), "output_tokens": stat["tokens"],
        "ok": stat["ok"], "fail": stat["fail"],
        "decode_tok_s": round(stat["tokens"] / wall, 2) if wall > 0 else 0,
        "ttft_ms": round(statistics.median(stat["ttfts"]) * 1000, 3) if stat["ttfts"] else "",
        "tpot_ms": round(statistics.median(stat["tpots"]), 3) if stat["tpots"] else "",
        "note": stat["err"],
    }


# ---------------------------------------------------------------- 缓存预热

def warmup_prefix(args, url, path, headers, prefix, prefix_len, seed):
    """发送 1 个请求（prefix + 1 token + max_tokens=1）填充前缀缓存。"""
    payload = {
        "model": args.model,
        "prompt": prefix + [random.Random(seed).randrange(100, max(200, args.vocab_size - 1000))],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }
    ok, dt, pt, ct, err, _, _ = nonstream_completion(
        url, path, headers, payload, args.timeout)
    if ok:
        print("[warmup] prefix_len=%d  prompt_tokens=%d  ok  %.3fs"
              % (prefix_len, pt, dt), file=sys.stderr)
    else:
        print("[warmup] prefix_len=%d  FAIL  %s" % (prefix_len, err),
              file=sys.stderr)
    return ok


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="纯 decode 吞吐扫描（共享前缀 + 前缀缓存 + 流式 SSE）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--path", default="/v1/completions")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--mode", choices=["batch", "steady"], default="batch")
    ap.add_argument("--prefix-lens", default="4096,32768,65536",
                    help="共享前缀长度（模拟 KV cache 已就位）")
    ap.add_argument("--output-lens", default="1024",
                    help="decode 输出长度")
    ap.add_argument("--batches", default="1,4,8,16,32,64,128",
                    help="batch 模式的并发数列表")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="steady 模式并发数")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="steady 模式持续时间（秒）")
    ap.add_argument("--new-tokens", type=int, default=1,
                    help="每个请求在共享前缀之后新增的 token 数（保持 1 以确保 prefill 近似为零）")
    ap.add_argument("--token-budget", type=int, default=196608,
                    help="跳过 prefix_len + output_len * batch 超过此值的格子")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2,
                    help="预热轮数（JIT/CUDA graph + 前缀缓存填充）")
    ap.add_argument("--vocab-size", type=int, default=151552)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--stream", action="store_true", default=True,
                    help="使用流式 SSE 精确测量 TTFT/TPOT（默认开启）")
    ap.add_argument("--no-stream", dest="stream", action="store_false",
                    help="禁用流式，回退到非流式（TPOT 含首步偏差）")
    ap.add_argument("--framework", default="unknown")
    ap.add_argument("--tp", default="")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="results/decode_http.csv")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()
    if args.repeats < 3:
        print("[WARN] --repeats=%d < 3：中位数无法排除单个离群值，建议 >=3"
              % args.repeats, file=sys.stderr)

    url = urlparse(args.base_url)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key

    before, _ = fetch_cache_metrics(args.base_url)
    print("[preflight] prefix-cache metrics: %s" % (before or "(不可得)"),
          file=sys.stderr)
    if before is None:
        print("[!] 无法获取 /metrics —— 请确认服务端已开启前缀缓存：\n"
              "    SGLang: 不要加 --disable-radix-cache（默认开启）\n"
              "    vLLM:   不要加 --no-enable-prefix-caching（默认开启）",
              file=sys.stderr)

    prefix_lens = [int(x) for x in args.prefix_lens.split(",") if x.strip()]
    output_lens = [int(x) for x in args.output_lens.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        w.writeheader()

    n = 0
    try:
        if args.mode == "batch":
            for PL in prefix_lens:
                prefix = make_prefix(random.Random(args.seed), PL, args.vocab_size)
                for OL in output_lens:
                    # 预热：填充前缀缓存 + JIT/CUDA graph
                    for i in range(args.warmup):
                        warmup_prefix(args, url, args.path, headers,
                                      prefix, PL, args.seed + 90000 + i)
                    for B in batches:
                        if PL + OL * B > args.token_budget:
                            print("[skip] PL=%d OL=%d B=%d -> budget %d"
                                  % (PL, OL, B, args.token_budget),
                                  file=sys.stderr)
                            continue
                        pool = ThreadPoolExecutor(max_workers=B)
                        try:
                            cell = []
                            for r in range(args.repeats):
                                row = run_batch_cell(
                                    args, url, args.path, headers, pool,
                                    prefix, B, PL, OL, r,
                                    args.seed + PL * 31 + OL * 17 + B * 7 + r)
                                w.writerow(row)
                                fh.flush()
                                cell.append(row)
                                n += 1
                                print("[batch] PL=%6d OL=%5d B=%3d r=%d  "
                                      "%10.0f tok/s  TTFT=%sms  TPOT=%sms  "
                                      "fail=%d %s" % (
                                          PL, OL, B, r, row["decode_tok_s"],
                                          row["ttft_ms"], row["tpot_ms"],
                                          row["fail"], row["note"]),
                                      file=sys.stderr)
                            vals = [c["decode_tok_s"] for c in cell]
                            med = statistics.median(vals)
                            flag = ""
                            if min(vals) > 0 and max(vals) / min(vals) > 1.5:
                                flag = ("   [!] 同格波动 %.1fx"
                                        % (max(vals) / min(vals)))
                            print("  +- median = %.0f tok/s%s"
                                  % (med, flag), file=sys.stderr)
                        finally:
                            pool.shutdown(wait=True)
        else:
            for PL in prefix_lens:
                prefix = make_prefix(random.Random(args.seed), PL, args.vocab_size)
                for OL in output_lens:
                    for i in range(args.warmup):
                        warmup_prefix(args, url, args.path, headers,
                                      prefix, PL, args.seed + 90000 + i)
                    for r in range(args.repeats):
                        row = run_steady_cell(
                            args, url, args.path, headers, prefix,
                            PL, OL, args.concurrency, r,
                            args.seed + PL * 31 + OL * 17 + r, args.duration)
                        w.writerow(row)
                        fh.flush()
                        n += 1
                        print("[steady] PL=%6d OL=%5d C=%3d r=%d  "
                              "%10.0f tok/s  ok=%d fail=%d  TTFT=%sms  TPOT=%sms"
                              % (PL, OL, args.concurrency, r,
                                 row["decode_tok_s"], row["ok"], row["fail"],
                                 row["ttft_ms"], row["tpot_ms"]),
                              file=sys.stderr)
    finally:
        fh.close()

    after, _ = fetch_cache_metrics(args.base_url)
    print("[postflight] prefix-cache metrics: %s" % (after or "(不可得)"),
          file=sys.stderr)
    print("\n写入 %s（%d 行）。" % (args.out, n), file=sys.stderr)


if __name__ == "__main__":
    main()
