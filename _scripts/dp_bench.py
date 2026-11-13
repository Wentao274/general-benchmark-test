#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dp_bench.py — DP 均衡 GSP 客户端（decode_dp_bench.sh / cache_dp_bench.sh 的执行核心）

背景（DP >= 2 的数据并行部署，组内 TP 任意，如 TP=1/DP=8、TP=2/DP=4）：
    上游 bench 工具（sglang.benchmark.serving / vllm bench serve）的数据集按"组块顺序"
    生成（组 0 的 P 条 → 组 1 的 P 条 → …），滑窗并发下在飞窗口内前缀高度集中，
    服务端 DP 路由（最短队列/缓存亲和）会把同批请求聚到少数 rank，DP 严重不均衡。
    上游工具无顺序/批次/交错参数可传，客户端也无法指定 rank。

本客户端把"请求组合 + 提交顺序 + 批节奏"完全接管：
    每个请求 = 组 g 的共享前缀（seed+g 确定性生成，token id 级）+ 每请求唯一后缀
    （decode: 1 个新 token；cache: 唯一后缀长度 = 输入 × 未命中率）。

两阶段（由 wrapper 分别调用，服务端 /metrics 计数器在两阶段间快照）：
    --phase warmup   轮转预热：DP_SIZE 轮 × 每轮每组 1 条（轮内组序逐轮轮转）。
                     round_robin 分派下每 rank 把 G 个前缀各缓存一遍 →
                     后续批间 shuffle 组序不会破坏缓存命中。
                     --warmup-mode affinity：只发 G 条（每组 1 条，固定顺序），
                     适用于服务端为"前缀亲和路由"的部署（同前缀永远同 rank），
                     此时必须配合 --no-shuffle。
    --phase test     均衡发送，两种模式：
        refill（默认）：按"层"交错（第 l 层 = 每组第 l 条，层内组序随机），滑窗补位，
                持续并发口径（与上游工具相同的测量语义，请求流不中断），
                稳态下在飞窗口为序列连续段 → 任意时刻各组在飞条数组间均衡（±1）。
        barrier：每批从每组各取 k = max(1, 并发/组数) 条，批内并发提交，
                等全部完成再发下一批（批 barrier，批间有周转间隙与批尾等待）；
                批间随机打乱组序（--no-shuffle 关闭）→ 每批组成严格一组一条，
                适合验证 DP 均衡是否生效。
    注：barrier 模式要求 并发 >= 组数，否则自动回退 refill。有效并发 = G×floor(C/G)。

指标输出（stdout，与 collect_results.py 的解析规则逐行兼容）：
    Successful requests / Benchmark duration / Input|Output|Total token throughput /
    Request throughput / Mean|Median|P90|P95|P99 TTFT|TPOT (ms)。
    TTFT/TPOT 由流式 SSE 首/末 chunk 时间得出（--no-stream 回退非流式，TPOT 含首步偏差）。
    另输出每批/每层组序等 DP 审计信息（stderr，不影响解析）。

前置条件：服务端开启前缀缓存（decode ~100% 命中 / cache ~R% 命中的前提）。

用法（一般不直接调用，由 wrapper 驱动）：
    python3 dp_bench.py --phase warmup --base-url http://127.0.0.1:8080 --model m \
        --gsp-num-groups 8 --dp-size 8 --prefix-len 8192 --suffix-len 1 --seed 123
    python3 dp_bench.py --phase test   ... --output-len 512 --num-prompts 40 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import http.client
import math
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

from urllib.parse import urlparse

_tl = threading.local()


# ---------------------------------------------------------------- HTTP 底座（同 decode_http_sweep.py）

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


def stream_completion(url, path, headers, payload, timeout):
    """流式 completion。返回 (ok, total_s, prompt_tokens, completion_tokens, err, ttft_s, decode_time_s)"""
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
            first_chunk = 0.0
            last_chunk = 0.0
            completion_tokens = 0
            got_first = False
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                now = time.perf_counter()
                if not got_first:
                    ttft = now - t0
                    first_chunk = now
                    got_first = True
                last_chunk = now
                buf += chunk
            # 解析 usage（stream_options.include_usage 时在最后一个 data chunk）
            for line in reversed(buf.decode("utf-8", "replace").splitlines()):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                body_txt = line[5:].strip()
                if not body_txt or body_txt == "[DONE]":
                    continue
                try:
                    data = json.loads(body_txt)
                except Exception:
                    continue
                usage = data.get("usage") or {}
                if usage.get("completion_tokens"):
                    completion_tokens = int(usage["completion_tokens"])
                break
            _ = first_chunk
            total = time.perf_counter() - t0
            decode_time = (last_chunk - first_chunk) if got_first else 0.0
            return (True, total, 0, completion_tokens, "", ttft, decode_time)
        except Exception as e:
            _drop_conn()
            if attempt == 1:
                dt = time.perf_counter() - t0
                return (False, dt, 0, 0, "%s: %s" % (type(e).__name__, e), 0.0, 0.0)
    return (False, 0.0, 0, 0, "unreachable", 0.0, 0.0)


def nonstream_completion(url, path, headers, payload, timeout):
    """非流式 completion。返回同 stream_completion（ttft=0, decode_time=total）。"""
    body = json.dumps(payload)
    t0 = time.perf_counter()
    for attempt in (0, 1):
        try:
            c = _conn(url, timeout)
            c.request("POST", path, body=body, headers=headers)
            resp = c.getresponse()
            raw = resp.read()
            _drop_conn()
            dt = time.perf_counter() - t0
            if resp.status != 200:
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
                return (False, dt, 0, 0, "%s: %s" % (type(e).__name__, e), 0.0, 0.0)
    return (False, 0.0, 0, 0, "unreachable", 0.0, 0.0)


# ---------------------------------------------------------------- 数据构造

def make_prefix(rng, length, vocab_size):
    """组共享前缀（token id 列表）。同 seed+组号 → 同前缀 → 组内缓存命中。"""
    lo, hi = 100, max(200, vocab_size - 1000)
    return [rng.randrange(lo, hi) for _ in range(length)]


def make_suffix(rng, length, vocab_size):
    lo, hi = 100, max(200, vocab_size - 1000)
    return [rng.randrange(lo, hi) for _ in range(length)]


# ---------------------------------------------------------------- 统计辅助

def percentile(vals, p):
    """nearest-rank 百分位。"""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, math.ceil(p * len(s) / 100.0) - 1)
    k = min(k, len(s) - 1)
    return s[k]


# ---------------------------------------------------------------- 阶段 1：轮转预热

def run_warmup(args, url, headers):
    G = args.gsp_num_groups
    rounds = args.dp_size if args.warmup_mode == "rotate" else 1
    prefixes = [make_prefix(random.Random(args.seed + g), args.prefix_len, args.vocab_size)
                for g in range(G)]
    total_ok = 0
    total_fail = 0
    with ThreadPoolExecutor(max_workers=max(G, 1)) as pool:
        for r in range(rounds):
            if args.warmup_mode == "rotate":
                order = [(g + r) % G for g in range(G)]   # 逐轮轮转 → 每 rank 覆盖全部组
            else:
                order = list(range(G))                    # 亲和：固定顺序，只填一轮
            payloads = []
            for g in order:
                # warmup 后缀走独立种子段，不与测试请求的后缀重复
                rng = random.Random(args.seed + g * 1000003 + 777000000 + r)
                prompt = prefixes[g] + make_suffix(rng, args.suffix_len, args.vocab_size)
                payloads.append({
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                    "ignore_eos": True,
                })
            t0 = time.perf_counter()
            results = list(pool.map(
                lambda pl: nonstream_completion(url, args.path, headers, pl, args.timeout),
                payloads))
            wall = time.perf_counter() - t0
            ok = sum(1 for x in results if x[0])
            fail = len(results) - ok
            total_ok += ok
            total_fail += fail
            print(">>> [dp-warmup] round %d/%d groups=%d order=%s ok=%d fail=%d %.2fs"
                  % (r + 1, rounds, G, " ".join(map(str, order)), ok, fail, wall),
                  flush=True)
    if total_fail:
        print("!!! [dp-warmup] %d request(s) failed" % total_fail, flush=True)
        return 1
    print(">>> [dp-warmup] done: %d round(s) x %d group(s), all prefixes cached"
          % (rounds, G), flush=True)
    return 0


# ---------------------------------------------------------------- 阶段 2：均衡测试

def run_test(args, url, headers):
    G = args.gsp_num_groups
    C = args.concurrency
    N = args.num_prompts
    stream = args.stream

    prefixes = [make_prefix(random.Random(args.seed + g), args.prefix_len, args.vocab_size)
                for g in range(G)]

    # 每组配额（N 按组均摊，余数给前 N%G 组）
    q_base, extra = divmod(N, G)
    quota = [q_base + (1 if g < extra else 0) for g in range(G)]
    max_quota = max(quota)

    send_mode = args.send_mode
    if send_mode == "barrier" and C < G:
        print(">>> [dp] barrier 需要并发 >= 组数 (C=%d < G=%d)，回退 refill" % (C, G),
              file=sys.stderr, flush=True)
        send_mode = "refill"

    def batch_order(tag):
        """某一批/层的组序：shuffle 开启时随机打乱，否则恒定 [0..G-1]。"""
        order = list(range(G))
        if args.shuffle:
            random.Random(args.seed + tag * 7919 + 13).shuffle(order)
        return order

    workers = max(C, 1)
    pool = ThreadPoolExecutor(max_workers=workers)
    recs = []          # 每请求记录 (ok, total, ptok, ctok, ttft, dtime)
    batch_audit = []   # (batch_id, order, n, ok)

    t_start = time.perf_counter()

    def do_one(payload):
        if stream:
            return stream_completion(url, args.path, headers, payload, args.timeout)
        return nonstream_completion(url, args.path, headers, payload, args.timeout)

    try:
        if send_mode == "barrier":
            k = max(1, C // G)                     # 每组每批取 k 条
            n_batches = max(1, math.ceil(max_quota / k))
            for b in range(n_batches):
                base = b * k
                order = batch_order(b)
                count = [max(0, min(k, quota[g] - base)) for g in range(G)]
                payloads = []
                for g in order:
                    for j in range(count[g]):
                        idx = base + j
                        rng = random.Random(args.seed + g * 1000003 + idx)
                        prompt = prefixes[g] + make_suffix(rng, args.suffix_len,
                                                           args.vocab_size)
                        p = {"model": args.model, "prompt": prompt,
                             "max_tokens": args.output_len, "temperature": 0.0,
                             "stream": stream, "ignore_eos": True}
                        if stream:
                            p["stream_options"] = {"include_usage": True}
                        payloads.append(p)
                if not payloads:
                    break
                t0 = time.perf_counter()
                results = list(pool.map(do_one, payloads))
                wall = time.perf_counter() - t0
                recs.extend(results)
                ok = sum(1 for x in results if x[0])
                batch_audit.append((b + 1, order, len(payloads), ok, wall))
                print(">>> [dp-batch] %d/%d groups=%d size=%d ok=%d %.2fs"
                      % (b + 1, n_batches, G, len(payloads), ok, wall),
                      file=sys.stderr, flush=True)
        else:
            # refill：层交错 + 滑窗补位，持续并发口径
            seq = []   # (g, idx)
            for l in range(max_quota):
                for g in batch_order(l):
                    if l < quota[g]:
                        seq.append((g, l))
            futures = set()
            it = iter(seq)
            stop = False
            while True:
                while len(futures) < C and not stop:
                    try:
                        g, idx = next(it)
                    except StopIteration:
                        stop = True
                        break
                    rng = random.Random(args.seed + g * 1000003 + idx)
                    prompt = prefixes[g] + make_suffix(rng, args.suffix_len,
                                                        args.vocab_size)
                    p = {"model": args.model, "prompt": prompt,
                         "max_tokens": args.output_len, "temperature": 0.0,
                         "stream": stream, "ignore_eos": True}
                    if stream:
                        p["stream_options"] = {"include_usage": True}
                    futures.add(pool.submit(do_one, p))
                if not futures:
                    break
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    recs.append(f.result())
            layers = max_quota
            print(">>> [dp-refill] layers=%d window=%d total=%d" % (layers, C, len(recs)),
                  file=sys.stderr, flush=True)
    finally:
        pool.shutdown(wait=True)

    duration = time.perf_counter() - t_start

    # ---------- 汇总（collect_results.py 兼容格式） ----------
    ok_recs = [r for r in recs if r[0]]
    fail_n = len(recs) - len(ok_recs)

    total_ptok = 0
    total_ctok = 0
    ttfts = []
    tpots = []
    for r in ok_recs:
        pt = r[2] or (args.prefix_len + args.suffix_len)
        ct = r[3] or args.output_len
        total_ptok += pt
        total_ctok += ct
        if r[5] > 0:
            ttfts.append(r[5] * 1000.0)
        if ct > 1 and r[6] > 0:
            tpots.append(r[6] / (ct - 1) * 1000.0)

    itps = total_ptok / duration if duration > 0 else 0.0
    otps = total_ctok / duration if duration > 0 else 0.0
    ttps = itps + otps
    rps = len(ok_recs) / duration if duration > 0 else 0.0

    print("", flush=True)
    print("---------- DP-Bench Summary ----------", flush=True)
    print("Successful requests: %d" % len(ok_recs), flush=True)
    print("Benchmark duration (s): %.2f" % duration, flush=True)
    print("Input token throughput (tok/s): %.2f" % itps, flush=True)
    print("Output token throughput (tok/s): %.2f" % otps, flush=True)
    print("Total token throughput (tok/s): %.2f" % ttps, flush=True)
    print("Request throughput (req/s): %.2f" % rps, flush=True)
    print("Mean TTFT (ms): %.1f" % (statistics.mean(ttfts) if ttfts else 0.0), flush=True)
    print("Median TTFT (ms): %.1f" % percentile(ttfts, 50), flush=True)
    print("P90 TTFT (ms): %.1f" % percentile(ttfts, 90), flush=True)
    print("P95 TTFT (ms): %.1f" % percentile(ttfts, 95), flush=True)
    print("P99 TTFT (ms): %.1f" % percentile(ttfts, 99), flush=True)
    print("Mean TPOT (ms): %.1f" % (statistics.mean(tpots) if tpots else 0.0), flush=True)
    print("Median TPOT (ms): %.1f" % percentile(tpots, 50), flush=True)
    print("P90 TPOT (ms): %.1f" % percentile(tpots, 90), flush=True)
    print("P95 TPOT (ms): %.1f" % percentile(tpots, 95), flush=True)
    print("P99 TPOT (ms): %.1f" % percentile(tpots, 99), flush=True)

    print("---------- DP Balance Audit ----------", flush=True)
    print("send_mode=%s groups=%d dp_size=%d shuffle=%s warmup_mode=%s"
          % (send_mode, G, args.dp_size, "on" if args.shuffle else "off",
             args.warmup_mode), flush=True)
    for b, order, n, ok, wall in batch_audit:
        print("batch %d: group order = %s, size=%d, ok=%d, %.2fs"
              % (b, " ".join(map(str, order)), n, ok, wall), flush=True)
    if fail_n:
        print("failed requests: %d (e.g. %s)" % (fail_n, recs[0][4][:120]), flush=True)

    return 0 if ok_recs else 1


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="DP 均衡 GSP 客户端（decode_dp_bench.sh / cache_dp_bench.sh 的执行核心）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="服务模型名（payload 的 model 字段）")
    ap.add_argument("--path", default="/v1/completions")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--phase", choices=["warmup", "test"], required=True,
                    help="warmup=轮转预热（不计量）；test=均衡测试（输出指标）")
    ap.add_argument("--gsp-num-groups", type=int, default=1,
                    help="共享前缀组数（组内共享，组间不同）")
    ap.add_argument("--dp-size", type=int, default=8,
                    help="服务端 DP rank 数（rotate 预热轮数 = dp-size）")
    ap.add_argument("--prefix-len", type=int, required=True, help="共享前缀 token 数")
    ap.add_argument("--suffix-len", type=int, required=True,
                    help="每请求唯一后缀 token 数（decode=1；cache=输入×未命中率）")
    ap.add_argument("--output-len", type=int, default=1, help="输出 token 数")
    ap.add_argument("--num-prompts", type=int, default=0, help="test 阶段总请求数")
    ap.add_argument("--concurrency", type=int, default=1, help="test 阶段目标并发")
    ap.add_argument("--send-mode", choices=["barrier", "refill"], default="refill",
                    help="refill=滑窗补位持续并发（默认，稳态压测）；"
                         "barrier=逐批同步（每批每组一条、批间 barrier，验证均衡用）")
    ap.add_argument("--warmup-mode", choices=["rotate", "affinity"], default="rotate",
                    help="rotate=DP_SIZE 轮 × 每组 1 条（round_robin 路由用）；"
                         "affinity=G 条固定顺序（前缀亲和路由用，需 --no-shuffle）")
    ap.add_argument("--shuffle", dest="shuffle", action="store_true", default=True,
                    help="批间随机打乱组序（默认开）")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    help="关闭批间打乱（前缀亲和路由部署必须关闭）")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--vocab-size", type=int, default=151552)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--stream", dest="stream", action="store_true", default=True,
                    help="流式 SSE 精确测量 TTFT/TPOT（默认开）")
    ap.add_argument("--no-stream", dest="stream", action="store_false",
                    help="非流式回退（TPOT 含首步偏差）")
    args = ap.parse_args()

    if args.gsp_num_groups < 1:
        ap.error("--gsp-num-groups 必须 >= 1")
    if args.dp_size < 1:
        ap.error("--dp-size 必须 >= 1")
    if args.prefix_len < 1 or args.suffix_len < 1:
        ap.error("--prefix-len / --suffix-len 必须 >= 1")
    if args.phase == "test":
        if args.num_prompts < 1:
            ap.error("test 阶段需要 --num-prompts >= 1")
        if args.concurrency < 1:
            ap.error("test 阶段需要 --concurrency >= 1")

    url = urlparse(args.base_url)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key

    if args.phase == "warmup":
        sys.exit(run_warmup(args, url, headers))
    sys.exit(run_test(args, url, headers))


if __name__ == "__main__":
    main()
