"""Async load driver. Sweeps concurrency levels against a single endpoint
and reports throughput + latency percentiles.

Usage:
    python -m bench.drive --url http://127.0.0.1:8000/q1 --label pxt
    python -m bench.drive --url http://127.0.0.1:8001/q1 --label pg
"""

import argparse
import asyncio
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

N_ROWS = 100_000
WARMUP_REQUESTS = 1_000
DURATION_S = 20.0
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]


async def one_request(client: httpx.AsyncClient, url: str, i: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    try:
        r = await client.get(url, params={'i': i})
        return time.perf_counter() - t0, r.status_code
    except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
        return time.perf_counter() - t0, -1


async def worker(client: httpx.AsyncClient, url: str, stop_at: float, latencies: list[float], errors: list[int]) -> None:
    while time.perf_counter() < stop_at:
        i = random.randint(0, N_ROWS - 1)
        lat, status = await one_request(client, url, i)
        latencies.append(lat)
        if status != 200:
            errors.append(status)


async def warmup(client: httpx.AsyncClient, url: str) -> None:
    sem = asyncio.Semaphore(16)

    async def hit(i: int) -> None:
        async with sem:
            await client.get(url, params={'i': i})

    await asyncio.gather(*(hit(random.randint(0, N_ROWS - 1)) for _ in range(WARMUP_REQUESTS)))


async def measure(url: str, concurrency: int) -> dict[str, float]:
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
        if concurrency == CONCURRENCY_LEVELS[0]:
            await warmup(client, url)
        latencies: list[float] = []
        errors: list[int] = []
        stop_at = time.perf_counter() + DURATION_S
        await asyncio.gather(*(worker(client, url, stop_at, latencies, errors) for _ in range(concurrency)))

    elapsed = DURATION_S
    n = len(latencies)
    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return float('nan')
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return latencies[idx] * 1000

    return {
        'concurrency': concurrency,
        'requests': n,
        'errors': len(errors),
        'throughput_rps': n / elapsed,
        'p50_ms': pct(0.50),
        'p90_ms': pct(0.90),
        'p99_ms': pct(0.99),
        'p999_ms': pct(0.999),
        'mean_ms': statistics.fmean(latencies) * 1000 if latencies else float('nan'),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', default=None, help='CSV output path (default bench/results-<label>.csv)')
    ap.add_argument('--levels', type=int, nargs='+', default=CONCURRENCY_LEVELS)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).with_name(f'results-{args.label}.csv')
    rows = []
    print(f'{"conc":>5} {"req":>8} {"rps":>9} {"mean":>8} {"p50":>8} {"p90":>8} {"p99":>8} {"p999":>8} {"err":>4}')
    for c in args.levels:
        r = await measure(args.url, c)
        rows.append(r)
        print(
            f'{r["concurrency"]:>5} {r["requests"]:>8d} {r["throughput_rps"]:>9.1f} '
            f'{r["mean_ms"]:>8.2f} {r["p50_ms"]:>8.2f} {r["p90_ms"]:>8.2f} '
            f'{r["p99_ms"]:>8.2f} {r["p999_ms"]:>8.2f} {r["errors"]:>4d}'
        )
        sys.stdout.flush()

    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    asyncio.run(main())
