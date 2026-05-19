"""
Benchmark driver for the Pixeltable FastAPI server.

Fires concurrent insert and lookup requests and reports throughput / latency
percentiles.

Usage:
    python -m bench.bench_driver --concurrency 1 --total-inserts 10 --batch-size 2 --total-lookups 5
    python -m bench.bench_driver --concurrency 8 --total-inserts 1000 --batch-size 10 --total-lookups 500
"""

import argparse
import asyncio
import itertools
import random
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
import uvloop


# --- data types ---

@dataclass
class RequestMetric:
    endpoint: str
    status_code: int
    elapsed_ms: float
    rows: int
    server_elapsed_ms: float
    ok: bool


@dataclass
class PhaseResult:
    name: str
    metrics: list[RequestMetric] = field(default_factory=list)
    wall_seconds: float = 0.0


# --- row generator ---

_counter = itertools.count()


def make_batch(thread_id: int, batch_size: int) -> list[dict]:
    return [
        {'thread_id': thread_id, 'row_idx': next(_counter), 'value': next(_counter)}
        for _ in range(batch_size)
    ]


# --- core request helpers ---

async def do_insert(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, thread_id: int, batch_size: int
) -> RequestMetric:
    rows = make_batch(thread_id, batch_size)
    async with sem:
        start = time.perf_counter()
        resp = await client.post('/insert', json={'rows': rows})
        elapsed_ms = (time.perf_counter() - start) * 1000
    ok = resp.status_code == 200
    body = resp.json() if ok else {}
    return RequestMetric(
        endpoint='insert',
        status_code=resp.status_code,
        elapsed_ms=round(elapsed_ms, 2),
        rows=body.get('num_rows', 0),
        server_elapsed_ms=body.get('elapsed_ms', 0),
        ok=ok,
    )


async def do_lookup(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, thread_id: int, limit: int = 50
) -> RequestMetric:
    async with sem:
        start = time.perf_counter()
        resp = await client.get('/lookup', params={'thread_id': thread_id, 'limit': limit})
        elapsed_ms = (time.perf_counter() - start) * 1000
    ok = resp.status_code == 200
    body = resp.json() if ok else {}
    return RequestMetric(
        endpoint='lookup',
        status_code=resp.status_code,
        elapsed_ms=round(elapsed_ms, 2),
        rows=body.get('count', 0),
        server_elapsed_ms=body.get('elapsed_ms', 0),
        ok=ok,
    )


# --- phase runners ---

async def run_insert_phase(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    total: int, batch_size: int, concurrency: int,
) -> PhaseResult:
    result = PhaseResult(name='INSERT')
    tasks = [do_insert(client, sem, i % concurrency, batch_size) for i in range(total)]
    start = time.perf_counter()
    result.metrics = await asyncio.gather(*tasks)
    result.wall_seconds = time.perf_counter() - start
    return result


async def run_lookup_phase(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    total: int, concurrency: int,
) -> PhaseResult:
    result = PhaseResult(name='LOOKUP')
    tasks = [do_lookup(client, sem, random.randint(0, concurrency - 1)) for _ in range(total)]
    start = time.perf_counter()
    result.metrics = await asyncio.gather(*tasks)
    result.wall_seconds = time.perf_counter() - start
    return result


async def run_mixed_phase(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    total_inserts: int, total_lookups: int, batch_size: int, concurrency: int,
) -> PhaseResult:
    result = PhaseResult(name='MIXED')
    tasks = []
    tasks.extend(do_insert(client, sem, i % concurrency, batch_size) for i in range(total_inserts))
    tasks.extend(do_lookup(client, sem, random.randint(0, concurrency - 1)) for _ in range(total_lookups))
    random.shuffle(tasks)
    start = time.perf_counter()
    result.metrics = await asyncio.gather(*tasks)
    result.wall_seconds = time.perf_counter() - start
    return result


# --- reporting ---

def report(phase: PhaseResult) -> None:
    metrics = phase.metrics
    ok = [m for m in metrics if m.ok]
    err = [m for m in metrics if not m.ok]
    total_rows = sum(m.rows for m in ok)
    latencies = np.array([m.elapsed_ms for m in ok]) if ok else np.array([0.0])

    req_rate = len(metrics) / phase.wall_seconds if phase.wall_seconds > 0 else 0
    row_rate = total_rows / phase.wall_seconds if phase.wall_seconds > 0 else 0

    print(f'\n=== {phase.name} Phase ===')
    print(f'Requests:       {len(metrics)} ({len(ok)} ok, {len(err)} err)')
    print(f'Rows:           {total_rows}')
    print(f'Duration:       {phase.wall_seconds:.2f}s')
    print(f'Throughput:     {req_rate:.1f} req/s  ({row_rate:.1f} rows/s)')
    if ok:
        print(f'Latency p50:    {np.percentile(latencies, 50):.1f} ms')
        print(f'Latency p95:    {np.percentile(latencies, 95):.1f} ms')
        print(f'Latency p99:    {np.percentile(latencies, 99):.1f} ms')

    if err:
        codes = {}
        for m in err:
            codes[m.status_code] = codes.get(m.status_code, 0) + 1
        print(f'Errors:         {codes}')


# --- main ---

async def main(args: argparse.Namespace) -> None:
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=60.0) as client:
        # health check
        resp = await client.get('/health')
        resp.raise_for_status()
        print(f'Server healthy at {args.url}')

        phases = {'insert', 'lookup', 'mixed'} if args.phase == 'all' else {args.phase}

        # warmup
        if args.warmup > 0:
            print(f'\nWarmup: {args.warmup} insert + lookup requests ...')
            warmup_tasks = []
            for i in range(args.warmup):
                warmup_tasks.append(do_insert(client, sem, i % args.concurrency, args.batch_size))
                warmup_tasks.append(do_lookup(client, sem, i % args.concurrency))
            await asyncio.gather(*warmup_tasks)
            print('Warmup done.')

        if 'insert' in phases:
            result = await run_insert_phase(client, sem, args.total_inserts, args.batch_size, args.concurrency)
            report(result)

            # verify row count
            resp = await client.get('/count')
            if resp.status_code == 200:
                print(f'Server row count: {resp.json()["count"]}')

        if 'lookup' in phases:
            result = await run_lookup_phase(client, sem, args.total_lookups, args.concurrency)
            report(result)

        if 'mixed' in phases:
            result = await run_mixed_phase(
                client, sem, args.total_inserts // 2, args.total_lookups // 2,
                args.batch_size, args.concurrency,
            )
            report(result)


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Pixeltable FastAPI benchmark driver')
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--total-inserts', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--total-lookups', type=int, default=500)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--phase', choices=['insert', 'lookup', 'mixed', 'all'], default='all')
    return p.parse_args()


if __name__ == '__main__':
    uvloop.install()
    args = cli()
    asyncio.run(main(args))
