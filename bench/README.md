# Pixeltable benchmarks

Two independent suites live here:

1. **Key-lookup latency** (`pxt serve` vs raw SQLAlchemy) — files `setup.py`, `queries.py`, `pg_serve.py`, `pg_realistic_serve.py`, `drive.py`, `profile_server.py`. Compares the per-request cost of a single `@pxt.query` endpoint exposed via `pxt serve` against a hand-written SQLAlchemy baseline that hits the same physical postgres table.
2. **Insert + lookup throughput** (`bench_server.py`, `bench_driver.py`) — a self-contained FastAPI app that defines an insert / lookup / count surface over a pixeltable table, plus a driver that runs concurrent insert / lookup / mixed phases and reports throughput + latency percentiles.

CSV result files (`results-*.csv`), profile dumps (`*.pstats`), and the generated `pg_config.json` are not tracked — they are produced by running the suites.

## Key-lookup benchmark

Measures throughput / latency for a single key-lookup endpoint exposed via `pxt serve`, against a SQLAlchemy-direct baseline that hits the same physical postgres table.

### Files

- `setup.py` — drops/recreates `bench.lookup` (single int PK + one computed column `doubled = id * 2`), inserts 100k rows, and writes `pg_config.json` with the postgres connection details + storage table/column names.
- `queries.py` — defines `q1(i: int)` as a `@pxt.query`; importable as `bench.queries.q1`.
- `pg_serve.py` — FastAPI app with one endpoint that runs the equivalent SQL via SQLAlchemy directly. Same lookup predicate as pixeltable (`id == :i AND v_max == MAX_INT`).
- `pg_realistic_serve.py` — same idea as `pg_serve.py`, with response-shape parity (pydantic model + ORM-style fetch) to factor out serialization differences.
- `drive.py` — async load driver (httpx + asyncio). Sweeps concurrency levels, reports rps + p50/p90/p99/p999.
- `profile_server.py` — wraps either server in `cProfile`, dumps pstats on Ctrl-C.

### Run

```bash
# 1. populate
python -m bench.setup

# 2a. pixeltable server (terminal 1)
pxt serve query --query bench.queries.q1 --path /q1 --inputs i \
    --one-row --method get --host 127.0.0.1 --port 8000

# 2b. postgres baseline server (terminal 2)
uvicorn bench.pg_serve:app --host 127.0.0.1 --port 8001 \
    --workers 1 --log-level warning

# 3. drive load against each (terminal 3)
python -m bench.drive --url http://127.0.0.1:8001/q1 --label pg
python -m bench.drive --url http://127.0.0.1:8000/q1 --label pxt
```

Results land in `bench/results-<label>.csv`.

### First-pass numbers (single uvicorn worker, localhost, smoke test only — 3 concurrency levels)

| label | conc | rps | p50 ms | p99 ms |
|---|---:|---:|---:|---:|
| pg | 1 | 1462 | 0.66 | 1.0 |
| pg | 8 | 1786 | 2.7 | 28.7 |
| pg | 32 | 980 | 20.7 | 178.8 |
| pxt | 1 | 89 | 11.1 | 14.5 |
| pxt | 8 | 256 | 28.4 | 89.7 (**3311 errors**) |
| pxt | 32 | 230 | 126.7 | 311.8 (**4063 errors**) |

The smoke run already shows two big things:
1. **~16x slower at single-thread** (pxt 89 rps vs pg 1462 rps; 11 ms vs 0.66 ms p50).
2. **Heavy error rate under concurrency** for the pxt path (3.3k / 4.0k failures). Likely a connection pool / lock contention issue. Need to inspect the response body / status code distribution before the next pass.

A real run should sweep the full set of concurrency levels (default `[1, 2, 4, 8, 16, 32, 64, 128]`) at 20s per level, and inspect the error responses before drawing conclusions.

### Instrumentation

#### Coarse: cProfile

```bash
python -m bench.profile_server pxt --port 8000     # terminal 1
python -m bench.drive --url http://127.0.0.1:8000/q1 --label pxt --levels 1   # terminal 2
# Ctrl-C the server; pstats lands at bench/profile-pxt.pstats
snakeviz bench/profile-pxt.pstats                  # or:
python -c "import pstats; pstats.Stats('bench/profile-pxt.pstats').sort_stats('cumulative').print_stats(50)"
```

Repeat with `pg` to diff the two profiles. Frames present only in pxt = pixeltable's per-request overhead.

#### Fine: py-spy flamegraph

```bash
# start either server, then:
py-spy record --rate 1000 --duration 30 --output bench/flame-pxt.svg --pid <server-pid>
# drive load during those 30s
```

The SVG flamegraph shows where wall time goes in the running server. Diff against the pg baseline's flamegraph to isolate pixeltable-specific frames.

#### Targets for stage-level instrumentation (next step, once we have a profile)

If the cProfile breakdown isn't conclusive, instrument these stages inside `pixeltable/serving/_fastapi.py` and the query execution path:
1. Request parsing / Pydantic input validation.
2. Query plan construction — verify it's cached per-route, not rebuilt per request.
3. SQL execution (the actual postgres roundtrip — should match the pg baseline).
4. Row fetch + conversion to pixeltable types.
5. Response serialization to JSON.

Per-stage timings can be accumulated in a thread-local and dumped on a timer or via a debug endpoint.

### Open questions surfaced by the smoke test

- Why the high error rate at concurrency ≥ 8 on the pxt path? Connection pool exhaustion, request-level lock, or something in the executor? Need response-body inspection.
- Is the query plan being rebuilt per request? `--levels 1` cProfile of 30s of traffic should make this obvious.
- Does `pxt serve` use a single shared engine/pool, or one per request? Check `_fastapi.py` startup.

## Throughput benchmark

Measures sustained insert + lookup throughput against a hand-rolled FastAPI app that talks to pixeltable directly. Unlike the key-lookup benchmark, there is no SQL baseline -- this suite is intended for tracking absolute numbers across pixeltable changes, not for diffing against postgres.

### Files

- `bench_server.py` — FastAPI app. `lifespan()` (re)creates `bench.throughput` (three required `Int` columns plus two computed columns: `doubled = value * 2`, `offset = value + 100`) and exposes four endpoints on port 8000: `POST /insert`, `GET /lookup?thread_id=&limit=`, `GET /count`, `GET /health`. Handlers are sync `def` so Starlette runs them in a thread pool (no event loop in the request context, which lets `ExecNode.__iter__` take its fast path).
- `bench_driver.py` — async load driver (httpx + asyncio + uvloop). Runs an optional warmup, then any subset of three phases (`--phase insert|lookup|mixed|all`) at a fixed `--concurrency`. Reports per-phase request count, row count, wall time, throughput, and p50/p95/p99 latency. Status-code histogram on errors.

### Run

```bash
# 1. server (terminal 1)
python -m bench.bench_server

# 2. drive load (terminal 2)
python -m bench.bench_driver --concurrency 8 --total-inserts 1000 --batch-size 10 --total-lookups 500

# Run only one phase:
python -m bench.bench_driver --phase insert --total-inserts 5000 --batch-size 50
python -m bench.bench_driver --phase lookup --total-lookups 2000
```

The server keeps the table across runs (lifespan only drops and recreates on startup), so consecutive driver invocations append to the existing table. Restart the server to reset.

### Dependencies

Beyond `pixeltable` itself: `pip install httpx numpy uvloop 'fastapi[standard]'` (the same `fastapi[standard]` extra that `pxt serve` uses).
