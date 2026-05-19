"""
FastAPI + Pixeltable throughput benchmark server.

Endpoints are sync `def` (not `async def`) so that Starlette runs them in a
thread pool where there is no running event loop. This lets ExecNode.__iter__
take the fast path (loop.run_until_complete) instead of the slower
_iter_via_thread fallback.

Usage:
    pip install fastapi uvicorn uvloop
    python -m bench.bench_server
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

import pixeltable as pxt

logger = logging.getLogger('bench_server')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

DIR = 'bench'
TABLE_PATH = 'bench.throughput'


# --- request / response models ---

class InsertRequest(BaseModel):
    rows: list[dict[str, Any]]


class InsertResponse(BaseModel):
    num_rows: int
    num_excs: int
    elapsed_ms: float


class LookupResponse(BaseModel):
    rows: list[dict[str, Any]]
    count: int
    elapsed_ms: float


class CountResponse(BaseModel):
    count: int
    elapsed_ms: float


# --- lifespan: create table on startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Setting up benchmark table ...')
    pxt.drop_dir(DIR, if_not_exists='ignore', force=True)
    pxt.create_dir(DIR, if_exists='ignore')
    t = pxt.create_table(
        TABLE_PATH,
        {
            'thread_id': pxt.Required[pxt.Int],
            'row_idx': pxt.Required[pxt.Int],
            'value': pxt.Required[pxt.Int],
        },
    )
    t.add_computed_column(doubled=t.value * 2)
    t.add_computed_column(offset=t.value + 100)
    logger.info('Table %s ready with computed columns [doubled, offset]', TABLE_PATH)
    yield
    logger.info('Shutting down.')


app = FastAPI(title='Pixeltable Benchmark', lifespan=lifespan)


# --- endpoints (all sync def) ---

@app.post('/insert', response_model=InsertResponse)
def insert(req: InsertRequest):
    t = pxt.get_table(TABLE_PATH)
    start = time.perf_counter()
    status = t.insert(req.rows)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return InsertResponse(
        num_rows=status.num_rows,
        num_excs=status.num_excs,
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.get('/lookup', response_model=LookupResponse)
def lookup(thread_id: int = Query(...), limit: int = Query(default=50)):
    t = pxt.get_table(TABLE_PATH)
    start = time.perf_counter()
    result = t.select(t.thread_id, t.row_idx, t.value, t.doubled, t.offset).where(t.thread_id == thread_id).limit(limit).collect()
    elapsed_ms = (time.perf_counter() - start) * 1000
    rows = list(result)
    return LookupResponse(
        rows=rows,
        count=len(rows),
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.get('/count', response_model=CountResponse)
def count():
    t = pxt.get_table(TABLE_PATH)
    start = time.perf_counter()
    n = t.count()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return CountResponse(count=n, elapsed_ms=round(elapsed_ms, 2))


@app.get('/health')
def health():
    return {'status': 'ok'}


if __name__ == '__main__':
    uvicorn.run('bench.bench_server:app', host='127.0.0.1', port=8000, loop='uvloop', workers=1)
