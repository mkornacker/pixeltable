"""More realistic SQLAlchemy + FastAPI baseline. Same physical table and lookup
predicate as pixeltable's q1, but adds the framework overhead a real app would
typically have on top of the bare-bones pg_serve.py:

- Pydantic response model (output validation via FastAPI's response_model).
- Pydantic input validation (already implicit via the typed query param).
- Transaction wrap around the read (`with conn.begin():`).
- ORM Session created per request (most apps split between Session and raw
  Connection; we do the heavier of the two so the baseline is comparable).
- Sync `def` endpoint dispatched via Starlette's threadpool (matches pixeltable's
  FastAPIRouter handler shape).

Run:
    uvicorn bench.pg_realistic_serve:app --host 127.0.0.1 --port 8002 \\
        --workers 1 --log-level warning
"""

import json
from pathlib import Path

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import orm

V_MAX_SENTINEL = 9223372036854775807

cfg = json.loads(Path(__file__).with_name('pg_config.json').read_text())
engine = sa.create_engine(cfg['db_url'], pool_size=32, max_overflow=0, pool_pre_ping=False, future=True)
stmt = sa.text(
    f'SELECT {cfg["doubled_column"]} AS doubled '
    f'FROM {cfg["table"]} '
    f'WHERE {cfg["id_column"]} = :i AND v_max = {V_MAX_SENTINEL}'
)


class Q1Response(BaseModel):
    doubled: int


app = FastAPI()


@app.get('/q1', response_model=Q1Response)
def q1(i: int) -> Q1Response:
    with engine.connect() as conn, orm.Session(conn) as _session, conn.begin():
        row = conn.execute(stmt, {'i': i}).first()
    if row is None:
        raise HTTPException(404)
    return Q1Response(doubled=row.doubled)
