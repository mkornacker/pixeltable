"""SQLAlchemy-direct baseline. Same physical table as bench.lookup, same lookup
predicate as pixeltable's query path (id == :i AND v_max == MAX_INT), but
without going through any pixeltable code.

Run:
    uvicorn bench.pg_serve:app --host 127.0.0.1 --port 8001 --workers 1 --log-level warning
"""

import json
from pathlib import Path

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException

V_MAX_SENTINEL = 9223372036854775807

cfg = json.loads(Path(__file__).with_name('pg_config.json').read_text())
engine = sa.create_engine(cfg['db_url'], pool_size=32, max_overflow=0, pool_pre_ping=False, future=True)
stmt = sa.text(
    f'SELECT {cfg["doubled_column"]} AS doubled '
    f'FROM {cfg["table"]} '
    f'WHERE {cfg["id_column"]} = :i AND v_max = {V_MAX_SENTINEL}'
)

app = FastAPI()


@app.get('/q1')
def q1(i: int) -> dict[str, int]:
    with engine.connect() as conn:
        row = conn.execute(stmt, {'i': i}).first()
    if row is None:
        raise HTTPException(404)
    return {'doubled': row.doubled}
