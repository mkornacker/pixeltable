"""(Re)create bench.lookup, load 100k rows, and write bench/pg_config.json
with the postgres connection details so the SQLAlchemy baseline server can
reach the same physical table without going through pixeltable."""

import json
from pathlib import Path

import pixeltable as pxt
from pixeltable.env import Env

N_ROWS = 100_000


def main() -> None:
    pxt.drop_dir('bench', force=True, if_not_exists='ignore')
    pxt.create_dir('bench')
    t = pxt.create_table('bench.lookup', {'id': pxt.Required[pxt.Int]}, primary_key='id')
    t.add_computed_column(doubled=t.id * 2)
    t.insert({'id': i} for i in range(N_ROWS))
    assert t.count() == N_ROWS

    tv = t._tbl_version_path.tbl_version.get()
    sa_tbl = tv.store_tbl.sa_tbl
    cfg = {
        'db_url': Env.get().db_url,
        'table': sa_tbl.name,
        'id_column': tv.cols_by_name['id'].store_name(),
        'doubled_column': tv.cols_by_name['doubled'].store_name(),
    }
    out = Path(__file__).with_name('pg_config.json')
    out.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == '__main__':
    main()
