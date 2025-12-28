import datetime
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL.Image
import pyarrow as pa
import pytest
import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import JSONB

import pixeltable as pxt
from pixeltable.env import Env

from ..utils import skip_test_if_not_installed


@pxt.udf
def udf_with_exc(i: int, val: int) -> int:
    if i == val:
        raise ValueError(f'Error for row {i}')
    return i


class TestLanceDb:
    def test_x(self, reset_db: None, tmp_path: Path) -> None:
        skip_test_if_not_installed('lancedb')
        import lancedb  # type: ignore[import-untyped]

        n_rows = 100_000
        schema = {
            'row_id': pxt.Int,
            'c_int': pxt.Int,
            'c_float': pxt.Float,
            'c_bool': pxt.Bool,
            'c_string': pxt.String,
            'c_timestamp': pxt.Timestamp,
            'c_date': pxt.Date,
            'c_json': pxt.Json,
            #'c_array': pxt.Array[(10,), pxt.Float],  # type: ignore[misc]
            #'c_image': pxt.Image,
        }
        t = pxt.create_table('test_export', schema)

        rows = [
            {
                'row_id': i,
                'c_int': i + 1 if i % 10 != 0 else None,
                'c_float': i * 10.0,
                'c_bool': bool(i % 2),
                'c_string': f'string_{i}',
                'c_timestamp': datetime.datetime.now() - datetime.timedelta(seconds=i),
                'c_date': datetime.date.today() - datetime.timedelta(days=i),
                'c_json': {'key': i, 'value': f'val_{i}', 'nested': {'data': i * 2}},
                #'c_array': np.array([i] * 10, dtype=np.float32),
                #'c_image': PIL.Image.new('RGB', (100, 100), color=(i % 256, (i * 2) % 256, (i * 3) % 256)),
            }
            for i in range(n_rows)
        ]
        t.insert(rows)

    def test_y(self, reset_db: None) -> None:
        """Baseline benchmark: insert same data using SQLAlchemy directly.

        Mimics Pixeltable's table structure including:
        - System columns: rowid, v_min, v_max
        - Composite btree index on system columns (rowid, v_min)
        - BRIN indices on v_min and v_max (as Pixeltable uses for PostgreSQL)
        - Btree indices on scalar user columns (Pixeltable's default behavior)
        """
        n_rows = 100_000
        max_version = 9223372036854775807  # Pixeltable's MAX_VERSION

        # Create SQLAlchemy table definition with system columns
        metadata = sql.MetaData()
        test_table = sql.Table(
            'test_sqlalchemy_insert',
            metadata,
            # System columns (like Pixeltable)
            sql.Column('rowid', sql.BigInteger, nullable=False),
            sql.Column('v_min', sql.BigInteger, nullable=False),
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(max_version)),
            # User columns
            sql.Column('row_id', sql.BigInteger),
            sql.Column('c_int', sql.BigInteger),
            sql.Column('c_float', sql.Float),
            sql.Column('c_bool', sql.Boolean),
            sql.Column('c_string', sql.String),
            sql.Column('c_timestamp', sql.TIMESTAMP(timezone=True)),
            sql.Column('c_date', sql.Date),
            sql.Column('c_json', JSONB),
            # Primary key on rowid + v_min (like Pixeltable)
            sql.PrimaryKeyConstraint('rowid', 'v_min'),
            # Composite btree index on system columns (speeds up joins and ORDER BY)
            sql.Index('sys_cols_idx', 'rowid', 'v_min', 'v_max'),
            # BRIN indices on v_min and v_max (like Pixeltable uses for PostgreSQL)
            sql.Index('vmin_idx', 'v_min', postgresql_using='brin'),
            sql.Index('vmax_idx', 'v_max', postgresql_using='brin'),
            # Btree indices on scalar user columns (Pixeltable creates these by default)
            # Note: c_bool is not indexed (btrees on bools aren't useful)
            # Note: c_json is not indexed (not a scalar type)
            sql.Index('idx_row_id', 'row_id', postgresql_using='btree'),
            sql.Index('idx_c_int', 'c_int', postgresql_using='btree'),
            sql.Index('idx_c_float', 'c_float', postgresql_using='btree'),
            sql.Index('idx_c_string', 'c_string', postgresql_using='btree'),
            sql.Index('idx_c_timestamp', 'c_timestamp', postgresql_using='btree'),
            sql.Index('idx_c_date', 'c_date', postgresql_using='btree'),
        )

        engine = Env.get().engine

        # Drop table if exists and create new one
        test_table.drop(engine, checkfirst=True)
        test_table.create(engine)

        # Generate same row data as test_x, plus system column values
        now = datetime.datetime.now()
        today = datetime.date.today()
        v_min = 1  # Version at which rows are created
        rows = [
            {
                # System columns
                'rowid': i,
                'v_min': v_min,
                'v_max': max_version,
                # User columns (same as test_x)
                'row_id': i,
                'c_int': i + 1 if i % 10 != 0 else None,
                'c_float': i * 10.0,
                'c_bool': bool(i % 2),
                'c_string': f'string_{i}',
                'c_timestamp': now - datetime.timedelta(seconds=i),
                'c_date': today - datetime.timedelta(days=i),
                'c_json': {'key': i, 'value': f'val_{i}', 'nested': {'data': i * 2}},
            }
            for i in range(n_rows)
        ]

        # Insert using SQLAlchemy
        start = time.perf_counter()
        with engine.begin() as conn:
            conn.execute(sql.insert(test_table), rows)
        elapsed = time.perf_counter() - start

        print(f'\nSQLAlchemy direct insert: {n_rows} rows in {elapsed:.2f}s ({n_rows/elapsed:.2f} rows/s)')

        # Clean up
        test_table.drop(engine, checkfirst=True)

    def test_w(self, reset_db: None) -> None:
        """Baseline benchmark: ADBC with staging table for JSONB support.

        Uses ADBC's fast COPY-based adbc_ingest() into a staging table (TEXT for JSON),
        then copies to target table with JSONB conversion via INSERT...SELECT.

        Performance findings (1M rows, 140 MB):

        Batch Size |  Total (s) |  Ingest (s) |   Copy (s) |       Rows/s
        ------------------------------------------------------------------------
            10,000 |       6.63 |        0.99 |       5.63 |      150,893
            25,000 |       6.37 |        0.87 |       5.50 |      156,960
            50,000 |       6.38 |        0.87 |       5.50 |      156,846
           100,000 |       6.38 |        0.89 |       5.49 |      156,674
           250,000 |       6.39 |        0.92 |       5.47 |      156,525
           500,000 |       6.43 |        0.93 |       5.50 |      155,606
         1,000,000 |       6.41 |        0.93 |       5.48 |      156,101

        Key observations:
        - ADBC ingest is very fast: ~0.9s for 1M rows (~1.1M rows/s)
        - The bottleneck is INSERT...SELECT with JSONB cast: ~5.5s (85% of total time)
        - Batch size has minimal impact on performance
        """
        import adbc_driver_postgresql.dbapi

        n_rows = 100_000
        max_version = 9223372036854775807  # Pixeltable's MAX_VERSION

        # Target table with JSONB column (same as test_y)
        metadata = sql.MetaData()
        table_name = 'test_adbc_target'
        staging_name = 'test_adbc_staging'

        target_table = sql.Table(
            table_name,
            metadata,
            sql.Column('rowid', sql.BigInteger, nullable=False),
            sql.Column('v_min', sql.BigInteger, nullable=False),
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(max_version)),
            sql.Column('row_id', sql.BigInteger),
            sql.Column('c_int', sql.BigInteger),
            sql.Column('c_float', sql.Float),
            sql.Column('c_bool', sql.Boolean),
            sql.Column('c_string', sql.String),
            sql.Column('c_timestamp', sql.TIMESTAMP(timezone=True)),
            sql.Column('c_date', sql.Date),
            sql.Column('c_json', JSONB),
            sql.PrimaryKeyConstraint('rowid', 'v_min'),
            sql.Index('sys_cols_idx_w', 'rowid', 'v_min', 'v_max'),
            sql.Index('vmin_idx_w', 'v_min', postgresql_using='brin'),
            sql.Index('vmax_idx_w', 'v_max', postgresql_using='brin'),
            sql.Index('idx_row_id_w', 'row_id', postgresql_using='btree'),
            sql.Index('idx_c_int_w', 'c_int', postgresql_using='btree'),
            sql.Index('idx_c_float_w', 'c_float', postgresql_using='btree'),
            sql.Index('idx_c_string_w', 'c_string', postgresql_using='btree'),
            sql.Index('idx_c_timestamp_w', 'c_timestamp', postgresql_using='btree'),
            sql.Index('idx_c_date_w', 'c_date', postgresql_using='btree'),
        )

        # Staging table with TEXT for JSON (no indices needed)
        staging_table = sql.Table(
            staging_name,
            metadata,
            sql.Column('rowid', sql.BigInteger, nullable=False),
            sql.Column('v_min', sql.BigInteger, nullable=False),
            sql.Column('v_max', sql.BigInteger, nullable=False),
            sql.Column('row_id', sql.BigInteger),
            sql.Column('c_int', sql.BigInteger),
            sql.Column('c_float', sql.Float),
            sql.Column('c_bool', sql.Boolean),
            sql.Column('c_string', sql.String),
            sql.Column('c_timestamp', sql.TIMESTAMP(timezone=True)),
            sql.Column('c_date', sql.Date),
            sql.Column('c_json', sql.Text),  # TEXT for ADBC COPY compatibility
        )

        engine = Env.get().engine

        # Create tables
        target_table.drop(engine, checkfirst=True)
        staging_table.drop(engine, checkfirst=True)
        target_table.create(engine)
        staging_table.create(engine)

        # Generate data as columnar arrays for PyArrow
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        today = datetime.date.today()
        v_min = 1

        rowids = list(range(n_rows))
        v_mins = [v_min] * n_rows
        v_maxs = [max_version] * n_rows
        row_ids = list(range(n_rows))
        c_ints = [i + 1 if i % 10 != 0 else None for i in range(n_rows)]
        c_floats = [i * 10.0 for i in range(n_rows)]
        c_bools = [bool(i % 2) for i in range(n_rows)]
        c_strings = [f'string_{i}' for i in range(n_rows)]
        c_timestamps = [now - datetime.timedelta(seconds=i) for i in range(n_rows)]
        c_dates = [today - datetime.timedelta(days=i) for i in range(n_rows)]
        c_jsons = [json.dumps({'key': i, 'value': f'val_{i}', 'nested': {'data': i * 2}}) for i in range(n_rows)]

        arrow_table = pa.table({
            'rowid': pa.array(rowids, type=pa.int64()),
            'v_min': pa.array(v_mins, type=pa.int64()),
            'v_max': pa.array(v_maxs, type=pa.int64()),
            'row_id': pa.array(row_ids, type=pa.int64()),
            'c_int': pa.array(c_ints, type=pa.int64()),
            'c_float': pa.array(c_floats, type=pa.float64()),
            'c_bool': pa.array(c_bools, type=pa.bool_()),
            'c_string': pa.array(c_strings, type=pa.string()),
            'c_timestamp': pa.array(c_timestamps, type=pa.timestamp('us', tz='UTC')),
            'c_date': pa.array(c_dates, type=pa.date32()),
            'c_json': pa.array(c_jsons, type=pa.string()),  # TEXT for staging
        })

        # Build ADBC connection URI
        url = engine.url
        if url.query.get('host'):
            import urllib.parse
            host = urllib.parse.unquote(url.query['host'])
            uri = f'postgresql://{url.username}@/{url.database}?host={host}'
        else:
            uri = f'postgresql://{url.username}@{url.host}:{url.port}/{url.database}'

        # Insert using ADBC into staging table, then copy to target with JSONB cast
        start = time.perf_counter()

        # Step 1: Fast COPY into staging table via ADBC
        with adbc_driver_postgresql.dbapi.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(staging_name, arrow_table, mode='append')
            conn.commit()

        # Step 2: Copy from staging to target with JSONB conversion
        with engine.begin() as conn:
            cols = ', '.join(c.name for c in staging_table.c if c.name != 'c_json')
            conn.execute(sql.text(
                f'INSERT INTO {table_name} ({cols}, c_json) '
                f'SELECT {cols}, c_json::jsonb FROM {staging_name}'
            ))

        elapsed = time.perf_counter() - start

        print(f'\nADBC staging + copy: {n_rows} rows in {elapsed:.2f}s ({n_rows/elapsed:.2f} rows/s)')

        # Clean up
        staging_table.drop(engine, checkfirst=True)
        target_table.drop(engine, checkfirst=True)

    def test_z(self, reset_db: None) -> None:
        """Baseline benchmark: insert using ADBC (Arrow Database Connectivity).

        Uses PyArrow tables for efficient bulk ingestion via ADBC protocol.
        Table structure matches test_y() for fair comparison.
        """
        import adbc_driver_postgresql.dbapi

        n_rows = 1000_000
        max_version = 9223372036854775807  # Pixeltable's MAX_VERSION

        # Create SQLAlchemy table definition with system columns (same as test_y)
        metadata = sql.MetaData()
        table_name = 'test_adbc_insert'
        test_table = sql.Table(
            table_name,
            metadata,
            # System columns (like Pixeltable)
            sql.Column('rowid', sql.BigInteger, nullable=False),
            sql.Column('v_min', sql.BigInteger, nullable=False),
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(max_version)),
            # User columns
            sql.Column('row_id', sql.BigInteger),
            sql.Column('c_int', sql.BigInteger),
            sql.Column('c_float', sql.Float),
            sql.Column('c_bool', sql.Boolean),
            sql.Column('c_string', sql.String),
            sql.Column('c_timestamp', sql.TIMESTAMP(timezone=True)),
            sql.Column('c_date', sql.Date),
            sql.Column('c_json', JSONB),
            # Primary key on rowid + v_min (like Pixeltable)
            sql.PrimaryKeyConstraint('rowid', 'v_min'),
            # Composite btree index on system columns
            sql.Index('sys_cols_idx_z', 'rowid', 'v_min', 'v_max'),
            # BRIN indices on v_min and v_max
            sql.Index('vmin_idx_z', 'v_min', postgresql_using='brin'),
            sql.Index('vmax_idx_z', 'v_max', postgresql_using='brin'),
            # Btree indices on scalar user columns
            sql.Index('idx_row_id_z', 'row_id', postgresql_using='btree'),
            sql.Index('idx_c_int_z', 'c_int', postgresql_using='btree'),
            sql.Index('idx_c_float_z', 'c_float', postgresql_using='btree'),
            sql.Index('idx_c_string_z', 'c_string', postgresql_using='btree'),
            sql.Index('idx_c_timestamp_z', 'c_timestamp', postgresql_using='btree'),
            sql.Index('idx_c_date_z', 'c_date', postgresql_using='btree'),
        )

        engine = Env.get().engine

        # Drop table if exists and create new one with indices
        test_table.drop(engine, checkfirst=True)
        test_table.create(engine)

        # Generate data as columnar arrays for PyArrow
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        today = datetime.date.today()
        v_min = 1

        # Build column arrays
        rowids = list(range(n_rows))
        v_mins = [v_min] * n_rows
        v_maxs = [max_version] * n_rows
        row_ids = list(range(n_rows))
        c_ints = [i + 1 if i % 10 != 0 else None for i in range(n_rows)]
        c_floats = [i * 10.0 for i in range(n_rows)]
        c_bools = [bool(i % 2) for i in range(n_rows)]
        c_strings = [f'string_{i}' for i in range(n_rows)]
        c_timestamps = [now - datetime.timedelta(seconds=i) for i in range(n_rows)]
        c_dates = [today - datetime.timedelta(days=i) for i in range(n_rows)]
        c_jsons = [json.dumps({'key': i, 'value': f'val_{i}', 'nested': {'data': i * 2}}) for i in range(n_rows)]

        # Create PyArrow table
        arrow_table = pa.table({
            'rowid': pa.array(rowids, type=pa.int64()),
            'v_min': pa.array(v_mins, type=pa.int64()),
            'v_max': pa.array(v_maxs, type=pa.int64()),
            'row_id': pa.array(row_ids, type=pa.int64()),
            'c_int': pa.array(c_ints, type=pa.int64()),
            'c_float': pa.array(c_floats, type=pa.float64()),
            'c_bool': pa.array(c_bools, type=pa.bool_()),
            'c_string': pa.array(c_strings, type=pa.string()),
            'c_timestamp': pa.array(c_timestamps, type=pa.timestamp('us', tz='UTC')),
            'c_date': pa.array(c_dates, type=pa.date32()),
            'c_json': pa.array(c_jsons, type=pa.json_(pa.utf8())),
        })

        # Build ADBC connection URI from engine URL components
        # ADBC doesn't support all SQLAlchemy URL options (like timezone)
        url = engine.url
        # For Unix socket connections, use host parameter
        if url.query.get('host'):
            import urllib.parse
            host = urllib.parse.unquote(url.query['host'])
            uri = f'postgresql://{url.username}@/{url.database}?host={host}'
        else:
            uri = f'postgresql://{url.username}@{url.host}:{url.port}/{url.database}'

        # Insert using ADBC with executemany (adbc_ingest uses COPY which doesn't handle JSONB)
        # Rename columns to $1, $2, ... for PostgreSQL bind parameter syntax
        col_names = arrow_table.column_names
        bind_batch = pa.record_batch(
            [arrow_table.column(col).combine_chunks() for col in col_names],
            names=[f'${i+1}' for i in range(len(col_names))]
        )

        # Build INSERT statement with explicit cast for JSONB column
        placeholders = []
        for i, col in enumerate(col_names):
            if col == 'c_json':
                placeholders.append(f'${i+1}::jsonb')
            else:
                placeholders.append(f'${i+1}')
        insert_sql = f"INSERT INTO {table_name} ({', '.join(col_names)}) VALUES ({', '.join(placeholders)})"

        start = time.perf_counter()
        with adbc_driver_postgresql.dbapi.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, bind_batch)
            conn.commit()
        elapsed = time.perf_counter() - start

        print(f'\nADBC direct insert: {n_rows} rows in {elapsed:.2f}s ({n_rows/elapsed:.2f} rows/s)')

        # Clean up
        test_table.drop(engine, checkfirst=True)

    def test_export(self, reset_db: None, tmp_path: Path) -> None:
        skip_test_if_not_installed('lancedb')
        import lancedb  # type: ignore[import-untyped]

        n_rows = 1000
        schema = {
            'row_id': pxt.Int,
            'c_int': pxt.Int,
            'c_float': pxt.Float,
            'c_bool': pxt.Bool,
            'c_string': pxt.String,
            'c_timestamp': pxt.Timestamp,
            'c_date': pxt.Date,
            'c_json': pxt.Json,
            'c_array': pxt.Array[(10,), pxt.Float],  # type: ignore[misc]
            'c_image': pxt.Image,
        }
        t = pxt.create_table('test_export', schema)

        rows = [
            {
                'row_id': i,
                'c_int': i + 1 if i % 10 != 0 else None,
                'c_float': i * 10.0,
                'c_bool': bool(i % 2),
                'c_string': f'string_{i}',
                'c_timestamp': datetime.datetime.now() - datetime.timedelta(seconds=i),
                'c_date': datetime.date.today() - datetime.timedelta(days=i),
                'c_json': {'key': i, 'value': f'val_{i}', 'nested': {'data': i * 2}},
                'c_array': np.array([i] * 10, dtype=np.float32),
                'c_image': PIL.Image.new('RGB', (100, 100), color=(i % 256, (i * 2) % 256, (i * 3) % 256)),
            }
            for i in range(n_rows)
        ]
        t.insert(rows)

        db_path = tmp_path / 'test_lancedb'

        def validate_data(lance_table_name: str, rows: list[dict[str, Any]]) -> None:
            db = lancedb.connect(str(db_path))
            lance_tbl = db.open_table(lance_table_name)
            lance_df = lance_tbl.to_pandas()
            assert len(lance_df) == len(rows)
            assert lance_df['row_id'].tolist() == [row['row_id'] for row in rows]
            assert [None if pd.isna(i) else i for i in lance_df['c_int'].tolist()] == [row['c_int'] for row in rows]
            assert lance_df['c_float'].tolist() == [row['c_float'] for row in rows]
            assert lance_df['c_bool'].tolist() == [row['c_bool'] for row in rows]
            assert lance_df['c_string'].tolist() == [row['c_string'] for row in rows]
            assert lance_df['c_timestamp'].tolist() == [row['c_timestamp'] for row in rows]
            assert lance_df['c_date'].tolist() == [row['c_date'] for row in rows]
            assert lance_df['c_json'].tolist() == [row['c_json'] for row in rows]
            all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(lance_df['c_array'], [r['c_array'] for r in rows]))
            for lance_img_bytes, row in zip(lance_df['c_image'], rows):
                lance_img = PIL.Image.open(io.BytesIO(lance_img_bytes))
                original_img = row['c_image']
                assert lance_img.size == original_img.size
                assert lance_img.mode == original_img.mode

        pxt.io.export_lancedb(t, db_path, 'test')
        validate_data('test', list(t.collect()))

        with pytest.raises(pxt.Error, match='already exists in'):
            pxt.io.export_lancedb(t, db_path, 'test', if_exists='error')

        with pytest.raises(pxt.Error, match='must be one of'):
            pxt.io.export_lancedb(t, db_path, 'test', if_exists='badval')  # type: ignore[arg-type]

        with pytest.raises(pxt.Error, match='exists and is not a directory'):
            pxt.io.export_lancedb(t, Path(__file__), 'test', if_exists='overwrite')

        # export query result containing PIL image, with if_exists='overwrite'
        t2 = pxt.create_table('test2', schema)
        t2.insert(rows[:100])
        query = t2.order_by(t2.row_id, asc=False).select(
            t2.row_id,
            t2.c_int,
            t2.c_float,
            t2.c_bool,
            t2.c_string,
            t2.c_timestamp,
            t2.c_date,
            t2.c_json,
            t2.c_array,
            c_image=t2.c_image.rotate(180),
        )
        pxt.io.export_lancedb(query, db_path, 'test', if_exists='overwrite')
        validate_data('test', list(query.collect()))

        # if_exists='append'
        pxt.io.export_lancedb(t, db_path, 'test', if_exists='append', batch_size_bytes=1024)
        validate_data('test', list(query.collect()) + list(t.collect()))

        # error during export
        error_db_path = tmp_path / 'error_db'
        with pytest.raises(pxt.Error):
            pxt.io.export_lancedb(t.select(t.c_int, udf_with_exc(t.c_int, 100)), error_db_path, 'test')
        assert not error_db_path.exists()
