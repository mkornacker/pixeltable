import platform
import uuid

import sqlalchemy as sql
from sqlalchemy import Integer, String, Boolean, BigInteger, LargeBinary, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import ForeignKey, UniqueConstraint, ForeignKeyConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Db(Base):
    __tablename__ = 'dbs'

    id = sql.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    name = sql.Column(String, nullable=False)


class Dir(Base):
    __tablename__ = 'dirs'

    id = sql.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    db_id = sql.Column(UUID(as_uuid=True), ForeignKey('dbs.id'), nullable=False)
    path = sql.Column(String, nullable=False)


class Table(Base):
    __tablename__ = 'tables'

    MAX_VERSION = 9223372036854775807  # 2^63 - 1

    id = sql.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    db_id = sql.Column(UUID(as_uuid=True), ForeignKey('dbs.id'), nullable=False)
    dir_id = sql.Column(UUID(as_uuid=True), ForeignKey('dirs.id'), nullable=False)
    name = sql.Column(String, nullable=False)
    parameters = sql.Column(JSON, nullable=False)

    # monotonically increasing w/in Table for both data and schema changes, starting at 0
    current_version = sql.Column(BigInteger, nullable=False)
    # each version has a corresponding schema version (current_version >= current_schema_version)
    current_schema_version = sql.Column(BigInteger, nullable=False)

    # if False, can't apply schema or data changes to this table
    # (table got dropped, but we need to keep a record of it for snapshots)
    is_mutable = sql.Column(Boolean, nullable=False)

    next_col_id = sql.Column(Integer, nullable=False)  # used to assign Column.id

    # - used to assign the rowid column in the storage table
    # - every row is assigned a unique and immutable rowid on insertion
    next_row_id = sql.Column(BigInteger, nullable=False)

    __table_args__ = (
        #ForeignKeyConstraint(
            #['id', 'current_schema_version'], ['tableschemaversions.tbl_id', 'tableschemaversions.schema_version']),
    )

    def storage_name(self) -> str:
        return f'tbl_{self.id.replace("-", "_")}'


# versioning: each table schema change results in a new record
class TableSchemaVersion(Base):
    __tablename__ = 'tableschemaversions'

    tbl_id = sql.Column(UUID(as_uuid=True), ForeignKey('tables.id'), primary_key=True, nullable=False)
    schema_version = sql.Column(BigInteger, primary_key=True, nullable=False)
    preceding_schema_version = sql.Column(BigInteger, nullable=False)


# - records when a column was added/dropped, which is needed to GC unreachable storage columns
#   (a column that was added after table snapshot n and dropped before table snapshot n+1 can be removed
#   from the stored table)
# - one record per column (across all schema versions)
class ColumnHistory(Base):
    __tablename__ = 'columnhistory'

    tbl_id = sql.Column(UUID(as_uuid=True), ForeignKey('tables.id'), primary_key=True, nullable=False)
    # immutable and monotonically increasing from 0 w/in Table
    col_id = sql.Column(Integer, primary_key=True, nullable=False)
    # table schema version when col was added
    schema_version_add = sql.Column(BigInteger, nullable=False)
    # table schema version when col was dropped
    schema_version_drop = sql.Column(BigInteger, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ['tbl_id', 'schema_version_add'], ['tableschemaversions.tbl_id', 'tableschemaversions.schema_version']),
        ForeignKeyConstraint(
            ['tbl_id', 'schema_version_drop'], ['tableschemaversions.tbl_id', 'tableschemaversions.schema_version'])
    )


# - records the logical (user-visible) schema of a table
# - contains the full set of columns for each new schema version: one record per (column x schema version)
class SchemaColumn(Base):
    __tablename__ = 'schemacolumns'

    tbl_id = sql.Column(UUID(as_uuid=True), ForeignKey('tables.id'), primary_key=True, nullable=False)
    schema_version = sql.Column(BigInteger, primary_key=True, nullable=False)
    # immutable and monotonically increasing from 0 w/in Table
    col_id = sql.Column(Integer, primary_key=True, nullable=False)
    pos = sql.Column(Integer, nullable=False)  # position in table, starting at 0
    name = sql.Column(String, nullable=False)
    col_type = sql.Column(String, nullable=False)  # json
    is_pk = sql.Column(Boolean, nullable=False)
    value_expr = sql.Column(String, nullable=True)  # json
    stored = sql.Column(Boolean, nullable=True)
    # if True, creates vector index for this column
    is_indexed = sql.Column(Boolean, nullable=False)

    __table_args__ = (
        UniqueConstraint('tbl_id', 'schema_version', 'pos'),
        UniqueConstraint('tbl_id', 'schema_version', 'name'),
        ForeignKeyConstraint(
            ['tbl_id', 'schema_version'], ['tableschemaversions.tbl_id', 'tableschemaversions.schema_version'])
    )


class TableSnapshot(Base):
    __tablename__ = 'tablesnapshots'

    id = sql.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    db_id = sql.Column(UUID(as_uuid=True), ForeignKey('dbs.id'), nullable=False)
    dir_id = sql.Column(UUID(as_uuid=True), ForeignKey('dirs.id'), nullable=False)
    name = sql.Column(String, nullable=False)
    tbl_id = sql.Column(UUID(as_uuid=True), nullable=False)
    tbl_version = sql.Column(BigInteger, nullable=False)
    tbl_schema_version = sql.Column(BigInteger, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(['tbl_id'], ['tables.id']),
        ForeignKeyConstraint(
            ['tbl_id', 'tbl_schema_version'], ['tableschemaversions.tbl_id', 'tableschemaversions.schema_version']),
    )


class Function(Base):
    """
    User-defined functions that are not library functions (ie, aren't available at runtime as a symbol in a known
    module).
    Functions without a name are anonymous functions used in the definition of a computed column.
    Functions that have names are also assigned to a database and directory.
    We store the Python version under which a Function was created (and the callable pickled) in order to warn
    against version mismatches.
    """
    __tablename__ = 'functions'

    id = sql.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    db_id = sql.Column(UUID(as_uuid=True), ForeignKey('dbs.id'), nullable=True)
    dir_id = sql.Column(UUID(as_uuid=True), ForeignKey('dirs.id'), nullable=True)
    name = sql.Column(String, nullable=True)
    md = sql.Column(JSON, nullable=False)  # Function.md
    eval_obj = sql.Column(LargeBinary, nullable=True)  # Function.eval_fn
    init_obj = sql.Column(LargeBinary, nullable=True)  # Function.init_fn
    update_obj = sql.Column(LargeBinary, nullable=True)  # Function.update_fn
    value_obj = sql.Column(LargeBinary, nullable=True)  # Function.value_fn
    python_version = sql.Column(
        String, nullable=False, default=platform.python_version, onupdate=platform.python_version)
