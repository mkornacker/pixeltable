import abc
import platform
from typing import TYPE_CHECKING

import sqlalchemy as sql

if TYPE_CHECKING:
    from pixeltable.runtime import IsolationLevel, XactMode

# Engine-level default isolation, used at bootstrap and as a fallback. Per-xact isolation is set by
# Runtime.begin_db_xact() via Dbms.isolation_level(mode). SERIALIZABLE is the safe default for backends
# that don't support REPEATABLE READ (e.g. older CockroachDB).
_DEFAULT_ISOLATION_LEVEL = 'SERIALIZABLE'


class Dbms(abc.ABC):
    """
    Provides abstractions for utilities to interact with a database system.
    """

    name: str
    transaction_isolation_level: str
    version_index_type: str
    db_url: sql.URL

    def __init__(self, name: str, transaction_isolation_level: str, version_index_type: str, db_url: sql.URL) -> None:
        self.name = name
        self.transaction_isolation_level = transaction_isolation_level
        self.version_index_type = version_index_type
        self.db_url = db_url

    def isolation_level(self, mode: 'XactMode') -> 'IsolationLevel':
        """Maps a logical transaction mode to a backend-specific isolation level."""
        from pixeltable.runtime import IsolationLevel

        return IsolationLevel(self.transaction_isolation_level)

    @abc.abstractmethod
    def drop_db_stmt(self, database: str) -> str: ...

    @abc.abstractmethod
    def create_db_stmt(self, database: str) -> str: ...

    @abc.abstractmethod
    def default_system_db_url(self) -> str: ...

    @abc.abstractmethod
    def create_vector_index_stmt(
        self, store_index_name: str, sa_value_col: sql.Column, metric: str
    ) -> sql.Compiled: ...


class PostgresqlDbms(Dbms):
    """
    Implements utilities to interact with Postgres database.
    """

    def __init__(self, db_url: sql.URL):
        super().__init__('postgresql', 'READ COMMITTED', 'brin', db_url)

    def isolation_level(self, mode: 'XactMode') -> 'IsolationLevel':
        from pixeltable.runtime import IsolationLevel, XactMode

        # MD_ACCESS wants snapshot isolation so multi-statement metadata reads are consistent.
        # WRITE_* and QUERY stay at READ COMMITTED.
        return IsolationLevel.REPEATABLE_READ if mode is XactMode.MD_ACCESS else IsolationLevel.READ_COMMITTED

    def drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database}'

    def create_db_stmt(self, database: str) -> str:
        match platform.system():
            case 'Windows':
                lc_ctype = '.UTF-8'
            case 'Darwin':
                lc_ctype = 'en_US.UTF-8'
            case _:
                lc_ctype = 'C.UTF-8'
        return f"CREATE DATABASE {database} TEMPLATE template0 ENCODING 'UTF8' LC_COLLATE 'C' LC_CTYPE '{lc_ctype}'"

    def default_system_db_url(self) -> str:
        a = self.db_url.set(database='postgres').render_as_string(hide_password=False)
        return a

    def create_vector_index_stmt(self, store_index_name: str, sa_value_col: sql.Column, metric: str) -> sql.Compiled:
        from sqlalchemy.dialects import postgresql

        sa_idx = sql.Index(
            store_index_name,
            sa_value_col,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={sa_value_col.name: metric},
        )
        return sql.schema.CreateIndex(sa_idx, if_not_exists=True).compile(dialect=postgresql.dialect())


class CockroachDbms(Dbms):
    """
    Implements utilities to interact with CockroachDb database.
    """

    def __init__(self, db_url: sql.URL):
        super().__init__('cockroachdb', _DEFAULT_ISOLATION_LEVEL, 'btree', db_url)

    def drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database} CASCADE'

    def create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} TEMPLATE template0 ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C'"

    def default_system_db_url(self) -> str:
        return self.db_url.set(database='defaultdb').render_as_string(hide_password=False)

    def sa_vector_index(self, store_index_name: str, sa_value_col: sql.schema.Column, metric: str) -> sql.Index | None:
        return None

    def create_vector_index_stmt(self, store_index_name: str, sa_value_col: sql.Column, metric: str) -> sql.Compiled:
        return sql.text(
            f'CREATE VECTOR INDEX IF NOT EXISTS {store_index_name} ON {sa_value_col.table.name}'
            f'({sa_value_col.name} {metric})'
        ).compile()
