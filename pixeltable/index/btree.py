from typing import Optional

import sqlalchemy as sql

import pixeltable.exceptions as excs
from pixeltable import catalog, exprs
from pixeltable.func.udf import udf

from .base import IndexBase


class BtreeIndex(IndexBase):
    """
    Interface to B-tree indices in Postgres.
    """
    MAX_STRING_LEN = 256

    def __init__(self, c: 'catalog.Column'):
        if not c.col_type.is_scalar_type() and not c.col_type.is_media_type():
            raise excs.Error(f'Index on column {c.name}: B-tree index requires scalar or media type, got {c.col_type}')
        self.value_expr = _str_filter(exprs.ColumnRef(c)) if c.col_type.is_string_type() else exprs.ColumnRef(c)

    def index_value_expr(self) -> 'exprs.Expr':
        return self.value_expr

    def records_value_errors(self) -> bool:
        return False

    def index_sa_type(self) -> sql.types.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        return self.value_expr.col_type.to_sa_type()

    def create_index(self, index_name: str, index_value_col: 'catalog.Column', conn: sql.engine.Connection) -> None:
        """Create the index on the index value column"""
        idx = sql.Index(index_name, index_value_col.sa_col, postgresql_using='btree')
        idx.create(bind=conn)

    @classmethod
    def display_name(cls) -> str:
        return 'btree'

    def as_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, c: 'catalog.Column', d: dict) -> 'BtreeIndex':
        return cls(c)


@udf
def _str_filter(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s[:BtreeIndex.MAX_STRING_LEN]
