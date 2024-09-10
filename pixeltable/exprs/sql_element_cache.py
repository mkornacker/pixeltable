from typing import Iterable, Union, Optional

import sqlalchemy as sql

from .expr import Expr


class SqlElementCache:
    """Cache of sql.ColumnElements for exprs"""

    cache: dict[int, Optional[sql.ColumnElement]]  # key: Expr.id

    def __init__(self):
        self.cache = {}

    def __getitem__(self, e: Expr) -> Optional[sql.ColumnElement]:
        try:
            return self.cache[e.id]
        except KeyError:
            pass
        el = e.sql_expr(self)
        self.cache[e.id] = el
        return el

    def contains(self, items: Union[Expr, Iterable[Expr]]) -> bool:
        if isinstance(items, Expr):
            return self[items] is not None
        return all(self[e] is not None for e in items)
