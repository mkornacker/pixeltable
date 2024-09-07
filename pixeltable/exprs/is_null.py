from __future__ import annotations

from typing import Optional, List, Dict

import sqlalchemy as sql

import pixeltable.type_system as ts
from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class IsNull(Expr):
    def __init__(self, e: Expr):
        super().__init__(ts.BoolType())
        self.components = [e]
        self.id = self._create_id()

    def __str__(self) -> str:
        return f'{str(self.components[0])} == None'

    def _equals(self, other: IsNull) -> bool:
        return True

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ClauseElement]:
        e = sql_elements[self.components[0]]
        if e is None:
            return None
        return e == None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row[self.components[0].slot_idx] is None

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert len(components) == 1
        return cls(components[0])

