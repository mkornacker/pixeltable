from __future__ import annotations

import copy
from typing import Optional, List, Any, Dict, Tuple

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder


class InlineDict(Expr):
    """
    Dictionary 'literal' which can use Exprs as values.
    """
    def __init__(self, d: Dict):
        from .inline_array import InlineArray
        super().__init__(ts.JsonType())  # we need to call this in order to populate self.components
        # dict_items contains
        # - for Expr fields: (key, index into components, None)
        # - for non-Expr fields: (key, None, value)
        self.dict_items: List[Tuple[str, Optional[int], Any]] = []
        for key, val in d.items():
            if not isinstance(key, str):
                raise excs.Error(f'Dictionary requires string keys, {key} has type {type(key)}')
            val = copy.deepcopy(val)
            if isinstance(val, dict):
                val = InlineDict(val)
            if isinstance(val, list) or isinstance(val, tuple):
                val = InlineArray(tuple(val), force_json=True)
            if isinstance(val, Expr):
                self.dict_items.append((key, len(self.components), None))
                self.components.append(val)
            else:
                self.dict_items.append((key, None, val))

        self.type_spec: Optional[Dict[str, ts.ColumnType]] = {}
        for key, idx, _ in self.dict_items:
            if idx is None:
                # TODO: implement type inference for values
                self.type_spec = None
                break
            self.type_spec[key] = self.components[idx].col_type
        self.col_type = ts.JsonType(self.type_spec)

        self.id = self._create_id()

    def __str__(self) -> str:
        item_strs: List[str] = []
        i = 0
        def print_val(val: Any) -> str:
            if isinstance(val, str):
                return f"'{val}'"
            return str(val)
        for key, idx, val in self.dict_items:
            if idx is not None:
                item_strs.append(f"'{key}': {str(self.components[i])}")
                i += 1
            else:
                item_strs.append(f"'{key}': {print_val(val)}")
        return '{' + ', '.join(item_strs) + '}'

    def _equals(self, other: InlineDict) -> bool:
        return self.dict_items == other.dict_items

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('dict_items', self.dict_items)]

    def to_dict(self) -> Dict[str, Any]:
        """Return the original dict used to construct this"""
        return {key: val if idx is None else self.components[idx] for key, idx, val in self.dict_items}

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        result = {}
        for key, idx, val in self.dict_items:
            assert isinstance(key, str)
            if idx is not None:
                result[key] = data_row[self.components[idx].slot_idx]
            else:
                result[key] = copy.deepcopy(val)
        data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        return {'dict_items': self.dict_items, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'dict_items' in d
        arg: Dict[str, Any] = {}
        for key, idx, val in d['dict_items']:
            # TODO Normalize idx -1 to None via schema migrations.
            # Long-term we should not be allowing idx == -1.
            if idx is not None and idx >= 0:  # Older schemas might have -1 instead of None
                arg[key] = components[idx]
            else:
                arg[key] = val
        return cls(arg)
