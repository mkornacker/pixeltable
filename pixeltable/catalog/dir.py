from __future__ import annotations

import dataclasses
import logging
from uuid import UUID

from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import XactMode, get_runtime

from .schema_object import SchemaObject

_logger = logging.getLogger('pixeltable')


class Dir(SchemaObject):
    def __init__(self, id: UUID):
        super().__init__(id)

    @classmethod
    def _create(cls, parent_id: UUID, name: str) -> Dir:
        session = get_runtime().session
        user = Env.get().user
        assert session is not None
        dir_md = schema.DirMd(name=name, user=user, additional_md={})
        dir_record = schema.Dir(parent_id=parent_id, md=dataclasses.asdict(dir_md))
        session.add(dir_record)
        session.flush()
        assert dir_record.id is not None
        assert isinstance(dir_record.id, UUID)
        return cls(dir_record.id)

    def _display_name(self) -> str:
        return 'directory'

    def _name(self) -> str:
        cat = get_runtime().catalog
        with cat.begin_xact(mode=XactMode.MD_ACCESS):
            components = cat.read_dir_path(self._id)
            return components[-1] if components else ''

    def _path(self) -> str:
        cat = get_runtime().catalog
        with cat.begin_xact(mode=XactMode.MD_ACCESS):
            return '/'.join(cat.read_dir_path(self._id))
