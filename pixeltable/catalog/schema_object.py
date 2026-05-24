import abc
from uuid import UUID


class SchemaObject(abc.ABC):
    """Base class of all addressable objects within a Db.

    Thread-safety contract:
    - no mutable state
    - each attribute access (eg, _name) needs to be implemented with guaranteed atomicity by the subclass
    - multiple consecutive attribute accesses are not guaranteed to be atomic
    - attributes or other state cannot be cached, which would be problematic with concurrent writes
    """

    _id: UUID

    def __init__(self, obj_id: UUID):
        self._id = obj_id

    @abc.abstractmethod
    def _name(self) -> str:
        """Name of this object, as recorded in the catalog."""

    @abc.abstractmethod
    def _path(self) -> str:
        """Full path to this object."""

    @abc.abstractmethod
    def _display_name(self) -> str:
        """Name displayed in error messages."""

    def _display_str(self) -> str:
        """Best-effort display string for this object."""
        return f'{self._display_name()} {self._path()!r}'
