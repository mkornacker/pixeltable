# pxt438 branch summary

## What this branch is trying to accomplish

PXT-438 hardens the Pixeltable catalog so that metadata operations are
**atomic against concurrent access**. Two related goals:

1. **Single-statement atomicity for composite catalog reads.** Things like
   "what's the path of this table" or "what's the cycle structure of this
   directory tree" used to be implemented as multiple SQL round-trips, which
   could race against `move()` / `rename()` and produce torn results. The
   branch reimplements them as one SQL statement (recursive CTEs) so a
   reader either sees the pre-move state or the post-move state, never a
   mix.

2. **Multi-statement metadata consistency, expressed in the type system.**
   When code does need to read several pieces of catalog metadata that have
   to agree (e.g. `get_metadata()` reads name + path + base + columns), the
   transaction is declared explicitly via `Catalog.begin_xact(mode=...)`
   with a mode that carries an isolation guarantee. Modes are first-class
   in the API rather than implicit via `for_write=True/False`.

## Design changes

### New `XactMode` (`pixeltable/runtime.py`)

```python
class XactMode(enum.Enum):
    MD_ACCESS  = 'md_access'   # REPEATABLE_READ: consistent multi-statement metadata reads
    QUERY      = 'query'       # READ_COMMITTED: single-statement / data queries
    WRITE_TBL  = 'write_tbl'   # READ_COMMITTED + X-lock the target table(s)
    WRITE_TREE = 'write_tree'  # READ_COMMITTED + X-lock target plus all mutable descendant views
```

`Catalog.begin_xact` now takes `mode=...`, `tvps=...`, `tbl_ids=...` instead
of the old `for_write=...`, `read_tvps=...`, `write_tvps=...`. Mode controls
both isolation level and lock scope; targets are interpreted per mode.

### Atomic catalog read helpers (`pixeltable/catalog/catalog.py`)

- `read_dir_path(dir_id)` — recursive CTE up the parent chain, returns
  full path components in one statement.
- `read_tbl_path(tbl_id)` — same shape but joins to `schema.Table` to
  include the table name.
- `read_tbl_name(tbl_id)` — single SELECT on `schema.Table.name`.
- `read_tbl_record(tbl_id)` / `read_dir_record(dir_id)` — single
  SELECT for the full record.
- `_check_dir_move_creates_cycle(src, dest)` — atomic SQL recursive CTE
  cycle detection, run after parent X-locks are held in `move()`.

### Identity column on `schema.Table`

`schema.Table.name` is now a dedicated column (not buried in the `md` JSON
blob). `Catalog._move_table` writes only `name` + `dir_id`, never touches
`md`. This makes the move atomic against concurrent metadata writers and
removes a class of "name in md is stale w.r.t. name we just wrote" bugs.

### `SchemaObject` thread-safety contract

```python
class SchemaObject:
    # Thread-safety contract:
    # - no mutable state
    # - each attribute access (e.g., _name) implemented with guaranteed atomicity by the subclass
    # - multiple consecutive attribute accesses are not guaranteed to be atomic
    # - attributes or other state cannot be cached, which would be problematic with concurrent writes
```

Subclasses (`Dir`, `Table`, `View`) implement `_name()` / `_path()` /
`_display_name()` as method calls that route through the atomic catalog
read helpers above.

### Table query-builder fail-fast on dropped tables

`Table.where` / `.join` / `.order_by` / `.group_by` / `.distinct` /
`.limit` / `.sample` / `.select(*args)` each open a
`begin_xact(mode=MD_ACCESS, tvps=[self._tbl_version_path])` for the
construction call. A dropped-table reference surfaces immediately at
construction time, not later during execution.

## Bugs fixed

1. **Concurrent `move()` could leave a stale name in the `md` JSON column.**
   Fixed by making `name` an identity column and updating it directly.

2. **`move()` could create a directory cycle under concurrent moves.**
   Fixed by `_check_dir_move_creates_cycle()` running an atomic recursive
   CTE while parent locks are held.

3. **Composite reads (`_name()` + `_path()`) could see torn results from a
   concurrent rename.** Fixed by `read_tbl_path()` returning all
   components in a single statement.

4. **`Table.distinct/limit/sample/...` returned a `Query` that failed only
   on `.collect()` when the table was already dropped.** Fixed by opening
   a fail-fast `begin_xact` at construction.

5. **`Table.compute()` did not exist as a documented API path** (added
   alongside the catalog work).

## Bugs picked up via master merges

The branch sat through several master merges; merge integration brought in
and adapted these fixes from master:

- **PXT-1149 (master)**: thread-safe `Table` handles. pxt438 adopted the
  per-thread `_local` cache pattern in `TableVersionHandle`.
- **PXT-1153 (master)**: `FastAPIRouter` rejects duplicate routes.
- **PXT-1172 (master)**: `Table` load could raise `AssertionError` if the
  table was dropped concurrently during `_create_system_columns`. pxt438
  adapted the defensive `len(col_names) == 0 → table_was_dropped` check
  to the new atomic-metadata structure.

## pxt1068 just merged in (this session)

PXT-1068 ("recompile SELECT * query") was merged into this branch.
Brings:

- **SELECT-* recompilation**: `Query.schema` and `Query._effective_select_list`
  re-resolve against the current catalog state at execution time rather
  than at construction. Stored SELECT-* queries pick up schema changes.
- **`validate_tbls_exist`**: early failure for stored Queries whose
  referenced tables no longer exist.
- **`Query.collect()` / `count()` / `to_coco_dataset()` /
  `to_pytorch_dataset()`** open outer xacts with full `tbl_ids` (including
  transitive `@pxt.query` UDF refs), so a sub-query UDF over a table outside
  the from-clause doesn't trip the mid-xact metadata-load assertion.
- **`Table.get_metadata`** `@retry_loop` decorator now declares
  `tvps=[self._tbl_version_path]`, so `View._get_base_table` finds the base
  pre-loaded in cache.
- **`ColumnRef._from_dict`** raises `NotFoundError(COLUMN_NOT_FOUND)` with
  the same "Column was dropped (...)" wording used by `ColumnHandle.get()`
  when a referenced column is gone, replacing a bare `KeyError`.
- **`QueryTemplateFunction._from_dict`** catches `NotFoundError` from
  `Query.from_dict` and returns an `InvalidFunction` placeholder, matching
  the existing UDF-symbol-missing pattern. A stored `@pxt.query` UDF whose
  referenced table or column is dropped now degrades gracefully (host table
  loads, column is marked invalid, new inserts raise `INVALID_STATE`) rather
  than crashing reload.
- **`_finalize_pending_ops`** handles `PendingTableOpsError` recursively
  for cross-table dependencies (random-ops stress test fix).
- **`Catalog.create_table`** post-`_roll_forward` lookup wrapped in
  `@retry_loop(mode=MD_ACCESS, tbl_ids=[tbl_id])` instead of a raw
  `begin_xact(WRITE_TBL, ...)` — it's a read lookup; was the wrong shape
  before.

## What is *not* in this branch

- The `begin_xact` join-path read-target assertion (proposed but reverted —
  retry_loop already handles the correctness concern; assertion was too
  aggressive).
- Isolation-mode compatibility assertion (deferred — would only enforce
  label hygiene given that nested begin_xact is a no-op).
- Renaming `MD_ACCESS` to `SNAPSHOT` (proposed, deferred).
- `View._get_base_table()` / `TableVersionPath._cached_tv()` refactor to
  drop their inner `begin_xact` (would be needed for clean semantic
  assertions; left for follow-up).
