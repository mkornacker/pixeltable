# pxt438 known bugs and review findings

Snapshot taken after the pxt1068 merge. Two sections:
1. **Known bugs** — surfaced in earlier sessions, deferred / not yet fixed.
2. **New findings** — from the most recent correctness review of pxt438 vs master (read-only).

Sibling docs: `pxt438_review_findings.md` (the original audit), `pxt438_branch_summary.md`
(what the branch does), `catalog_pending_ops_findings.md` (the pxt1068-era catalog
findings inherited via the merge).

---

## Known bugs (deferred or open)

### `_finalize_pending_ops` / `_roll_forward` error-wrap issues (`catalog/catalog.py`)

**P1. `_roll_forward` wraps every non-None return from `_finalize_pending_ops` as `INTERNAL_ERROR` (~`catalog.py:739`).**
Conceals `ConcurrencyError`/`excs.Error` returned from the abort branch, defeating
outer retry semantics and producing misleading user-facing error codes.
*Fix sketch:* dispatch on exception type — re-raise `excs.Error`/`excs.ConcurrencyError`
as-is; only wrap genuinely unexpected exceptions in `INTERNAL_ERROR`.

**P1. `_finalize_pending_ops`'s broad `except Exception` (~`catalog.py:917`) catches `ConcurrencyError`.**
Treats it as an op failure — either switches the table to ROLLBACK or "logs but continues"
without backoff. Tight-loops on persistent errors.
*Fix sketch:* explicit `except excs.ConcurrencyError: raise` before the catch-all. For the
non-abortable branch, propagate the exception (or retry with counter + sleep) rather than
silent-looping.

**P1. `_finalize_pending_ops` self-recursion in the `except PendingTableOpsError` branch is unbounded.**
Under adversarial concurrency (another worker continually adding pending ops on a dependency),
each iteration grows the Python stack. Bounded only by Python's recursion limit.
*Fix sketch:* convert to an iterative drain — collect dependency tbl_ids in a local set/queue,
finalize each, then retry the outer tbl_id in the existing `while True:` loop instead of
stacking calls.

### `Catalog.begin_xact` semantics (`catalog/catalog.py:337-344`)

**P2. Join path silently drops inner `tvps`/`tbl_ids`.**
`retry_loop`'s `PendingTableOps` handling makes this correctness-safe inside retry_loop
contexts, but inefficient — mid-body PendingTableOps forces a wasted retry instead of being
caught at xact start by `begin_xact`'s own lock-acquisition phase.
*Fix sketch:* on the join path, call `_refresh_tbl_cache(tbl_id)` per id and
`_acquire_path_locks(tvp, for_write=False)` per tvp, wrapped in `_allow_tbl_md_read()`.

**P3. Join path doesn't check isolation-mode compatibility.**
`begin_xact(mode=MD_ACCESS)` nested inside an outer `QUERY`/`WRITE_*` runs at READ COMMITTED,
not REPEATABLE READ. Mostly label-hygiene given that nested begin_xact is a no-op, but the
code reads as if MD_ACCESS provides snapshot guarantees when it actually doesn't on join.
*Fix sketch:* assert `Env.get().dbms.isolation_level(mode) == get_runtime().isolation_level`.

### Atomic-move semantics (`catalog/catalog.py:1188+`)

**P2. `_check_dir_move_creates_cycle` doesn't lock intermediate ancestor chain.**
Recursive CTE walks ancestor chain but doesn't row-lock intermediates. Concurrent move of
an intermediate (by another worker with disjoint locks) can commit between our walk and our
`_move_dir`, leaving a cycle.
*Fix sketch:* `SELECT ... FOR UPDATE` on the recursive CTE so it locks each row it
traverses (lock ordering deterministic along chain direction → no deadlock). Or run at
SERIALIZABLE and rely on retry.

### `View._get_metadata` cross-xact composition (`catalog/view.py:282`)

**P3. `base_tbl._path()` opens its own `begin_xact(MD_ACCESS)`.**
Today's only caller (`Table.get_metadata`) wraps in `@retry_loop(mode=MD_ACCESS)` so the
inner join inherits REPEATABLE READ and the composition is atomic. Any future caller from
a non-MD_ACCESS context would see torn `md['name']`/`md['path']`/`md['base']`.
*Fix sketch:* replace nested `base_tbl._path()` with `cat.read_tbl_path(base_tbl_id)` inside
the existing outer xact (the pattern `Table._get_metadata` already uses).

### Misc

**P3. `_acquire_write_lock` (`catalog.py:704`) doesn't forward `check_pending_ops`.**
Inner `_get_tbl_version` call uses default `True` even when caller passed `False`. No current
caller hits the broken combination, but a future one will see surprise `PendingTableOpsError`.

**P3. `_roll_forward_ids` not discarded on successful finalize (`catalog.py:733-739`).**
Each caller `.clear()`s before reuse, so the blast radius today is zero. Cost: future helper
that calls `_roll_forward` without clearing first will re-finalize already-LIVE tables.
Finalization is idempotent (early return on LIVE), so wasted xact per stale id.

**P4. `MD_ACCESS` mode name.**
The semantic is "multi-statement consistency"; the name suggests "any metadata access".
Many "metadata access" sites don't need snapshot isolation. Rename proposed (`SNAPSHOT`?),
deferred. Mechanical search-and-replace when picked up.

---

## New findings from the post-pxt1068-merge review

### P1

**1. `_drop_tbl` TOCTOU on `dir_id` (`catalog/catalog.py:1879`).**

```python
self._acquire_dir_xlock(dir_id=self.read_tbl_record(tbl_id).dir_id)
self._acquire_write_lock(tbl_id=tbl_id)
```

Unlocked read of `dir_id`, then X-locks that dir, then X-locks the table. A concurrent
`_move_table` can shift the table between the unlocked read and the dir-X-lock — we'd hold
a write lock on the *wrong* (old) directory while dropping. The dir-X-lock's intent is to
prevent a racing `create_table` at the same `(dir, name)` slot from succeeding during the
drop; that protection is silently lost.

*Fix.* Re-read `dir_id` from inside the locked Table row, or lock the Table first then read
`dir_id` then X-lock the dir.

### P2

**2. `_fastapi.py:1158` uses `QUERY` for a metadata read.**

```python
with begin_xact(mode=XactMode.QUERY, tvps=template_query._from_clause.tbls):
    effective_select_list = list(template_query._effective_select_list)
```

This is metadata resolution, not data execution. Per the new contract should be
`MD_ACCESS`. The read happens to be single-step today so READ COMMITTED doesn't bite;
future additions to the block would silently see torn snapshots.

*Fix.* `mode=XactMode.MD_ACCESS`. (`_fastapi.py:1261`'s `run_query` is correctly `QUERY`.)

**3. `Catalog.update_additional_md` clears cache *before* SQL UPDATE (`catalog.py:1806`).**

```python
self._clear_tv_cache(TableVersionKey(tbl_id, None, None))
result = conn.execute(q)
assert result.rowcount == 1
```

Cache invalidated first, then SQL ran. Reverses the standard "modify store, then invalidate
cache" protocol. Today the surrounding WRITE_TBL X-lock prevents concurrent drop, but if
`rowcount != 1` for any reason we've corrupted the cache and crashed with `AssertionError`
instead of a clean `table_was_dropped`.

*Fix.* Move `_clear_tv_cache` after the asserted UPDATE; convert the assert to
`if result.rowcount != 1: raise excs.table_was_dropped(tbl_id)`.

### P3

**4. `globals.py:680` opens a pointless outer `begin_xact` around `_path()`.**

```python
with get_runtime().catalog.begin_xact(mode=XactMode.MD_ACCESS):
    tbl_path = table._path()
```

`_path()` already opens its own atomic MD_ACCESS xact. Wrapper declares no targets, takes
no locks, holds no extra invariant. Misleading: invites a future reader to add a second
read into the block assuming snapshot consistency.

*Fix.* Drop the wrapper. `tbl_path = table._path()`.

**5. `_lock_tbl_if_exists` / `_acquire_write_lock` don't assert UPDATE rowcount (`catalog.py:649, 676`).**

```python
# SELECT ... FOR UPDATE NOWAIT  (locks the row)
conn.execute(sql.update(...).values(lock_dummy=1).where(...))
# rowcount not asserted
```

Today safe because the prior `SELECT FOR UPDATE NOWAIT` guarantees the row exists. But the
rowcount-check is the explicit invariant; future refactor that moves the SELECT silently
loses protection.

*Fix.* `result = conn.execute(...); assert result.rowcount == 1, tbl_id`.

**6. `get_view_ids` (`catalog.py:2074`) doesn't filter on mutability.**

JSON-path query matches any view with `base_versions[0][0] == tbl_id.hex` — including
snapshot views. Compare to `_load_tbl_version` line 2916 which adds `base_versions[0][1] IS NULL`
for mutable views. Drop-cascade callers want all dependents, so conservative-correct today.
But the asymmetry is fragile: a future caller wanting "mutable only" would use this and overshoot.

*Fix.* Rename to clarify (`get_all_view_ids`), or accept `mutable_only: bool = False`.

---

## Categories examined that yielded no new issues

- Atomic-read invariants (`assert get_runtime().in_xact`, recursive-CTE atomicity).
- Write-locking protocol (WRITE_TBL/WRITE_TREE targets X-locked before mutation).
- `TableVersion` thread/xact coherence (`is_validated` reset at xact end, per-thread `_local`).
- Name-column invariants (no surviving `md->'name'` or `tbl_md.name` reads for tables).
- SchemaObject contract (`Dir`/`Table`/`View` atomic `_name`/`_path`/`_display_name`; no surviving `_dir_id`/`_parent` callers).
- PendingTableOps handling for raw `with begin_xact(...)` blocks (all checked sites only resolve already-cached metadata; no new `_load_tbl_version` triggered).

## Recommended pick-up order (when this gets attention)

1. Findings P1 (#1 `_drop_tbl` TOCTOU). Possible reachable bug.
2. Known P1 issues in `_finalize_pending_ops` / `_roll_forward` (the error-wrap and recursion items). Cheap, makes debugging future incidents tractable.
3. Findings P2 (#2 / #3). Hygiene + ordering.
4. Known P2 (`begin_xact` join path read-target extension, `_check_dir_move_creates_cycle` chain locking).
5. P3/P4 cleanup batch.
