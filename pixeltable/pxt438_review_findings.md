# pxt438 correctness review — findings

Review focus: correctness on the pxt438 branch after the master merge (which brought in
PXT-1149 thread-safe Table, PXT-1153 FastAPIRouter, PXT-1172 store-tbl concurrent-drop fix),
and in anticipation of the upcoming pxt1068 (SELECT-* recompilation) merge.

The trigger for this review was a bug we found on pxt1068 where `Query.collect()` opens an
outer `begin_xact(read_tvps=...)` without `read_tbl_ids`, causing the inner
`_output_row_iterator`'s `read_tbl_ids` to be silently dropped on the join path — voiding
PXT-1149's fix for query-UDF sub-queries. pxt438 has the same shape in several places.

The decision was: wait for pxt1068 to merge into master and then into pxt438, since the
issues here will likely be touched by that merge anyway. This doc captures the findings so
they aren't lost.

---

## P1 — bugs/regressions exposed or caused by the master merge

### 1. Catalog.begin_xact join path silently drops tvps/tbl_ids

**Location:** `pixeltable/catalog/catalog.py:337-344`

The `in_xact` branch only asserts that any inner *write* targets are already locked by the
outer xact. Inner *read* targets (tvps/tbl_ids that would extend the catalog metadata cache)
are no-op'd. So any caller that supplies additional read targets while nested inside an
outer xact has those targets silently ignored — the cache never gets refreshed, and a later
lookup mid-xact trips the assertion at `catalog.py:2076` ("Loading new table metadata is
not allowed in the middle of a transaction").

**Current manifestations on pxt438:**

- `_query.py:1811` (`to_coco_dataset`) and `_query.py:1856` (`to_pytorch_dataset`) open an
  outer `begin_xact(MD_ACCESS, tvps=self._from_clause.tbls)` — no `tbl_ids`. Then the
  query execution path eventually calls `_output_row_iterator` which DOES pass
  `tbl_ids=self.referenced_tbl_ids()`, but joins → `tbl_ids` lost. Any sub-query UDF
  that references a table outside the from-clause triggers the assertion.

**Same shape but currently benign:** `globals.py:680-681` (`pxt.drop_table` by handle)
where the inner `_path()` uses `read_tbl_path` (a single statement) and doesn't go through
`_get_tbl_version`. The contract is still fragile.

**Fix sketch:** in the `in_xact` branch of `begin_xact`, process `tvps`/`tbl_ids` before
yielding — call `_refresh_tbl_cache(tbl_id, check_pending_ops=...)` per id and
`_acquire_path_locks(tvp, for_write=False, ...)` per tvp. Then yield. Fixes both call sites
and any future ones.

**Critical for pxt1068 merge:** pxt1068's `Query.collect()` opens its own outer xact (with
both tvps and tbl_ids — see the fix I landed on pxt1068). When merging pxt1068 unchanged,
`_output_row_iterator`'s nested call will hit this join-path issue. Fix #1 first.

---

### 2. begin_xact join path does no isolation-mode compatibility check

**Location:** `pixeltable/catalog/catalog.py:337-344`

PostgreSQL connection isolation is set once on the outer `begin_store_xact`
(`runtime.py:200`). So `begin_xact(mode=XactMode.MD_ACCESS)` nested inside an outer
`QUERY` or `WRITE_*` xact silently runs at READ COMMITTED, defeating the
snapshot-consistency promise that `MD_ACCESS` documents.

`Dir._name()` (`dir.py:38`), `Dir._path()` (`dir.py:44`), `Table._name()`
(`table.py:78`), `Table._path()` (`table.py:83`) all do bare
`begin_xact(mode=MD_ACCESS)` with no targets. Any caller that wraps them in a
non-MD_ACCESS outer xact gets the downgrade. The downgrade is silent — no error, no
warning — and produces subtly inconsistent reads if a rename commits between two atomic
reads that the caller assumed were in the same snapshot.

Note: `runtime.begin_store_xact` (`runtime.py:189`) *does* assert isolation-level match
for its own callers — but `Catalog.begin_xact` joins before that check fires.

**Fix sketch:** in the `in_xact` branch, assert
`Env.get().dbms.isolation_level(mode) == get_runtime().isolation_level`. If they don't
match, that's a programming error. Alternative: document that the inner `mode` is
advisory when joined and leave it; less safe.

---

### 3. _finalize_pending_ops lacks the recursive PendingTableOpsError handler we added on pxt1068

**Location:** `pixeltable/catalog/catalog.py:714-905`

The exception ladder catches `AssertionError`, `DBAPIError`, generic `Exception` — but
no explicit `PendingTableOpsError`. On pxt1068 the equivalent function was patched: when
finalizing table A's pending ops needs to load metadata for table B (e.g., a stored
expression referencing B's columns), and B has its own pending ops, the load raises
`PendingTableOpsError(B)`. The handler recurses (`self._finalize_pending_ops(e.tbl_id)`),
then continues with A. Without it, the error falls into the generic catch-all and either
gets logged-and-swallowed (`is_rollback=False`) or rethrown as `INTERNAL_ERROR`. Random-ops
will hit this the same way it did on pxt1068.

**Fix sketch:** port the pxt1068 handler:
```python
except PendingTableOpsError as e:
    self._finalize_pending_ops(e.tbl_id)
    continue
```
placed BEFORE the generic `except Exception` in the loop's exception ladder.

---

## P2 — atomic-metadata correctness

### 4. _check_dir_move_creates_cycle doesn't lock the intermediate ancestor chain

**Location:** `pixeltable/catalog/catalog.py:1188, 1193-1215`

The cycle check is a recursive CTE walk from `dest_parent_id` up to the root, run after
the immediate src/dest X-locks are held. Intermediate ancestors on the chain aren't
locked. The xact runs at READ_COMMITTED (WRITE_TBL). A concurrent move of one of those
intermediates can commit between our CTE walk and our subsequent `_move_dir`, leaving a
cycle behind.

**Concrete scenario:** src=A, dest=under D; chain at check is `D→C→B→root`. We hold A,
D, A's parent. Another worker concurrently moves B under A (it holds A, C, B — disjoint
from our locks). Both commits succeed → cycle `D→C→B→A→D`.

**Fix sketch:** make the recursive CTE use `SELECT ... FOR UPDATE` so it acquires row
locks along the traversed chain (lock acquisition deterministic, ordered by chain
direction so no deadlock). Alternative: run the cycle-checking xact at SERIALIZABLE and
rely on retry. The lock variant is simpler.

---

### 5. View._get_metadata calls base_tbl._path() which opens its own xact

**Location:** `pixeltable/catalog/view.py:282`

After capturing the atomic snapshot via `super()._get_metadata()`, the call
`base_tbl._path()` opens a fresh `begin_xact(mode=MD_ACCESS)`. If the outer caller's
xact is MD_ACCESS, the join continues the snapshot — fine. If the outer caller's xact
is anything else (joined at READ_COMMITTED per Finding #2), `_path()` reads from a
*different* statement, so `md['name']`/`md['path']` (current table) and `md['base']`
(base table) can disagree if a rename commits between them.

Today's only direct caller, `Table.get_metadata`, uses `@retry_loop(mode=MD_ACCESS)`, so
currently safe. But the contract is fragile — any future caller from a non-MD_ACCESS
xact gets torn data.

**Fix sketch:** replace the nested `base_tbl._path()` with a direct
`cat.read_tbl_path(base_tbl_id)` call inside the existing outer xact, mirroring the
pattern `Table._get_metadata` already uses (`table.py:164`).

---

## P3 — forward-looking: pxt1068 merge

pxt1068 will be merged into master and then into pxt438. Inventory of things to translate:

### 6. begin_xact API translation

Every pxt1068 `begin_xact` call site will need translation to the `mode=...` API:

- `for_write=False, read_tvps=X` → `mode=XactMode.QUERY, tvps=X` (or `MD_ACCESS` —
  pxt1068 doesn't distinguish; read each site for caller intent)
- `for_write=False, read_tvps=X, read_tbl_ids=Y` (the new `Query.collect()` /
  `_output_row_iterator`) → `mode=XactMode.QUERY, tvps=X, tbl_ids=Y`
- `for_write=True, write_tbl_ids=[id], lock_mutable_tree=False` → `mode=WRITE_TBL`
- `for_write=True, write_tvps=[...], lock_mutable_tree=True` → `mode=WRITE_TREE`

**Critical ordering:** Fix #1 (begin_xact join path extends tvps/tbl_ids) must land
first. Otherwise the pxt1068 Query.collect() outer xact + _output_row_iterator inner
xact pattern carries the same bug into pxt438.

### 7. validate_tbls_exist on Catalog

pxt1068 adds `Catalog.validate_tbls_exist(tbl_ids)` (`catalog.py:264-273`) used in the
SELECT-* recompilation path. Its `begin_xact()` (no-arg, pxt1068 default) needs an
explicit `mode=MD_ACCESS` when ported. Mechanical.

### 8. _effective_select_list / Query.schema

pxt1068 adds re-resolution of the select list against current catalog state for
SELECT-* queries. The re-resolution walks the from-clause tvps — no obvious interaction
with pxt438's atomic-metadata work, but worth verifying that the re-resolution happens
inside a MD_ACCESS xact (otherwise different columns from different snapshots).

---

## Items checked and dismissed

- **PXT-1172 (`_create_system_columns` defensive `len(col_names) == 0` check)**: under
  MD_ACCESS REPEATABLE_READ the snapshot is consistent across the exists-check and the
  info_schema query, so the defensive branch becomes dead code (harmless). Under
  WRITE_TBL/WRITE_TREE READ_COMMITTED the check still works as intended. No actual bug,
  but worth a comment in the code.
- **PXT-1149 `_dir_id()` / `_parent()` abstraction**: dropped during the merge; tree
  grepped, no stale callers remain.
- **PXT-1153 (FastAPIRouter dedup)**: orthogonal to catalog work; no diff between
  master and pxt438 on `_fastapi.py`.

---

## Recommended fix order (after pxt1068 merges in)

1. **#1** (begin_xact join path extends targets) — unblocks the pxt1068 merge integration
   and kills the latent `to_coco_dataset`/`to_pytorch_dataset` bugs.
2. **#3** (_finalize_pending_ops recursion) — random-ops stress will hit it; cheap to
   port from pxt1068.
3. **#2** (isolation-mode check on join) — silent downgrade → hard failure; small.
4. **#4** (cycle-check chain locking) — concurrent-move correctness.
5. **#5** (View._get_metadata direct read) — defensive hardening.

## File:line index

- `pixeltable/catalog/catalog.py:306-448` — `begin_xact` (join path: 337-344)
- `pixeltable/catalog/catalog.py:714-905` — `_finalize_pending_ops`
- `pixeltable/catalog/catalog.py:1184-1215` — `_move_table` / `_move_dir` / cycle check
- `pixeltable/catalog/catalog.py:2076` — assertion that fires under Finding #1
- `pixeltable/catalog/table.py:76-84,164` — `_name`/`_path` and `_get_metadata`
- `pixeltable/catalog/view.py:270-302,333-339` — `_get_metadata`
- `pixeltable/_query.py:782-794` — `_output_row_iterator` (passes tbl_ids)
- `pixeltable/_query.py:1811,1856` — `to_coco_dataset` / `to_pytorch_dataset` (outer xacts)
- `pixeltable/runtime.py:41-57,169-210` — `XactMode`, isolation in `begin_store_xact`
- `pixeltable/utils/dbms.py:61-66` — MD_ACCESS → REPEATABLE_READ mapping
- `pixeltable/store.py:125-173` — `_create_system_columns` (no issue under MD_ACCESS)
