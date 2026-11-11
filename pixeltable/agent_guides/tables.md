# Tables

Declaring tables, writing rows, and changing a table's schema.

## Declaring a table

An application declares its tables as `TableModel` classes in `app.py`; `pxt schema update app.py TARGET` creates
them under the catalog directory `TARGET`.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    title: pxt.String
    body: pxt.String | None
    tags: pxt.Json[list[str]]
    cover: pxt.Image | None
    title_upper = pxtf.string.upper(title)
    thumbnail = pxt.Column(value=cover.resize((64, 64)), stored=False)
```

- An annotation declares a stored column; an assignment declares a computed column.
- `T` is non-nullable: an insert without a value fails with `expected non-None value`. `T | None` is nullable.
- `pxt.Column()` takes what an annotation cannot express: `primary_key=True`, `stored=False` (computed when a query
  reads it, never stored), `media_validation='on_read'` (the default is `'on_write'`), `destination=` (where media
  files go), `comment=` and `custom_metadata=`.
- Types: `String`, `Int`, `Float`, `Bool`, `Timestamp`, `Date`, `UUID`, `Binary`, `Json`, `Array[(3, 4), pxt.Float]`,
  `Image`, `Video`, `Audio`, `Document`.
- `pxt.Json[...]` takes a TypedDict or `list[...]`; its paths then return typed values. A `NotRequired[...]` key's
  path is nullable.
- In a script, `TableModel.bind_all('my_app')` binds the classes to the tables, after which `Docs.insert(...)` and
  `Docs.select(...)` work. `pxt.get_table('my_app.docs')` returns the same table as a `Table`.

## Primary keys

- A primary key is declared when the table is created. A model that adds one to an existing table is refused:
  `Cannot add primary key column ... after table creation`.
- `batch_update()`, `pxt get` and `pxt errors` require a primary key, and so do update and delete routes.
- Inserting a row whose key already exists fails with `CONSTRAINT_VIOLATION` (`Duplicate primary key value`).

## Writing rows

```python
pxt.create_dir('my_project', if_exists='ignore')
docs = pxt.create_table(
    'my_project.docs', {'id': pxt.Int, 'title': pxt.String, 'body': pxt.String | None}, primary_key='id'
)
docs.add_computed_column(title_upper=pxtf.string.upper(docs.title))

status = docs.insert([{'id': 1, 'title': 'a'}, {'id': 2, 'title': 'b', 'body': 'text'}], return_rows=True)
docs.insert(id=3, title='c')
docs.insert([{'id': 3, 'title': 'c2'}, {'id': 5, 'title': 'e'}], if_exists='ignore')
docs.update({'body': 'reviewed'}, where=docs.id == 1)
docs.batch_update([{'id': 2, 'title': 'b2'}, {'id': 4, 'title': 'd'}], if_not_exists='insert')
docs.delete(where=docs.id == 3)
```

- `insert()` takes a list of dicts, one row as keyword arguments, a file path or URL (`source_format=` is `'csv'`,
  `'excel'`, `'parquet'` or `'json'`; `schema_overrides=` sets column types), or a query.
- Known issue (PXT-1749): an insert from a query does not compute the target's computed columns; they stay null.
- `if_exists='ignore'` skips each row whose primary key is already in the table, and does not compute the skipped
  rows' columns; `status.row_count_stats.skipped_rows` counts them. If a batch repeats a new key, the first row wins,
  and concurrent inserts of the same key insert it once. The default, `if_exists='error'`, fails the insert.
- `return_rows=True` returns the inserted rows, computed columns included, in `status.rows`.
- A computed column that raises fails the whole insert and stores nothing. With `on_error='ignore'` the row is
  stored, the failed cell is null, and `col.errortype` and `col.errormsg` record the error.
- `update()` recomputes the computed columns that depend on the updated columns (`cascade=False` keeps them).
- `batch_update()` matches rows by primary key. `if_not_exists='insert'` inserts the rows with unknown keys;
  `'ignore'` skips them. Every matched row's dependent computed columns are recomputed, even when the new values
  equal the old ones, so a column that calls a paid model runs again.
- `delete()` without `where` deletes every row.
- Known issue (PXT-1760): a write that changes 0 rows still creates a table version.
- Each `insert()`, `update()`, `batch_update()` and `delete()` is atomic, and writes to one table run one at a time.
  `status.num_rows` reports the rows an operation changed, so an `update()` whose `where` includes the expected
  current value is a compare-and-set.

## Computed columns

- A computed column is evaluated when its row is inserted, or when an update changes one of its inputs, in the same
  operation.
- Adding a computed column to a table that has rows computes it for every existing row.
- A computed column without column inputs, such as `uuid7()`, gets its value at insert; updates never recompute it.
- `recompute_columns('summary', errors_only=True)` reruns the failed cells of a column; `where=` restricts the rows,
  and the columns that depend on it are recomputed too unless `cascade=False`. The CLI form is
  `pxt recompute PATH COLUMN`, with `--errors-only` and `--no-cascade`.

## Changing the schema

`pxt schema diff app.py TARGET` shows what `pxt schema update app.py TARGET` would change in each table:

| Change in `app.py` | Classified as | Effect of `pxt schema update` |
|---|---|---|
| A new column | additive | Added; a new computed column is computed for every existing row |
| A new expression for a computed column | additive | Existing rows keep their values; the command prints the `pxt recompute` command that recomputes them |
| A stored column becoming nullable (`T` to `T \| None`) | additive, or unsupported if a non-nullable computed column of the table or of one of its views references the column | Altered in place; the refusal lists those computed columns |
| A removed column | destructive | Applied only with `--allow-destructive`; the column's data is dropped |
| Any other change to a column: its type, `primary_key`, `stored`, `media_validation`, `destination`, `comment` or `custom_metadata` | unsupported | Refused |

One refused change refuses the whole update: nothing is applied, not even the additive changes.

A computed column over a non-nullable column is non-nullable unless its expression can return null, so it blocks
making that column nullable. A UDF return type that becomes narrower, such as `str | None` to `str`, is not a change.

In a notebook, the table methods change the schema directly: `add_column()`, `add_computed_column()`,
`drop_column()`, `rename_column()`, `alter_column(col, type_=T | None)`, which makes a column nullable under the same
rule, and `alter_computed_column()`, which recomputes the column by default (`recompute=False` keeps the old values).

## Versions

- Each change creates a new version of the table. `t.history()` and `pxt history PATH` list them.
- `t.revert()` and `pxt revert PATH --steps N` undo the last operations on that table, for everyone who uses it.
  A revert refuses version 0 (`Cannot revert version 0`) and a version that a snapshot needs (`Current version is
  needed for 1 snapshot: (docs_v3)`).
- `pxt.get_table('my_app.docs:3')` opens version 3, read-only. `pxt.create_snapshot('my_app.docs_v3', docs)` keeps
  the current rows as a read-only snapshot.

```bash
pxt schema diff app.py my_app
pxt schema update app.py my_app
pxt errors my_app/docs
pxt recompute my_app/docs title_upper --errors-only -f
pxt get my_app/docs 2
pxt history my_app/docs
pxt revert my_app/docs --steps 2 -n
```
