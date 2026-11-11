# Expressions

Operators, casts, nulls and time on column values, and how to find and try the built-in functions.

## Finding functions

- `pxt functions` lists the functions, aggregates and iterators, one per line, with how to call each one and the
  first paragraph of its docstring. It needs no daemon or catalog.
- `pxt functions video` lists a module of `pixeltable.functions`, and `pxt functions app` the UDFs of `app.py`.
  `--on Timestamp` lists the methods and properties of a type, `--search crop` matches names and docstrings, `--kind`
  keeps one of `udf`, `aggregate`, `iterator` and `query`, and `--json` prints the records with their full
  docstrings.
- `pxt.list_functions()` takes the same filters (`module`, `on=`, `search=`, `kind=`) and returns a `FunctionList` of
  dicts with `path`, `name`, `kind`, `call`, `signatures` and `summary`.
- A property, such as `Timestamp.year` or `Image.width`, takes no parentheses: `t.departs_at.year`.

## Operators

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

pxt.create_dir('exprs', if_exists='ignore')
trips = pxt.create_table(
    'exprs.trips',
    {'id': pxt.Int, 'driver': pxt.String | None, 'delay': pxt.Float, 'departs': pxt.String, 'note': pxt.String | None},
)
trips.insert(
    [
        {'id': 1, 'driver': 'Ana', 'delay': 7.5, 'departs': '03/04/2026 10:30'},
        {'id': 2, 'driver': None, 'delay': 1.0, 'departs': '04/04/2026 08:00', 'note': 'rerouted'},
    ]
)
late = trips.where((trips.delay > 5.0) & (trips.driver != None)).select(trips.id, label='Driver ' + trips.driver)
num_late = trips.select(pxtf.sum((trips.delay > 5.0).astype(pxt.Int))).collect()
parsed = trips.select(
    departs_at=pxtf.timestamp.strptime(trips.departs, '%d/%m/%Y %H:%M'),
    driver=pxtf.coalesce(trips.driver, trips.note, 'unknown'),
    checked_at=pxtf.timestamp.now(),
).collect()
```

- Numbers take arithmetic, strings take `+`, and conditions combine with `&`, `|` and `~`. `x == None` and
  `x != None` test for null; `x.isin(['a', 'b'])` tests membership.
- Python `and`, `or` and `not` on a column raise `TypeError`, and `x is None` is a Python `bool`, not a condition.
- An f-string or `str.format()` on a column raises `An f-string or str.format() on a Pixeltable expression produces
  one constant string, not a value per row`. `str(col)` and `'%s' % col` raise nothing and give one constant string
  for every row; `pxt schema check` warns about a model computed column that references no column.
  `'Driver ' + driver` and `pxtf.string.format('Driver {0}', driver)` build a string per row.

## Casts

`astype()` sets an expression's type. These casts also convert the value:

| From | `astype()` to | Result |
|---|---|---|
| Bool | `pxt.Int`, `pxt.Float` | 0 for `False`, 1 for `True`, so `pxtf.sum(cond.astype(pxt.Int))` counts matching rows |
| Int | `pxt.Float` | the number as a float |
| ISO 8601 String | `pxt.Timestamp`, `pxt.Date` | the parsed value; a timestamp without a time zone is in the default time zone |
| path or URL String | `pxt.Image` | the image file |

- `pxtf.timestamp.strptime(s, '%d/%m/%Y %H:%M')` and `pxtf.date.strptime(s, '%d/%m/%Y')` parse other formats; a
  string that does not match fails that row.
- A UUID does not cast to a String: `astype(pxt.String)` fails each row with `TypeError`. `pxtf.uuid.to_string(col)`
  returns the string.
- A cast of a Json value checks the value when the expression runs: `pxt guide json`.

## Nulls

- `pxtf.coalesce(a, b, ...)` returns its first argument that is not null. The result type is the common supertype of
  the arguments' types (`Float` for `Int` and `Float`, `Json` for `String` and `Int`), and it is nullable only if every
  argument is.
- Pixeltable has no conditional expression; a branch on a value is a UDF (`pxt guide udfs`, which also says how UDFs
  treat a null argument).

## Time

- `pxtf.timestamp.now()` is the current time in the default time zone. `pxtf.date.today()` is the current date in the
  default time zone, and `pxtf.date.today('Asia/Tokyo')` in another one.
- A query, or a computed column with `stored=False`, evaluates them when it runs. A stored computed column records
  the time of the insert, and a recompute replaces it with the time of the recompute.
- Timestamps take no `+` or `-`, which raise `+ requires numeric types`. The difference of
  two `posix_timestamp()` values is the number of seconds between them; `date.add_days(n)` moves a Date.

## Media values

- `col.fileurl` is the URL of a media cell's file, and `col.localpath` a local path to it.
- `col.errortype` and `col.errormsg` record a failed cell of a stored computed or media column (`pxt guide tables`).

## Trying an expression

- A `select()` evaluates an expression on existing rows and stores nothing; `limit()` keeps the number of model
  calls small.
- `pxt.eval_udf(pxtf.string.upper, 'text')` calls a UDF, a built-in function or a `@pxt.query` function on Python
  values, without a table. A media argument can be a file path, a URL or a PIL image.
- `eval_udf()` refuses an aggregate (`sum() is an aggregate function, which needs rows; use it in a query`) and a
  column as an argument (`the arguments must be Python values, not columns`).

```bash
pxt functions --on Timestamp
pxt functions --search coalesce
pxt functions video --kind iterator
pxt functions app --kind udf --json
```
