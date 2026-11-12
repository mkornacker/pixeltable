# Queries

Building and running queries, reading their results, and defining `@pxt.query` functions.

## Queries in an application

In `app.py`, a query is a `@pxt.query` function over the model classes. It runs as the value of a computed column in
another model, or as an HTTP route added with `FastAPIRouter.add_query_route()`.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


class Orders(TableModel, name='orders'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    customer: pxt.String
    amount: pxt.Float


@pxt.query
def customer_total(customer: str) -> pxt.Query:
    return Orders.where(Orders.customer == customer).select(pxtf.sum(Orders.amount))


class Customers(TableModel, name='customers'):
    name = pxt.Column(type=pxt.String, primary_key=True)
    total = customer_total(name)
```

- A `@pxt.query` over models cannot join: `join(): a query over model ... cannot be joined`.

## Building a query

```python
pxt.create_dir('shop', if_exists='ignore')
orders = pxt.create_table(
    'shop.orders', {'id': pxt.Int, 'region': pxt.String, 'amount': pxt.Float, 'note': pxt.String | None}
)
orders.insert([{'id': 1, 'region': 'east', 'amount': 10.0}, {'id': 2, 'region': 'west', 'amount': 2.5, 'note': 'rush'}])
rows = (
    orders.where((orders.amount > 5) & (orders.note == None))
    .order_by(orders.amount, asc=False)
    .select(orders.id, doubled=orders.amount * 2)
    .limit(10)
    .collect()
)
by_region = orders.group_by(orders.region).select(orders.region, total=pxtf.sum(orders.amount)).collect()
sample = orders.select(orders.id, orders.region).sample(n=2, seed=0).collect()

regions = pxt.create_table('shop.regions', {'region': pxt.String})
regions.insert([{'region': 'east'}, {'region': 'north'}])


@pxt.query
def region_total(region: str) -> pxt.Query:
    return orders.where(orders.region == region).select(pxtf.sum(orders.amount))


totals = regions.select(regions.region, total=region_total(regions.region)).collect()
```

- Each method returns a new query; nothing runs before `collect()`, `count()`, `head()`, `tail()`, `show()` or
  `cursor()`.
- A query takes one `where()`; a second raises `where() clause already specified`. Combine conditions with `&`, `|`
  and `~`. A query also takes one `select()` and one `group_by()`.
- `col is None` is a Python `bool`, which `where()` rejects; write `col == None`. `pxt guide expressions` covers the
  operators.
- Known issue (PXT-1457): `where()` that compares an unstored computed column with a constant, as in
  `t.where(t.flag == True)`, raises `AssertionError`. A Bool column works alone (`t.where(t.flag)`, `~t.flag`), and
  writing the column's expression instead of the column works too.
- `order_by()` calls accumulate: later keys break ties. Nulls sort last in ascending order and first in descending
  order.
- Known issue (PXT-1514): `order_by()` on a UDF call does not sort, and raises `AttributeError` with `asc=False`.
  Order by a stored column instead.
- `limit(n, offset=k)` skips `k` rows. Without `select()`, a query returns every column.
- Result column names: a keyword argument sets the name, a column keeps its name, a function call gets the
  function's name, and any other expression gets `col_0`, `col_1`, ... by its position. A repeated name gets `_1`.
  Names are lowercased.
- `sample()` comes after `where()` and `select()`; calling `where()`, `order_by()`, `limit()`, `group_by()` or
  `join()` after `sample()` raises. The same `seed` returns the same rows; `stratify_by=` samples each stratum
  separately.

## Results

- `collect()` returns a `ResultSet`. `rows[0]` is a dict, `rows['doubled']` a list of the column's values, and
  `rows[0, 'doubled']` one value; iterating over `rows` yields dicts.
- Values come back as Python objects: an `Image` as a `PIL.Image.Image`, a `Video`, `Audio` or `Document` as a local
  file path, a `Timestamp` as an aware `datetime` in the default time zone, `Json` as dicts and lists, an `Array` as
  a `numpy.ndarray`.
- `rows.to_pandas()` returns a DataFrame and `rows.to_pydantic(Model)` yields model instances; a query has no
  `to_pandas()`. `cursor()` yields rows one at a time instead of loading them all.
- `count()` counts rows, or groups after `group_by()`. After `limit()` it raises `count() cannot be used with
  limit() or offset()`.
- `head(n)` and `tail(n)` follow insertion order; they refuse a query with `order_by()`, `group_by()`, `sample()` or
  a join.

## Aggregates and joins

- Aggregates include `pxtf.count()`, `pxtf.sum()`, `pxtf.mean()`, `pxtf.min()`, `pxtf.max()` and
  `pxtf.json.make_list()`, which collects values into a list. Without `group_by()`, an aggregate query returns one
  row.
- A column in the select list that is neither grouped nor aggregated raises `Invalid non-aggregate expression in
  aggregate query`.
- An aggregate with `group_by=` and `order_by=` arguments is a window function, with one value per row:
  `pxtf.sum(t.amount, group_by=t.region, order_by=t.id)` is a running total per region.
- `distinct()` removes duplicate rows. A computed column refuses an aggregate: `aggregate functions are not
  supported in computed columns`.
- `t.join(other, on=t.region == other.region, how='left')` joins two tables. `on=` also takes a column, matched by
  name in both tables. `how='cross'` takes no `on=`; every other join type requires it.
- A join without `select()` returns the columns of both tables, with a suffix on a repeated name (`region_1`).
- Known issue (PXT-1709): `join()` with a bound model on either side raises `AttributeError`; join the `pxt.get_table()`
  handles instead.

## `@pxt.query` functions

- The function runs once, when it is decorated, with placeholders for its parameters, and returns the query. Python
  logic on a parameter, such as `if region == 'east':`, raises.
- A parameter needs a type annotation (`Cannot infer pixeltable type for parameter ...`), or a type in
  `@pxt.query(param_types=[...])`. A parameter can appear in `where()`, in `limit(n)` and in `similarity()`
  (`pxt guide indexes`).
- Calling the function returns an expression, not rows. In `where()`, `select()` or a computed column, it runs once
  per row, as a separate query. For a value per group over many rows, such as each email's first signup, one
  `group_by()` query with an aggregate computes every group at once; match its rows to the table's rows by key.
  `pxt.eval_udf(region_total, 'east')` runs it on Python values and returns its result.
- The shape of the query sets what a call returns:

| The query | Selects one unnamed expression, as in `select(t.id)` | Any other select list |
|---|---|---|
| Aggregates without `group_by()`, or ends in a literal `limit(1)` | the value, or `None` | the row as a dict, or `None` |
| Any other query | a list of values | a list of dicts |

- `limit(n)` with a parameter returns a list, even when `n` is 1. `select(id=t.id)` returns dicts.
- A computed column that calls a `@pxt.query` runs it when the row is inserted or its arguments change. Later
  changes to the queried table do not update it. A `select()` that calls it sees the table's current rows.
- The query does not see the other rows of the same insert: for a query over the table itself, rows inserted
  together do not see each other, while a row inserted later sees them.
- Known issue (PXT-1736): a `@pxt.query` declared over models works in a model's computed column, in a route and in
  `pxt.eval_udf()`, but fails in the `select()` or `where()` of a query in a script, with `Table was dropped` or
  `declared over a model rather than a table`. For that use, declare the query over `pxt.get_table()` handles.
- A computed column stores the query itself, so a `@pxt.query` defined in a notebook works in one. After a change to
  the function's body, `pxt schema diff` reports the column's expression as changed, and existing rows keep their
  values.

```bash
pxt rows my_app/orders -n 5 --cols id,amount
pxt count my_app/orders
```
