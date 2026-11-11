# Json

Json columns: typed and untyped schemas, paths into them, the Json methods, and conversion to other types.

## Typed and untyped Json

```python
from typing import NotRequired, TypedDict

import pixeltable as pxt

TableModel = pxt.model_base()


class Stop(TypedDict):
    name: str
    delay: float
    platform: NotRequired[int]


class Trips(TableModel, name='trips'):
    id = pxt.Column(type=pxt.String, primary_key=True)
    stops: pxt.Json[list[Stop]]
    tags: pxt.Json[list[str]]
    payload: pxt.Json | None
    first_stop = stops[0].name
    last_delay = stops[-1].delay
    first_platform = stops[0].platform
    delays = stops.delay
    later_stops = stops[1:].name
    late_stops = stops.filter(lambda s: s.delay > 5.0)
    num_late = stops.filter(lambda s: s.delay > 5.0).len()
    worst_stop = stops.sort(lambda s: s.delay, asc=False)[0].name
    max_delay = stops.delay.max()
    rush = tags.contains('rush')
    author = payload.author.astype(pxt.String | None)
```

- `pxt.Json[...]` takes a TypedDict, a Pydantic model or `list[...]`. `pxt.Json` is untyped, and so is
  `pxt.Json[dict[str, str]]`, since a `dict` has no declared keys.
- An insert checks a typed value: `missing required key: 'delay'`, `unexpected key: 'extra'`, and
  `expected float, got int` (an int in a `float` field fails too).
- A path into a typed column has the declared type, made nullable: `first_stop` is `String | None`, `last_delay` is
  `Float | None`, and `later_stops` is a list of `String | None`.
- A missing key or an index past the end yields null, not an error. A key missing from a typed column's schema,
  such as a misspelled one, gives untyped `Json | None` and does not raise.
- A path into untyped Json, such as `payload.author`, is untyped Json.

## Paths

- `.name` and `['name']` select a key, `[0]` and `[-1]` an element, `[1:3]` a slice, and `['*']` every element.
- A key selected on a list applies to every element: `stops.delay` is the same as `stops['*'].delay`.
- A method name takes precedence over a key of the same name: `payload.values` without a call raises
  `payload.values is the method values(); call it to get a value, or use payload['values'] for the JSON field`.
- An attribute of the expression, such as `col_type`, also takes precedence. `payload['values']` and
  `payload['col_type']` select the keys.
- A key after a typed `String`, `Int` or `Float`, as in `stops[0].name.first`, raises at definition. In untyped Json,
  a key after a string yields null.

## Methods

- `map()`, `filter()` and `sort()` take a lambda over one element, as in `late_stops` above.
- `pxtf.map()`, `pxtf.filter()` and `pxtf.sort()` are the same functions.
- Inside the lambda, conditions combine with `&`, `|` and `~`.
- `sort()` without a lambda orders the elements by their own values; `asc=False` reverses the order.
- On lists: `len()`, `contains(value)`, `count(value)`, `flatten()`, `concat(other)` and `is_empty()`; on lists of
  numbers also `sum()` (0 for an empty list), `min()`, `max()` and `mean()` (null for an empty list).
- On objects: `get(key, default)`, `keys()`, `values()`, `items()`, `contains(key)` and `merge(other)`, in which
  `other` wins on a shared key.
- `len()` counts the elements of a list, the keys of an object or the characters of a string.
- On a typed object, `get(key)` has the declared type of `key`.
- `pxtf.json.dumps(value)` serializes a Json value to a `String`, and `pxtf.json.loads(s)` parses a JSON string; an
  invalid string fails that row.
- `pxtf.json.make_list(value)` is an aggregate that collects values into a list.

## Converting to other types

`astype()` sets the type of a Json value and checks the value when the expression runs. It converts only these
values:

| From | `astype()` to | Result |
|---|---|---|
| Int | `pxt.Float` | the number as a float |
| Int | `pxt.Bool` | `False` for 0, `True` otherwise |
| ISO 8601 String | `pxt.Timestamp`, `pxt.Date` | the parsed value; a timestamp without a time zone is in the default time zone |
| String | `pxt.UUID` | the parsed UUID |
| path or URL String | `pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document` | the media file |

- Any other mismatch raises when the value is computed, as in `Expected string, got dict`,
  `Expected string, got int` and `Expected int, got float`. `pxtf.json.dumps()` turns a dict or list into a `String`.
- `astype(pxt.String)` is non-nullable and raises `Expected non-None value` on a null. `astype(pxt.String | None)`
  passes the null through.
- String methods and `+` need a `String`. On untyped Json, `payload.title.upper()` selects a key `upper` and fails
  with `is not callable`, and `payload.title + '!'` fails with `+ requires numeric types`. Cast first:
  `payload.title.astype(pxt.String | None).upper()`.

```bash
pxt describe my_app/trips
```
