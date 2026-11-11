# Serving

Serving tables over HTTP with `FastAPIRouter` routes, run by `pxt service update` or mounted in a FastAPI app.

## Declaring routes

An application file declares routers next to its models. `pxt service update app.py TARGET` binds the models to the
tables under `TARGET` and runs each router as a service of its own, on its own port.

```python
import pydantic

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    id = pxt.Column(type=pxt.String, primary_key=True)
    title: pxt.String
    body: pxt.String | None
    cover: pxt.Image | None
    title_upper = pxtf.string.upper(title)
    thumbnail = cover.resize((64, 64))


ingest = FastAPIRouter(name='ingest', prefix='/api')
ingest.add_insert_route(Docs, path='/docs', inputs=['id', 'title', 'body'], outputs=['id', 'title_upper'])
ingest.add_insert_route(
    Docs, path='/upload', inputs=['id', 'title'], uploadfile_inputs=['cover'], outputs=['thumbnail'], background=True
)
ingest.add_update_route(Docs, path='/docs/update', inputs=['body'], outputs=['id', 'body'])
ingest.add_delete_route(Docs, path='/docs/delete')
ingest.add_compute_route(Docs, path='/preview', inputs=['title'], outputs=['title_upper'])


class Headline(pydantic.BaseModel):
    headline: str


@ingest.insert_route(Docs, path='/docs/headline', inputs=['id', 'title'], outputs=['title_upper'])
def headline(*, title_upper: str) -> Headline:
    return Headline(headline=f'{title_upper}!')
```

- Serving needs `pip install 'pixeltable[serve]'`. Without it the file still imports and its route definitions do
  nothing, so `pxt schema update` works.
- `name=` is the service name; without it, the service takes the name of the router's variable. `prefix=` comes before
  every path of the router, the media and job paths included.
- An insert, update, delete or compute route is a POST with one JSON field per input; a nullable input may be left
  out. With `uploadfile_inputs`, the body is multipart form data: a file per listed media column, form fields for the
  other inputs.
- An insert route stores one row and returns the `outputs`. Its inputs must include every non-nullable stored column.
- An update route finds the row by its primary key, which is always in the body; key and media columns cannot be
  inputs. An unknown key answers 404 `row not found`.
- A delete route deletes the rows whose primary key, or `match_columns=`, matches the body's values, and returns
  `{"num_rows": n}`.
- A compute route runs `Table.compute()` and stores nothing. On a view with an iterator it returns a JSON array;
  otherwise one object, or `null` when the view's filter drops the row.
- The decorators `insert_route()`, `update_route()` and `compute_route()` call the function with each output column
  as a keyword-only argument, a media column as a URL `str`, and send its pydantic model as the response.
- A path takes no `{parameter}`, and `/_pxt` is reserved. `add_insert_route()`, `add_update_route()` and
  `add_compute_route()` refuse Array and Binary columns in `outputs`.

## Query routes

```python
@pxt.query
def titles() -> pxt.Query:
    return Docs.order_by(Docs.title).select(Docs.title)


@pxt.query
def find_doc(title: str) -> pxt.Query:
    return Docs.where(Docs.title == title).select(Docs.id, Docs.body).limit(1)


ingest.add_query_route(path='/titles', query=titles, method='get')
ingest.add_query_route(path='/find', query=find_doc)
```

- The query's parameters are the inputs: JSON fields for POST, the default, and query-string parameters with
  `method='get'`.
- A query returns at most one row if it aggregates without `group_by()` or has a literal `limit(1)`. `limit(n)` with a
  parameter `n` returns many rows, even when `n` is 1.
- Many rows of one unnamed expression are a JSON array of values (`/titles`); other many-row queries return
  `{"rows": [...]}`. One row of one unnamed expression is the value itself; other one-row queries return an object
  (`/find`). `select(title=Docs.title)` gives the expression a name.
- A one-row query that finds no row answers 404 `query returned no rows`.

## Responses

- With `background=True`, a route answers at once with `{"id": ..., "job_url": ...}` and runs the request in a
  background thread. `job_url` is `{prefix}/_pxt/jobs/{name}/{id}`, or `{prefix}/_pxt/jobs/{id}` without `name=`.
- A GET on `job_url` returns `status`: `pending`, with a `Retry-After: 2` header; `done`, with the route's response in
  `result`; or `error`, with the error's `detail` object in `error_detail`.
- A finished job is kept for 10 minutes, or for 60 seconds after the first poll that returns its final status,
  whichever ends first; then a poll answers 404 `unknown or expired job`. Once the service instance that ran the job
  stops, a poll answers 410.
- A router accepts at most 64 unfinished jobs per service instance; the next request answers 503 `JOB_REFUSED`,
  retryable, with `Retry-After: 5`.
- A media output is a URL. A local service serves the file under `{prefix}/_pxt/media/...`; a service in Pixeltable
  Cloud returns a signed URL that is valid for one hour.
- `return_fileresponse=True` returns the file of the route's only output, which must be a media column, instead of
  JSON. A query route needs a one-row query for it, and it cannot be combined with `background=True`.
- A route answers a Pixeltable error with the HTTP status of its error code and
  `{"detail": {"error_code": ..., "message": ..., "retryable": ...}}`, plus `retry_after` and a `Retry-After` header
  when the error has a retry delay. A duplicate primary key is 422 `CONSTRAINT_VIOLATION`.
- A computed column that raises fails the request with 400 and stores nothing; `message` gives the exception.
- A service binds its tables when it starts. After `pxt schema update` changes a table, the insert, update, compute
  and delete routes on that table answer 409 `table schema changed since route was registered; please restart the
  service` until `pxt service update` restarts the service. Query routes keep answering; their select list is fixed
  when the service starts.
- On an insert, update or compute route, `export_sql=SqlExport(...)` (`from pixeltable.serving import SqlExport`)
  writes each response to an existing table in another SQL database after Pixeltable commits. If that write fails,
  the request answers 500, and the Pixeltable change stays.

## A FastAPI app of your own

```python
import fastapi

app = fastapi.FastAPI()
app.include_router(ingest)


@app.get('/api/stats')
def stats() -> dict[str, int]:
    return {'docs': Docs.count()}
```

- A file with a `fastapi.FastAPI` object is one service, named after its module (`app` for `app.py`). The object must
  include every router of the file, and at most one of those routers can have background routes and no `name=`.
- The service binds the file's models at `TARGET` before it starts, so a handler uses `Docs` directly.
- Declare a handler with `def`, not `async def`: FastAPI runs a `def` handler on a thread pool, and an `async def`
  handler that calls Pixeltable blocks the event loop.
- To include a router in an app that `pxt service` does not run, call `ingest.bind('my_app')` after
  `pxt schema update`, then `app.include_router(ingest)`. `bind()` fails if a table is missing or differs from its
  model, and an app with an unbound router fails at startup.

```bash
pxt service check app.py
pxt schema update app.py my_app
pxt service update app.py my_app -f
pxt service list my_app --json
pxt service restart my_app/app
```
