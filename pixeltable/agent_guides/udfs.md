# UDFs

Defining UDFs and aggregates, how they treat nulls, batches and media files, and how they become LLM tools.

## Defining a UDF

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.func import Batch

TableModel = pxt.model_base()


@pxt.udf
def excerpt(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[:n] + '...'


@pxt.udf
def severity_label(severity: str | None) -> str:
    return 'unknown' if severity is None else severity.upper()


@pxt.udf(batch_size=16)
def word_counts(texts: Batch[str]) -> Batch[int]:
    return [len(text.split()) for text in texts]


class Tickets(TableModel, name='tickets'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    body: pxt.String | None
    severity: pxt.String | None
    preview = excerpt(body, n=20)
    label = severity_label(severity)
    words = word_counts(body)
```

- Every parameter and the return value need a type hint. Without one, `@pxt.udf` raises
  `Cannot infer pixeltable type for parameter 'x'` or `Cannot infer pixeltable return type`.
- The return hint sets the column type: `str` is `String`, `PIL.Image.Image` is `Image`, `pxt.Video` is `Video`,
  a TypedDict or `list[str]` is typed Json, and `dict` is untyped Json.
- Pixeltable records a UDF by its module path, such as `app.excerpt`, relative to the project root: the directory of
  `pixeltable.toml`, which `pxt init` writes. The daemon and the services import the module by that path.
- A UDF in the main module of a script raises `Defining the UDF 'f' directly in the global namespace of a Python
  script is not allowed`. A UDF nested in a function raises `nested functions are not supported`.
- When the module can no longer be imported, Pixeltable warns `The computed column 'preview' in table 'tickets' is no
  longer valid`. The stored values stay readable, and an insert fails with `the column 'preview' is currently invalid`.
- In a notebook, a UDF defined in a cell works for that session only; another process cannot load it.
- `pxt.eval_udf(excerpt, 'a long text', n=5)` calls a UDF on Python values, without a table (`pxt guide
  expressions`).
- `pxt.get_table()` inside a UDF raises `Catalog functions such as pxt.get_table() cannot run while a query or update
  executes`. Known issue (PXT-1711): `add_computed_column()` accepts such a column, and the next insert or recompute
  fails.

## Nulls

- A parameter typed `T` is non-nullable: for a null argument, Pixeltable skips the call and sets the cell to null
  without recording an error.
- A nullable argument for such a parameter makes the column nullable: `preview` is `String | None` even though
  `excerpt()` returns `str`.
- A parameter typed `T | None` receives the null, and the UDF handles it: `label` is `String`.
- Built-in functions skip calls the same way, and an aggregate skips a row with a null argument for a non-nullable
  `update()` parameter.

## Batched and async UDFs

- With `batch_size=N`, each `Batch[...]` parameter receives a list of up to N values, and the function returns a
  list of equal length.
- A parameter without `Batch[...]` takes a constant: a column fails with `parameter offset must be a constant value`.
- Pixeltable runs an `async def` UDF as one task per row, concurrently.
- `@pxt.udf(resource_pool='request-rate:my_api')` on an `async def` UDF limits its calls to the requests per minute
  in the environment variable `MY_API_RATE_LIMIT` (600 by default), and retries 408, 429, 503 and 504 errors up to 3
  times. A `[my_api]` section in `config.toml` fails with `Unrecognized option 'pixeltable.my_api'`. A UDF that is not
  `async def` fails with `resource_pool requires an async function`.

## Media results

- A UDF with an image return type returns a PIL image, and Pixeltable stores it as a file in its media store.
- A UDF that returns `pxt.Video`, `pxt.Audio` or `pxt.Document` returns a file path or URL. Pixeltable copies a file
  in a system temp directory, such as one from `tempfile.mkdtemp()`, into its media store, and records any other
  local path as it is, so that file has to stay in place.
- When the column sets `destination=`, Pixeltable copies the file there.

## Aggregates

```python
@pxt.uda
class mean_length(pxt.Aggregator):
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def update(self, text: str) -> None:
        self.lengths.append(len(text))

    def value(self) -> float | None:
        return sum(self.lengths) / len(self.lengths) if len(self.lengths) > 0 else None


by_severity = Tickets.group_by(Tickets.severity).select(Tickets.severity, avg_length=mean_length(Tickets.body))
```

- `__init__()` takes the constant arguments, `update()` takes one row's values, and the return hint of `value()`
  sets the result type.
- Aggregates run in queries; a computed column refuses one with `aggregate functions are not supported in computed
  columns`.
- An aggregate declared with `requires_order_by=True`, such as `pxtf.video.make_video`, takes the ordering expression
  as its first positional argument: `make_video(frames.pos, frames.frame, fps=25)`. `order_by=` raises
  `order_by invalid, this function requires the first argument to be the ordering expression`.

## Tools

```python
@pxt.udf
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f'{city}: 22C, sunny'


tools = pxt.tools(get_weather, pxt.tool(excerpt, name='shorten', description='Shortens a text.'))


class Assistant(TableModel, name='assistant'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    message: pxt.String
    response = pxtf.openai.chat_completions(
        messages=[{'role': 'user', 'content': message}], model='gpt-4o-mini', tools=tools
    )
    tool_output = pxtf.openai.invoke_tools(tools, response)
```

- A tool's name is the UDF's name, and its description is the UDF's whole docstring; `pxt.tool()` replaces either.
- A tool's non-nullable parameters are required.
- `pxt.tools()` takes UDFs, `@pxt.query` functions and `pxt.retrieval_udf()` results. An aggregate raises
  `Aggregator UDFs cannot be used as tools`.
- A tool's parameters can be `String`, `Int`, `Float`, `Bool`, `UUID` or Json; a parameter of another type fails.
- `openai`, `anthropic`, `gemini`, `bedrock` and `groq` each have an `invoke_tools()` for their own replies.
- `invoke_tools()` returns a dict with a key for every tool: the list of results when the model called it, and null
  when it did not, as in `{'get_weather': ['Paris: 22C, sunny'], 'shorten': None}`.
- `pxt.retrieval_udf(table, parameters=['sku'])` returns up to `limit` rows (10 by default) whose columns equal the
  arguments, as a list of dicts with every column; with `limit=1` it returns one dict, or null.
- `pxt.retrieval_udf()` takes a `Table` from `pxt.get_table()`, not a model class.
- `pxt.mcp_udfs(url)` connects to the MCP server at `url` over streamable HTTP and returns a UDF for each of its tools.

```bash
pxt errors my_app/tickets --col preview
pxt recompute my_app/tickets preview --errors-only -f
```
