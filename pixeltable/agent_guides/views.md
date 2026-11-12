# Views

Filtered views and iterator views: tables whose rows derive from a base table and stay current as it changes.

## Declaring a view

A model with `base=` declares a filtered view; adding `iterator=` declares an iterator view, with one row per
chunk, frame, segment or element of each base row.

```python
# ruff: noqa: F821
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions.document import document_splitter

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    doc: pxt.Document
    title: pxt.String
    score: pxt.Int
    title_upper = pxtf.string.upper(title)


class TopDocs(TableModel, name='top_docs', base=Docs.where(Docs.score >= 10)):
    headline = title_upper + '!'
    reviewer: pxt.String | None


class Chunks(
    TableModel,
    name='chunks',
    base=Docs,
    iterator=document_splitter(Docs.doc, separators='page,token_limit', limit=300, metadata='page'),
):
    n_chars = text.len()
```

- `base=` takes a model or a query on a model, such as `Docs.where(...)`.
- `iterator=` requires `base=`.
- In the class body, the base's columns and the iterator's outputs are available by name: `title_upper` and `text`
  above.
- Redeclaring a base column or an iterator output in the class body raises an error such as `'text' is already
  defined by the iterator; it cannot be redeclared`.
- Linters report these names as undefined.
- For ruff, `# ruff: noqa: F821` at the top of the file silences the undefined-name reports.
- For mypy, `plugins = pixeltable.mypy.mypy_plugin` in the mypy configuration lets a model class take the keywords
  `name=`, `base=` and `iterator=`.
- For mypy, `# type: ignore[name-defined]` silences the undefined-name error on each line that uses a base column or
  an iterator output.
- A view's own input columns must be nullable: `Non-computed columns in views must be nullable`.

In a notebook, `pxt.create_view()` creates a view from a table or a query:

```python
from pixeltable.functions.json import list_iterator
from pixeltable.functions.video import frame_iterator

pxt.create_dir('media', if_exists='ignore')
posts = pxt.create_table('media.posts', {'id': pxt.Int, 'tags': pxt.Json[list[str]], 'score': pxt.Int})
posts.insert([{'id': 1, 'tags': ['a', 'b'], 'score': 12}, {'id': 2, 'tags': [], 'score': 3}])
top_posts = pxt.create_view('media.top_posts', posts.where(posts.score >= 10))
tags = pxt.create_view('media.tags', posts, iterator=list_iterator(tag=posts.tags))
tags.add_computed_column(tag_upper=tags.tag.upper())
tag_counts = tags.group_by(tags.id).select(tags.id, n=pxtf.count(tags.pos)).collect()

videos = pxt.create_table('media.videos', {'video': pxt.Video})
frames = pxt.create_view('media.frames', videos, iterator=frame_iterator(videos.video, fps=1.0))
```

- Creating a view on a base that has rows fills the view in the same call.
- A view's columns are the base's columns plus its own.
- A `select()` in the base query replaces the base's columns with the selected ones.
- In a model, each item of the base query's `select()` must be a column or a keyword argument, such as
  `select(n=...)`.
- Grouping an iterator view by the base's key (`tags.group_by(tags.id)`) aggregates per base row.

## Iterators

Every iterator view also has `pos`, the row's position within its base row, starting at 0.

| Iterator | Module | Output columns |
|---|---|---|
| `document_splitter()` | `pixeltable.functions.document` | `text`; `image` with `elements=['text', 'image']`; `title`, `heading`, `sourceline`, `page`, `bounding_box` as listed in `metadata=` |
| `string_splitter()` | `pixeltable.functions.string` | `text` |
| `frame_iterator()` | `pixeltable.functions.video` | `frame` (not stored), `frame_attrs` (`index`, `pts`, `dts`, `time`, `is_corrupt`, `key_frame`, `pict_type`, `interlaced_frame`) |
| `legacy_frame_iterator()` | `pixeltable.functions.video` | `frame` (not stored), `frame_idx`, `pos_msec`, `pos_frame` |
| `video_splitter()` | `pixeltable.functions.video` | `segment_start`, `segment_start_pts`, `segment_end`, `segment_end_pts`, `video_segment` |
| `audio_splitter()` | `pixeltable.functions.audio` | `segment_start`, `segment_end`, `audio_segment` |
| `tile_iterator()` | `pixeltable.functions.image` | `tile` (not stored), `tile_coord`, `tile_box` |
| `list_iterator()` | `pixeltable.functions.json` | one column per keyword argument, or one per key of the TypedDict of the list's elements |
| `sam3_for_video_segmentation()` | `pixeltable.functions.huggingface` | `frame` (not stored), `frame_attrs`, `object_ids`, `labels`, `scores`, `boxes`, `masks` |

- A column that is not stored is decoded again from the base's video or image whenever a query reads it.
- `frame_attrs.time` is the frame's timestamp in seconds.
- For a PDF with `'page'` in `separators`, `page` is the page number, starting at 0; in every other case, `page` is
  null.
- `heading` applies to HTML and Markdown.
- `document_splitter()` with `separators='paragraph'` fails on a PDF: `Paragraph splitting is not currently
  supported for PDF documents`.
- `separators='sentence'` in `document_splitter()` and `string_splitter()` needs spaCy and a spaCy model:
  `en_core_web_sm`, or the model in `spacy_model=`.
- A missing spaCy model raises `Failed to locate spaCy model`.
- The `'token_limit'` separator needs `tiktoken`.
- Creating a `document_splitter()` view checks the requirements of `'sentence'` and `'token_limit'`.
- `document_splitter()` extracts images only from a PDF with `separators='page'`.
- `string_splitter()` supports only `separators='sentence'`.
- `video_splitter()` needs `ffmpeg` on the `PATH`: `ffmpeg is not installed or not in PATH`.
- `list_iterator()` needs typed Json, such as `pxt.Json[list[str]]`; on untyped Json, it raises
  `list_iterator(): Expected a type for ...`.
- `list_iterator()` produces no rows for a null list.

## How a view follows its base

- An insert into the base adds the view rows and computes their columns in the same operation.
- `status.num_rows` in the result of `insert()` includes the view rows.
- An update that makes a base row fail the filter removes its view row.
- An update to the iterator's input replaces the view rows of each updated base row.
- Deleting a base row removes its view rows.
- `insert()` on a view raises `Cannot insert into a view`, and `delete()` on a view raises `Cannot delete from a view`.
- `update()` on a view changes only the view's own columns: updating a base column raises `Column 'score' is a base
  table column and cannot be updated`.
- The values that `update()` sets on a view survive updates to the base row.
- Dropping a base with views raises an error listing them; `pxt drop PATH --cascade` drops the views too.

## What a view may reference

- A view's computed columns may use the base's columns (computed ones included), the iterator's outputs and the
  view's earlier columns.
- Referencing a column of another table raises `The value expression of column ... is not bound by view ...`.
- A base table cannot reference its views.
- A filtered or iterator view of another view works the same way as a view of a table.
- A view's base query takes no `group_by()`, `order_by()`, `limit()`, `join()` or aggregate: `` `group_by` cannot be
  used in a view definition``.
- For sampling, a view's base query accepts only `sample(fraction=...)` without `stratify_by=`.

## Snapshot views

- `pxt.create_snapshot()` freezes the rows of a query, as in
  `pxt.create_snapshot('media.top_v1', posts.where(posts.score >= 10))`.
- `pxt.create_view(..., is_snapshot=True)` also creates a snapshot of a query.
- `pxt guide tables` covers snapshots of whole tables.
- A snapshot's query accepts `sample(n=...)` and `stratify_by=`.
- At creation, a snapshot accepts `additional_columns=`.
- After creation, Pixeltable refuses new columns and indexes on a snapshot, with errors such as `Cannot add columns
  to a snapshot.`
- `similarity()` on a snapshot raises `Snapshot does not support indices`.
- A model cannot declare a snapshot.

```bash
pxt describe my_app/chunks
pxt ls my_app --tree
pxt drop my_app/docs --cascade -n
```
