# Indexes

Embedding indexes for similarity search, B-tree indexes for lookups, and how both stay current.

## Declaring indexes

An application lists a model's indexes in `__indexes__`; `pxt schema update` creates them with the table.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()
embed_text = pxtf.openai.embeddings.using(model='text-embedding-3-small')


class Articles(TableModel, name='articles'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    body: pxt.String
    category: pxt.String
    __indexes__ = [pxt.EmbeddingIndex(body, embedding=embed_text, name='body_idx'), pxt.BtreeIndex(category)]


@pxt.query
def search(q: str, k: int = 5) -> pxt.Query:
    sim = Articles.body.similarity(string=q)
    return Articles.where(sim > 0.3).order_by(sim, asc=False).select(Articles.id, score=sim).limit(k)
```

In a notebook, the table methods add indexes, to a table with or without rows:

```python
import numpy as np


@pxt.udf
def embed(text: str) -> pxt.Array[(16,), pxt.Float]:
    # a stand-in for a real embedding model
    vec = np.zeros(16, dtype=np.float32)
    for word in text.lower().split():
        vec[sum(word.encode()) % 16] += 1.0
    return vec


pxt.create_dir('kb', if_exists='ignore')
notes = pxt.create_table('kb.notes', {'id': pxt.Int, 'text': pxt.String, 'tag': pxt.String})
notes.insert([{'id': 1, 'text': 'red apple pie', 'tag': 'food'}, {'id': 2, 'text': 'blue sky', 'tag': 'sky'}])
notes.add_embedding_index('text', idx_name='text_idx', embedding=embed)
notes.add_embedding_index('text', idx_name='text_l2', embedding=embed, metric='l2')
notes.add_btree_index('tag')

sim = notes.text.similarity(string='red apple', idx='text_idx')
top = notes.order_by(sim, asc=False).limit(3).select(notes.text, score=sim).collect()
```

- `name=` sets an `EmbeddingIndex`'s name in a model; `add_embedding_index()` takes `idx_name=` instead.
  `EmbeddingIndex(idx_name=...)` raises `TypeError`.
- An index without a name gets one (`idx0`, `idx1`, ...). Index names are case-insensitive.
- In a model, two embedding indexes on one column need names: `column 'body' has multiple embedding indexes; they
  must be given explicit names`.
- `pxt.BtreeIndex` takes no name, and a column has at most one B-tree index.
- `add_embedding_index()` with the definition of an existing index raises `An identical embedding index already
  exists`; `if_exists='ignore'` skips it.

## Embedding functions

- An embedding function is a module-level UDF with one required parameter of the indexed column's type (`String`,
  `Image`, `Audio`, `Video` or `Document`), returning a fixed-length array such as `pxt.Array[(16,), pxt.Float]`.
  Any other function raises `The function ... is not a valid embedding`, with the reason.
- `.using(...)` binds the other parameters to constants:
  `pxtf.huggingface.sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2')`,
  `pxtf.huggingface.clip.using(model_id='openai/clip-vit-base-patch32')`,
  `pxtf.openai.embeddings.using(model='text-embedding-3-small')`. Without it, `clip` is not a valid embedding.
- `sentence_transformer.using()` and `clip.using()` load the model to find the embedding's length, so importing a
  file that calls them loads the model.
- `embedding=` covers each type accepted by one of the function's signatures: a CLIP index on an `Image` column
  answers `similarity(string=...)` and `similarity(image=...)`.
- `string_embed=`, `image_embed=`, `audio_embed=`, `video_embed=` and `document_embed=` set the function for one
  type, override `embedding=` for it, and raise if the function does not take that type.
- Each query type needs its own function in the index: with only `image_embed=`, `similarity(string=...)` raises
  `does not have a string embedding and does not support string queries`.
- An index on an `Array[(N,), Float]` column of precomputed vectors needs no function and answers only
  `similarity(vector=...)`.

## Searching

- `similarity()` takes the query value as one keyword: `string=`, `image=` (a path, URL or `PIL.Image.Image`),
  `audio=`, `video=` and `document=` (a path or URL), or `vector=` (a 1-dimensional float NumPy array of the
  index's length). A positional value is deprecated.
- The value is a constant or a `@pxt.query` parameter. A column raises `similarity(): requires a value, not an
  expression`.
- With several embedding indexes on the column, `idx=` picks one; without it, `similarity()` raises `Column 'text'
  has multiple embedding indices`.
- `metric='cosine'` (the default) scores by cosine similarity and `'ip'` by inner product: higher is closer, and
  `order_by(sim, asc=False)` lists the closest rows first. `'l2'` scores by Euclidean distance: lower is closer.
- `similarity()` cannot be the value of a computed column: evaluating it raises `similarity(): cannot be used in a
  computed column`. A snapshot has no indexes: `Snapshot does not support indices`.
- `notes.text.embedding(idx='text_idx')` selects the stored vectors.

## How an index stays current

- An insert computes the new rows' embeddings in the same operation. An embedding function that raises fails the
  insert; with `on_error='ignore'`, the row is stored, and its similarity is null.
- An update of the indexed column recomputes the row's embedding.
- Adding an index to a table with rows embeds every row in the same call, so a paid embedding model runs for each
  row.
- An index on a view's column, such as the `text` of a chunk view, embeds new view rows when a base insert creates
  them.
- In `__indexes__`, a new entry is additive: `pxt schema update` creates the index and embeds existing rows. A
  changed definition replaces the index and a removed entry drops it; both need `--allow-destructive`.
- `drop_embedding_index()` and `drop_index()` take `idx_name=`, or `column=` for a column with one index.

## B-tree indexes

- No B-tree index exists until one is declared, except with `has_default_idxs=True` on the table, which adds one to
  each eligible column.
- A B-tree index serves `where()` comparisons and `order_by()` on its column. It takes a stored column: a scalar
  other than `Bool`, or a media column that is neither computed nor an iterator output.
- The primary key has its own unique index, not listed in the table's index metadata.

```bash
pxt idxs my_app/articles
pxt idxs my_app --embedding
```
