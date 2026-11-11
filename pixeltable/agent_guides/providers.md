# Providers

Calling model providers in computed columns: keys, request options, replies, embeddings, throttling and retries.

## Modules and keys

The provider modules under `pixeltable.functions` are `anthropic`, `bedrock`, `bfl`, `deepseek`, `fabric`, `fal`,
`fireworks`, `gemini`, `groq`, `huggingface`, `jina`, `llama_cpp`, `mistralai`, `nebius`, `ollama`, `openai`,
`openrouter`, `replicate`, `runwayml`, `together`, `twelvelabs`, `vllm`, `vlmrun`, `vlmrun_gateway`, `voyageai`,
`whisper`, `whisperx` and `yolox`. Importing a module does not need the provider's SDK package; using its functions
does.

- Pixeltable reads a provider's key from an environment variable or from `$PIXELTABLE_HOME/config.toml`:
  `OPENAI_API_KEY` or `api_key` in the `[openai]` section, `ANTHROPIC_API_KEY` or `[anthropic]`, and so on. The
  environment variable takes precedence.
- The variable and section names differ from the module name in four cases: `voyageai` reads `VOYAGE_API_KEY`,
  `mistralai` reads `MISTRAL_API_KEY`, `replicate` reads `REPLICATE_API_TOKEN` and `runwayml` reads
  `RUNWAYML_API_SECRET`.
- Without `GEMINI_API_KEY`, `gemini` leaves authentication to the Google GenAI client: `GOOGLE_API_KEY`, or Vertex AI
  with `GOOGLE_GENAI_USE_VERTEXAI=true` and `GOOGLE_CLOUD_PROJECT`.
- Without `BEDROCK_API_KEY`, `bedrock` uses boto3's default AWS credentials; `BEDROCK_REGION_NAME` sets the region.
- `ollama` connects to `OLLAMA_HOST` if it is set, and to the `ollama` package's default host otherwise.
- `OPENAI_BASE_URL` sends the `openai` functions to another OpenAI-compatible server, such as a LiteLLM proxy, and
  `DEEPSEEK_BASE_URL` does the same for `deepseek`; `base_url` in the `[openai]` or `[deepseek]` section works too.
- A missing key fails the call, not the column definition: `` `openai` client not initialized: parameter `api_key`
  is not configured``.
- The `pxt` daemon reads the environment when it starts; `pxt daemon restart` applies a changed variable.
- `pxt config --section openai` shows the source of each value.

## Calling a model

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()

embed = pxtf.openai.embeddings.using(model='text-embedding-3-small')


class Notes(TableModel, name='notes'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    body: pxt.String
    photo: pxt.Image
    reply = pxtf.openai.chat_completions(
        messages=[{'role': 'user', 'content': body}], model='gpt-4o-mini', model_kwargs={'temperature': 0.2}
    )
    summary = reply.choices[0].message.content
    caption = pxtf.openai.chat_completions(
        messages=[
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Describe the photo.'}, {'type': 'image_url', 'image_url': photo}],
            }
        ],
        model='gpt-4o-mini',
    )
    critique = pxtf.anthropic.messages(
        messages=[{'role': 'user', 'content': body}], model='claude-sonnet-5-5', max_tokens=1024
    )
    critique_text = critique.content.filter(lambda b: b.type == 'text')[0].text
    __indexes__ = [pxt.EmbeddingIndex(body, embedding=embed, name='body_idx')]
```

- Most functions take the model as `model=`. The exceptions are `model_id=` (`bedrock`, `huggingface`, `yolox`),
  `model_name=` (`twelvelabs`), `model_path=` or `repo_id=` with `repo_filename=` (`llama_cpp`), `app=` (`fal`) and
  `ref=` (`replicate`).
- Request options go in one dict: `model_kwargs=` for `openai`, `anthropic` and the OpenAI-compatible providers,
  `config=` for `gemini`, `inference_config=` for `bedrock.converse`, `options=` for `ollama`, and
  `sampling_params=` for `vllm`.
- An undeclared keyword argument fails at definition: `got an unexpected keyword argument 'temperature'`.
- `anthropic.messages` requires `max_tokens=`: `missing a required argument: 'max_tokens'`.
- In an OpenAI message, an image column goes in `{'type': 'image_url', 'image_url': photo}`, and
  `'image_url': {'url': photo}` works too; Pixeltable sends the image as a PNG data URL.
- OpenAI fetches a URL string, such as `photo.fileurl`, itself, so the URL has to be publicly reachable.
- `openai.responses` takes an image as `{'type': 'input_image', 'image_url': photo}`.
- `openai.vision()` is deprecated and warns with `PixeltableDeprecationWarning`; `chat_completions()` and
  `responses()` take images as above.

## Replies

- `openai.chat_completions`, `openai.responses` and `anthropic.messages` return typed Json with the fields of the
  SDK's `ChatCompletion`, `Response` and `Message`. A path into the reply has the field's type:
  `.choices[0].message.content` is `String | None`, so string functions and `+` apply to it directly.
- A key missing from the reply's type, such as a misspelled one, gives untyped `Json | None` and a null value;
  it does not raise. A column that already stores untyped replies keeps its type.
- The other chat functions, except those of `vllm`, return the provider's reply as untyped Json, and every path into
  it is untyped Json. String functions and `+` need a `String`: `.astype(pxt.String | None)` makes the text one, and
  `astype(pxt.String)` raises on a null.

These paths select the text of a reply:

| Function | Path to the text |
|---|---|
| `openai.chat_completions` | `.choices[0].message.content`, a `String \| None` |
| `openai.responses` | `.output_text`, a `String \| None` |
| `anthropic.messages` | `.content.filter(lambda b: b.type == 'text')[0].text`, a `String \| None` |
| `chat_completions` of `deepseek`, `fabric`, `fireworks`, `groq`, `mistralai`, `nebius`, `openrouter`, `together`, `vlmrun` and `vlmrun_gateway`; `llama_cpp.create_chat_completion` | `.choices[0].message.content` |
| `gemini.generate_content` | `.candidates[0].content.parts[0].text` |
| `bedrock.converse` | `.output.message.content[0].text` |
| `ollama.chat`, `ollama.generate` | `.message.content`, `.response` |
| `vllm.chat_completions`, `vllm.generate` | `.outputs[0].text` (typed `String`) |

## Embeddings

- An embedding function takes its model arguments through `.using()`, as `embed` above does; `.using()` rejects an
  unknown argument with `Unknown parameter`.
- An embedding index needs a vector of fixed length. `openai.embeddings` knows the length for
  `text-embedding-3-small`, `text-embedding-3-large` and `text-embedding-ada-002`; another model needs
  `model_kwargs={'dimensions': N}`, or the index fails with `it must return a 1-dimensional array of a specific
  length`.
- Binding `model_id` of `huggingface.sentence_transformer` or `huggingface.clip`, in `.using()` or in a call, loads
  the model to learn its vector length, and downloads it if needed.

## Throttling and retries

- `openai.chat_completions`, `openai.responses`, `openai.embeddings` and `anthropic.messages` throttle requests by the
  rate-limit headers of the provider's replies, with no configuration. They retry a failed request up to 10 times
  when the error is retryable, such as a rate limit or a timeout.
- Most other API functions send at most `rate_limit` requests per minute, set in the provider's config section:
  `GROQ_RATE_LIMIT` or `rate_limit` in `[groq]`, for example. The default is 600.
- For `bedrock` and the other `openai` functions, the limit is per model, with the model id as the key in
  `[bedrock.rate_limits]` or `[openai.rate_limits]`. For `together`, it is per kind of request, with the key `chat`,
  `embeddings` or `images` in `[together.rate_limits]`.
- These functions retry 408, 429, 503 and 504 errors and throttling errors up to 3 times, after the delay in a
  `Retry-After` header if there is one.
- The local functions, such as those of `huggingface`, `whisper`, `ollama` and `llama_cpp`, are neither throttled
  nor retried.
- A request that fails after its retries fails the insert; with `on_error='ignore'`, the row is stored and the
  cell records the error.

```bash
pxt config --section openai
pxt daemon restart
pxt errors my_app/notes --col reply
pxt recompute my_app/notes reply --errors-only -f
```
