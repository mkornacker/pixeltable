# Cloud

Deploying an application file to Pixeltable Cloud with `pxt`: sign-in, hosted databases, secrets and logs.

## Signing in

- `pxt login` prints a code and a URL, and returns once the code is confirmed in a browser, which can run on another
  machine. It waits up to 10 minutes for the confirmation. The session is cached in the Pixeltable home; when its
  token expires, the next command renews it.
- An API key takes precedence over a session: `PIXELTABLE_API_KEY`, or `api_key` in the `[pixeltable]` section of the
  Pixeltable config file. A change to `PIXELTABLE_API_KEY` restarts the daemon at the next command (`pxt guide cli`).
- `pxt whoami` shows the credential in use, and whether Pixeltable Cloud recognizes it.
- Without a credential, a command that calls Pixeltable Cloud fails with `MISSING_CREDENTIALS`.

## The database entry

A hosted database is `pxt://ORG:DB`, such as `pxt://acme:main`. The organization and database names use lowercase
letters, digits and hyphens, start with a letter, end with a letter or digit, and have at most 29 characters. The
project configuration describes the database in a `[[pixeltable.database]]` entry of its own, after the local entry
from `pxt init`:

```toml
[[pixeltable.database]]

[[pixeltable.database]]
name = 'pxt://acme:main'
cpu = 2
memory_mb = 4096
workers = 1

[pixeltable.database.services.'my_app/ingest']
workers = 2
file_cache_gb = 5
```

- `pxt db update pxt://acme:main` uses the entry whose `name` is the URI. Without one, it fails and gives the command
  that appends it: `pxt db diff pxt://acme:main --add-config-entry`.
- `--add-config-entry` on `pxt db diff` or `pxt db update` appends the entry, with `name = 'pxt://acme:main'`, unless
  one with that name exists.
- The entry also selects the project files for the database and the contents of its image, and sets its capacity;
  `pxt db --help` describes the keys. An unknown key, such as `secrets`, makes the configuration fail to load.
- `services` sets the size of a hosted service, keyed by `'<directory>/<service name>'`, or by the name alone for a
  service at the database root. `pxt service update` applies it. The local entry cannot have `services`.
- `pxt://acme:main/my_app/docs` is the table `my_app/docs` in that database: `pxt.get_table()` takes it, and
  `pxt schema` and `pxt service` take `pxt://acme:main/my_app` as `TARGET`.

## Deploying

A deployment runs `pxt db update`, then `pxt schema update`, then `pxt service update`.

- `pxt db update` uploads the project files, rebuilds the image when `uv.lock`, `requirements.txt`, `pyproject.toml`,
  the Python version, the system packages or the Pixeltable version changed, and applies the settings and capacity.
  It creates no tables and starts no services. The first `pxt db update` of a URI creates the database.
- The upload contains the project files that `.gitignore` does not exclude, adjusted by the entry's `include` and
  `exclude`, plus `uv.lock`, `requirements.txt` and `pyproject.toml`. Without `uv.lock` or `requirements.txt`,
  `pxt db update` warns that the image has Pixeltable and no other package.
- A lower `cpu`, `memory_mb`, `disk_gb` or `workers` is destructive and needs `--allow-destructive`.
- `pxt db update` asks for confirmation, then waits until the change is done, showing its progress on stderr.
- Ctrl-C (exit 130) or SIGTERM (exit 143, which a tool timeout sends) stops only the waiting: Pixeltable Cloud
  finishes the change, the command prints `next: pxt db status pxt://acme:main`, and rerunning the update reports
  the change. After three hours the command stops waiting and exits 2. `--no-wait` returns once Pixeltable Cloud
  accepts the change.
- A failed update exits 1 with `next: pxt db retry pxt://acme:main`; `pxt service retry` retries a failed service.
- A change to the entry's settings alone, such as `vars` or a provider section, is applied without uploading the
  project.
- `PAYMENT_REQUIRED`, `PLAN_LIMIT` and `STORE_FULL` print Pixeltable Cloud's reason as it is written; retrying does
  not help until the organization's plan, payment or storage changes.
- `pxt schema update app.py pxt://acme:main/my_app` exits 1 without a change while a file that defines a UDF called
  by the models differs from its uploaded copy; its `next:` line gives the `pxt db update` command.
- `pxt service update app.py pxt://acme:main/my_app` refuses a service before the first `pxt db update`, and while the
  project differs from what `pxt db update` last applied. It waits until each started service answers requests.
- `pxt service restart pxt://acme:main/my_app/ingest` restarts a hosted service on its current code;
  `pxt service update` moves it to the database's latest code.
- `pxt service run` refuses a `pxt://` target: it serves from the local process only.

## Secrets

- A provider key is a secret: `pxt secret set pxt://acme OPENAI_API_KEY=...` sets it for every database of the
  organization, and `pxt secret set pxt://acme:main ...` for one database, whose secret wins over the organization's.
- A secret never goes in the project configuration. For a hosted database, an `api_key` or another credential in any
  section of `pixeltable.toml` makes `pxt db update` and `pxt service update` fail; the error gives the
  `pxt secret set` command to run instead.
- A process reads its secrets when it starts. After `pxt secret set` or `pxt secret delete`, `pxt db restart` restarts
  the database and `pxt service restart` a service; `pxt secret set` prints a `next:` line for each one that
  is running.
- `pxt secret list` prints the names of the secrets, never their values.

## Media

- A hosted database stores inserted and computed media in its home bucket, `pxt://acme:main/buckets/home`, unless a
  column sets `destination=` or the database entry sets `db_input_media_dest` or `db_output_media_dest`.
- A media cell's `fileurl` is `pxt://acme:main/buckets/home/<key>`; the key is the part after `buckets/home/`.
- `https://storage.pixeltable.com/acme/main/buckets/home/<key>`, with each segment of the key URL-encoded, opens the
  file after a Pixeltable sign-in. A script sends the API key as `Authorization: Bearer` and follows the redirect;
  `X-api-key` is refused there, because clients send it on to the redirect target.
- `POST https://storage.pixeltable.com/acme/main/sign`, with the API key as `X-api-key` or `Authorization: Bearer`
  and the body `{"keys": [...], "expires_in": 3600}`, returns `{"urls": {key: url}, "expires_at": ...}`. It signs 1
  to 100 keys, for 1 second to 7 days, 3600 seconds by default. Anyone with a signed URL can read the file until it
  expires.
- A service route returns a media output as a URL signed for one hour (`pxt guide serving`).

## Calling services and reading logs

- `pxt service list pxt://acme:main` prints the URL of each service in the database.
- A hosted service's OpenAPI schema requires a Pixeltable API key in the `X-api-key` header on every route except
  `/health`.
- `pxt service logs pxt://acme:main/my_app/ingest` merges the service's log records, requests included, with its
  console output, which has the traceback of a failed start. A line appears a few seconds after it is written.
- `pxt db logs pxt://acme:main` reads the log of the database's pod. Both commands read the last hour by default.

```bash
pxt login
pxt whoami
pxt db diff pxt://acme:main --add-config-entry
pxt db update pxt://acme:main -f
pxt schema update app.py pxt://acme:main/my_app
pxt service update app.py pxt://acme:main/my_app -f
pxt service list pxt://acme:main
pxt secret set pxt://acme:main OPENAI_API_KEY=sk-example
pxt db restart pxt://acme:main
pxt service restart pxt://acme:main/my_app/ingest
pxt service logs pxt://acme:main/my_app/ingest --since 10m --tail 50
pxt db logs pxt://acme:main
pxt db status pxt://acme:main
```
