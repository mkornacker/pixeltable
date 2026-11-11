# CLI

Running `pxt` in a project: the project root, the daemon, applying an application file, confirmation and inspection.

## The project root

- `pxt init` makes the working directory the project root: it writes `pixeltable.toml` with a `[[pixeltable.database]]`
  entry for the local database. In a directory with a `pyproject.toml`, it appends `[[tool.pixeltable.database]]` to
  that file instead.
- In a directory that is already a project root, `pxt init` reports the configuration file and changes nothing.
- `pxt init` refuses, with exit code 3, a directory under another project root, the home directory or a directory above
  it, and the Pixeltable home.
- The project root is the nearest directory, from the working directory up, that has a `pixeltable.toml` or a
  `pyproject.toml` with a `[tool.pixeltable]` section. Run each command for a project from its root or below it.
- `pxt schema` and `pxt service` refuse an application file outside a project root. A UDF is recorded as its module
  path from the root (`app.excerpt` for `excerpt()` in `app.py`), so each directory between the root and the file needs
  a name that is a Python identifier.

## The daemon

- A command that needs the daemon starts it in the background if it is not running, on `127.0.0.1:22089` or on the
  port in `PXT_PORT`. Later commands send their requests to the same daemon.
- The daemon serves one project. A command in another project root restarts it for that project; a command outside any
  project uses it as it is.
- A command also restarts the daemon when the Pixeltable version, the Python interpreter, the installed packages or a
  `PIXELTABLE_*` variable differs from the daemon's. While the daemon serves a request, the command fails with
  `DAEMON_BUSY` instead.
- The daemon reads its other settings, such as `OPENAI_API_KEY` in the environment and the values in `config.toml`,
  once, at startup. After one of them changes, most commands fail with 409 `STALE_CONFIG` (`the daemon started with a
  different environment`, or `configuration has changed since the daemon started`) until `pxt daemon restart`;
  `pxt ls` and `pxt describe` still answer.
- `pxt daemon restart` refuses while the daemon serves a request; with `-f` it restarts anyway, which stops every
  request in progress.
- The daemon starts each local service as a separate process, with the daemon's environment, and the service keeps
  it: after `pxt daemon restart`, `pxt service restart my_app/ingest` restarts the service with the new values.

## Applying an application file

- `pxt schema check app.py` imports the file without reading the catalog, and checks that another process can import
  each UDF called by its columns. It warns about a project module that has the name of an installed package, and
  about a computed column that references no column, such as `label = str(title)`, which stores one value in every
  row.
- `pxt functions` lists functions without a daemon or catalog: `pxt guide expressions`.
- `pxt schema diff app.py my_app` prints the plan for the tables under `my_app`. `pxt schema update app.py my_app`
  applies it and starts no service; `pxt guide tables` says what each kind of change does.
- `pxt service update app.py my_app` starts each service of the file in a background process, on a port from the OS,
  and restarts a running service when its routes, the project's files or `--otel` changed. A restarted service
  keeps its port.
- `pxt service update` refuses a service while the tables differ from the models; it exits 1 with a
  `next: pxt schema update ...` line.
- `pxt service list` prints each service's endpoint, process id and routes; with `--json`, each service is an object
  with its `endpoint` and `port`.
- `pxt service logs` reads the logs of services in Pixeltable Cloud only. For a local service it exits 1 and gives the
  log file, `$PIXELTABLE_HOME/logs/services/my_app/ingest.log` for `my_app/ingest`.
- `pxt service update` leaves a service running when the file no longer defines it; `pxt service prune app.py my_app`
  stops it.
- `pxt service run app.py my_app` serves one service in the foreground and records nothing, so `pxt service list` and
  `pxt service stop` do not see it.
- `pxt recompute my_app/docs title_upper` recomputes stored values; `pxt guide tables` says which rows and columns.

## Confirmation and exit codes

- `pxt schema`, `pxt service` and `pxt db` exit with 0 when done or when nothing was pending, 1 on an error, and 2 when
  changes are pending, from a `diff` or from `-n` on an `update` or `prune`. Every command exits with 3 on a refusal.
- These commands ask for confirmation: `pxt service update` with a service to start or restart, even a first start;
  `pxt schema update` with a destructive change; the `prune` verbs, `pxt drop`, `pxt drop-dir`, `pxt recompute`,
  `pxt revert`, `pxt db update` and `pxt db delete`. `-f` skips the question. An additive `pxt schema update` does not
  ask.
- Without a terminal, a command that would ask exits 3 and prints a `next:` line on stderr with the command to rerun:
  `next: pxt service update app.py my_app -f  # start or restart 1 service(s)`.
- `pxt schema update` refuses to drop a column or an index, and `pxt service update` refuses to change or remove a
  route, unless given `--allow-destructive`. The refusal exits 3 and applies nothing, not even the additive changes;
  its `next:` line adds the flag.
- On a command that changes something, `-n` (`--dry-run`) prints the plan and applies nothing.
- With `--json`, the output is one JSON document on stdout. A failure adds `error_code`, `message` and `retryable` to
  it, plus `retry_after` and `next`, a list of `{"command": ..., "effect": ...}`, when they apply. The error message
  also goes to stderr.

## Inspecting the catalog

- A catalog path separates directories with `/`: `my_app/docs`. `pxt count my_app.docs` exits 2 with
  `pxt paths use '/' as the separator`.
- `pxt rows my_app/docs` prints the first 10 rows. It leaves out unstored computed columns, which it would have to
  compute, unless `--cols` lists them.
- `pxt get my_app/docs d1` looks up a row by primary key, and `pxt errors my_app/docs` lists the rows with a failed
  computed column; both need a table with a primary key.
- `pxt errors` ends with a `next:` line for each column whose failed cells can be recomputed.
- `pxt config` prints every setting with its value and its source: an environment variable, a file, or unset.
  Credentials are redacted.
- `pxt feedback --kind gap 'TEXT'` prints a report for the Pixeltable team and sends nothing; `--send` sends it. With
  `DO_NOT_TRACK` set to `1`, `true` or `yes`, `--send` sends nothing and exits 3.
- `pxt dashboard` opens the daemon's dashboard in a browser and prints its URL.

```bash
pxt init
pxt schema check app.py
pxt schema diff app.py my_app
pxt schema update app.py my_app
pxt service update app.py my_app -f
pxt service list my_app --json
pxt daemon restart
pxt service restart my_app/ingest
pxt service prune app.py my_app -n
pxt ls my_app
pxt describe my_app/docs
pxt rows my_app/docs -n 5 --cols id,title
pxt get my_app/docs d1
pxt count my_app/docs
pxt errors my_app/docs --col title_upper
pxt history my_app/docs -n 5
pxt config --section openai
pxt dashboard
```
