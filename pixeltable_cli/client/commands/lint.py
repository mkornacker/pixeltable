"""`pxt lint` - review a project's Python code against the rules of the agent guide 'review'."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from typing import Any, NoReturn

from pixeltable_cli.utils import PROJECT_CONFIG_FILE, PYPROJECT_FILE, find_project_root, parse_toml

from ..parser import Parser
from ..utils import AGENT_MARKERS, EXIT_CHANGES_PENDING, EXIT_ERROR, plural, print_error_document
from .guide import guide_paths

# by coding agent: a command that answers a prompt from stdin, with read-only tools
REVIEWER_PRESETS = {'claude': 'claude -p --model sonnet --tools Read,Grep,Glob --allowedTools Read,Grep,Glob'}
REVIEWER_ENV_VAR = 'PXT_LINT_REVIEWER'
REVIEWER_TIMEOUT_SECS = 600

SKIPPED_DIRS = {'__pycache__', 'node_modules', 'site-packages', 'venv', 'build', 'dist'}
MAX_SOURCE_CHARS = 400_000

EPILOG = f"""\
Examples:
  pxt lint                           # every Python file of the project
  pxt lint report.py app.py          # these files only
  pxt lint --setup                   # make your coding agent the reviewer
  pxt lint --reviewer none           # print the rules and the files, to review them yourself

Exit codes:
  0  no findings, or no reviewer: the command printed the rules and the files for you to review
  1  error: bad arguments, no reviewer for --setup, or the reviewer failed or did not answer as asked
  2  findings: fix each one, or say why it does not apply

Notes:
  'pxt guide review' prints the rules. A reviewer is a command that reads a prompt on stdin and answers on
  stdout, such as a coding agent's non-interactive mode. --reviewer sets it for one run; otherwise
  ${REVIEWER_ENV_VAR}, then 'reviewer' in the [lint] section of the project configuration, which --setup
  writes. Without a reviewer, the command prints the rules and the files for the coding agent that runs it.
  A review sends the files to the reviewer's model, on the reviewer's account."""

PROMPT = """\
You review the Python code of a project that uses Pixeltable {version}. Report only places that match one of the
rules below: code that does in Python what Pixeltable does, or that uses Pixeltable wrongly. Do not review style,
naming or general Python quality.

A place is a finding only if it matches a rule's "Look for" and none of its "Not a finding" cases. When a finding's
fix depends on how Pixeltable behaves, check it before you report it: the guides to this release are in
{guides_dir}, and the installed package is in {package_dir}. You may read files; change nothing.

<rules>
{rules}
</rules>

{files}

Answer with one JSON object and nothing else:
{{"findings": [{{"rule": "R05", "file": "report.py", "line": 14, "why": "...", "fix": "..."}}]}}
"why" says what this code does wrong, and "fix" what to change in it, in one or two sentences each. An empty list is
a valid answer."""

CHECKLIST = """\
No reviewer is configured, so review these files yourself against the rules below. For each place that matches a
rule, fix it, or say in your report why it does not apply. 'pxt lint --setup' has a separate model review them
instead; ask the user first, since it runs that model on their account.

Files:
{files}

{rules}"""


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt lint',
        description="review the project's Python code against the rules in 'pxt guide review'",
        epilog=EPILOG,
        usage_exit_code=EXIT_ERROR,
    )
    parser.add_argument(
        'paths', nargs='*', metavar='PATH', help='files or directories to review (default: the project)'
    )
    parser.add_argument(
        '--reviewer', metavar='COMMAND', help="the reviewer command for this run; 'none' for no reviewer"
    )
    parser.add_argument(
        '--setup', action='store_true', help='make the coding agent that runs this command the reviewer'
    )
    parser.add_argument('--json', action='store_true', dest='as_json', help='Emit JSON output')
    args = parser.parse_args(argv)

    root = find_project_root(pathlib.Path.cwd()) or pathlib.Path.cwd()
    if args.setup:
        _setup(root, args.as_json)
        return
    for path in args.paths:
        if not os.path.exists(path):
            parser.error(f'{path} does not exist')
    files = _python_files(root, [pathlib.Path(p) for p in args.paths])
    if len(files) == 0:
        _fail('there are no Python files to review', 'NO_FILES', args.as_json)
    sources = {_display(path, root): path.read_text(encoding='utf-8', errors='replace') for path in files}
    if sum(len(source) for source in sources.values()) > MAX_SOURCE_CHARS:
        _fail(
            f'the files have more than {MAX_SOURCE_CHARS} characters; name the files to review',
            'TOO_LARGE',
            args.as_json,
        )

    rules_path = guide_paths()['review']
    reviewer = args.reviewer if args.reviewer is not None else _configured_reviewer(root)
    if reviewer is None or reviewer == 'none':
        _print_checklist(list(sources), rules_path, args.as_json)
        return

    findings = _review(reviewer, sources, rules_path, root, args.as_json)
    rules = _rule_headings(rules_path)
    if args.as_json:
        print(json.dumps({'findings': findings}, indent=2))
    else:
        for f in findings:
            title, guide = rules.get(f['rule'], ('', ''))
            location = f['file'] if f.get('line') is None else f'{f["file"]}:{f["line"]}'
            print(f'{location}: {f["rule"]} {title}'.rstrip())
            print(f'    Why: {f["why"]}')
            print(f'    Fix: {f["fix"]}')
            if guide != '':
                print(f'    Guide: {guide}')
        print(
            'No findings.'
            if len(findings) == 0
            else f'{plural(len(findings), "finding")}: fix each, or say why it does not apply.'
        )
    if len(findings) > 0:
        sys.exit(EXIT_CHANGES_PENDING)


def _python_files(root: pathlib.Path, paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Return the named files, and the .py files under the named directories, or under root if none is named."""
    files: list[pathlib.Path] = []
    for path in paths if len(paths) > 0 else [root]:
        if path.is_file():
            files.append(path)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith('.') and d not in SKIPPED_DIRS)
            files += [pathlib.Path(dirpath) / name for name in sorted(filenames) if name.endswith('.py')]
    return list(dict.fromkeys(files))


def _display(path: pathlib.Path, root: pathlib.Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(root.resolve())) if resolved.is_relative_to(root.resolve()) else str(path)


def _configured_reviewer(root: pathlib.Path) -> str | None:
    if os.environ.get(REVIEWER_ENV_VAR, '') != '':
        return os.environ[REVIEWER_ENV_VAR]
    section = _lint_section(root)
    reviewer = section.get('reviewer') if section is not None else None
    return reviewer if isinstance(reviewer, str) and reviewer != '' else None


def _lint_section(root: pathlib.Path) -> dict[str, Any] | None:
    if (root / PROJECT_CONFIG_FILE).is_file():
        section = parse_toml(root / PROJECT_CONFIG_FILE).get('lint')
    elif (root / PYPROJECT_FILE).is_file():
        section = parse_toml(root / PYPROJECT_FILE).get('tool', {}).get('pixeltable', {}).get('lint')
    else:
        return None
    return section if isinstance(section, dict) else None


def _setup(root: pathlib.Path, as_json: bool) -> None:
    configured = _lint_section(root)
    if configured is not None and isinstance(configured.get('reviewer'), str):
        _report_setup(configured['reviewer'], None, as_json)
        return
    agent = next((name for marker, name in AGENT_MARKERS.items() if os.environ.get(marker, '') != ''), None)
    if agent is None or agent not in REVIEWER_PRESETS:
        known = ', '.join(sorted(REVIEWER_PRESETS))
        _fail(
            f'pxt lint has a reviewer command for these coding agents only: {known}. Set {REVIEWER_ENV_VAR} to a '
            'command that reads a prompt on stdin and answers on stdout, or pass --reviewer.',
            'NO_REVIEWER',
            as_json,
        )
    command = REVIEWER_PRESETS[agent]
    executable = shlex.split(command)[0]
    if shutil.which(executable) is None:
        _fail(f'{executable} is not on PATH, so it cannot review', 'NO_REVIEWER', as_json)
    if (root / PROJECT_CONFIG_FILE).is_file():
        config_file, header = root / PROJECT_CONFIG_FILE, '[lint]'
    elif (root / PYPROJECT_FILE).is_file():
        config_file, header = root / PYPROJECT_FILE, '[tool.pixeltable.lint]'
    else:
        _fail("no project configuration here; run 'pxt init' to write one", 'NO_PROJECT', as_json)
    text = config_file.read_text(encoding='utf-8')
    separator = '' if text == '' else '\n' if text.endswith('\n') else '\n\n'
    # json.dumps() of an ASCII string is a valid TOML basic string
    addition = f'{separator}{header}\nreviewer = {json.dumps(command)}\n'
    try:
        tomllib.loads(text + addition)
    except tomllib.TOMLDecodeError as e:
        _fail(f'{config_file.name} does not parse with {header} appended: {e}', 'INVALID_CONFIGURATION', as_json)
    with open(config_file, 'a', encoding='utf-8') as fp:
        fp.write(addition)
    _report_setup(command, config_file, as_json)


def _report_setup(command: str, written: pathlib.Path | None, as_json: bool) -> None:
    if as_json:
        print(json.dumps({'reviewer': command, 'written': None if written is None else str(written)}, indent=2))
    elif written is None:
        print(f'The project already has a reviewer: {command}')
    else:
        print(f'Wrote the reviewer to {written.name}: {command}')


def _print_checklist(files: list[str], rules_path: pathlib.Path, as_json: bool) -> None:
    rules = rules_path.read_text(encoding='utf-8')
    if as_json:
        print(json.dumps({'reviewer': None, 'files': files, 'rules': rules}, indent=2))
        return
    print(CHECKLIST.format(files='\n'.join(f'  {f}' for f in files), rules=rules), end='')


def _review(
    reviewer: str, sources: dict[str, str], rules_path: pathlib.Path, root: pathlib.Path, as_json: bool
) -> list[dict[str, Any]]:
    """Send the rules and the files to the reviewer command, and return its findings."""
    spec = importlib.util.find_spec('pixeltable')
    assert spec is not None and spec.submodule_search_locations is not None
    package_dir = pathlib.Path(next(iter(spec.submodule_search_locations)))
    files = '\n\n'.join(
        f'<file path="{name}">\n'
        + '\n'.join(f'{i:>5}  {line}' for i, line in enumerate(source.splitlines(), start=1))
        + '\n</file>'
        for name, source in sources.items()
    )
    prompt = PROMPT.format(
        version=importlib.metadata.version('pixeltable'),
        guides_dir=rules_path.parent,
        package_dir=package_dir,
        rules=rules_path.read_text(encoding='utf-8'),
        files=files,
    )
    try:
        result = subprocess.run(
            shlex.split(reviewer),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=REVIEWER_TIMEOUT_SECS,
            cwd=root,
            check=False,
        )
    except FileNotFoundError:
        _fail(f'the reviewer command was not found: {reviewer}', 'REVIEWER_FAILED', as_json)
    except subprocess.TimeoutExpired:
        _fail(f'the reviewer did not answer within {REVIEWER_TIMEOUT_SECS} s: {reviewer}', 'REVIEWER_FAILED', as_json)
    if result.returncode != 0:
        tail = result.stderr.strip()[-500:]
        _fail(
            f'the reviewer exited {result.returncode}: {tail if tail != "" else reviewer}', 'REVIEWER_FAILED', as_json
        )
    findings = _parse_findings(result.stdout)
    if findings is None:
        _fail(
            f'the reviewer did not answer with the JSON it was asked for: {result.stdout.strip()[:300]}',
            'REVIEWER_FAILED',
            as_json,
        )
    return findings


def _parse_findings(answer: str) -> list[dict[str, Any]] | None:
    """Parse the findings out of the reviewer's answer; None if the answer has no valid {"findings": [...]} object."""
    # a model may wrap the object in prose or a code fence
    start, end = answer.find('{'), answer.rfind('}')
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(answer[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get('findings'), list):
        return None
    findings: list[dict[str, Any]] = []
    for f in parsed['findings']:
        if not isinstance(f, dict) or not all(isinstance(f.get(key), str) for key in ('rule', 'file', 'why', 'fix')):
            return None
        line = f.get('line')
        findings.append({**f, 'line': line if isinstance(line, int) else None})
    return findings


def _rule_headings(rules_path: pathlib.Path) -> dict[str, tuple[str, str]]:
    """Return each rule's title and the 'pxt guide' commands of its Guide line, by rule id."""
    rules: dict[str, tuple[str, str]] = {}
    for section in re.split(r'^## ', rules_path.read_text(encoding='utf-8'), flags=re.MULTILINE)[1:]:
        rule_id, _, title = section.splitlines()[0].partition(' ')
        guide = re.search(r'^Guide: (.+)$', section, flags=re.MULTILINE)
        areas = [] if guide is None else [area.strip() for area in guide.group(1).split(',')]
        rules[rule_id] = (title, ', '.join(f'pxt guide {area}' for area in areas))
    return rules


def _fail(message: str, error_code: str, as_json: bool) -> NoReturn:
    if as_json:
        print_error_document(error_code, message, retryable=False)
    print(f'pxt lint: {message}', file=sys.stderr)
    sys.exit(EXIT_ERROR)
