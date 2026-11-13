"""Tests for 'pxt lint', with a script in place of a coding agent as the reviewer."""

import json
import os
import pathlib
import shlex
import stat
import subprocess
import sys

import pytest

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import lint
from pixeltable_cli.client.commands.guide import guide_paths

_APP = 'import pixeltable as pxt\n\nTableModel = pxt.model_base()\n'
_REPORT = 'import app\n\nrows = app.Signups.collect()\nseen = set()\n'


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A project with an application file, a script in a subdirectory, and skipped directories."""
    for marker in [*utils.AGENT_MARKERS, lint.REVIEWER_ENV_VAR, 'PIXELTABLE_CONFIG']:
        monkeypatch.delenv(marker, raising=False)
    (tmp_path / 'home').mkdir()
    monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path / 'home'))
    (tmp_path / 'pixeltable.toml').write_text('[[pixeltable.database]]\n', encoding='utf-8')
    (tmp_path / 'app.py').write_text(_APP, encoding='utf-8')
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports' / 'report.py').write_text(_REPORT, encoding='utf-8')
    for skipped in ('.venv', '__pycache__', 'node_modules'):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / 'skipped.py').write_text('x = 1\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    return tmp_path


def reviewer(tmp_path: pathlib.Path, body: str) -> str:
    """Return the command of a reviewer script; the script saves its prompt in prompt.txt, then runs body."""
    script = tmp_path / f'reviewer_{len(list(tmp_path.glob("reviewer_*")))}.py'
    script.write_text(
        'import pathlib, sys, time\n'
        f'pathlib.Path({str(tmp_path / "prompt.txt")!r}).write_text(sys.stdin.read())\n' + body,
        encoding='utf-8',
    )
    return f'{shlex.quote(sys.executable)} {shlex.quote(str(script))}'


def answer(*findings: dict[str, object]) -> str:
    """Return a reviewer body that prints findings as JSON, wrapped in prose with braces and a code fence."""
    return (
        'print("I checked {app.py} first.")\nprint("```json")\n'
        f'print({json.dumps({"findings": list(findings)})!r})\nprint("```")\n'
    )


def pixeltable_reviewer(project: pathlib.Path) -> str | None:
    """Return the [lint] reviewer that Pixeltable's own Config reads from project's configuration."""
    # Pixeltable refuses a configuration file with an option it does not know
    loaded = subprocess.run(
        [
            sys.executable,
            '-c',
            'import pathlib, sys; from pixeltable.config import Config; '
            'config = Config({}, project_root=pathlib.Path(sys.argv[1])); '
            'print(config.get_string_value("reviewer", section="lint"))',
            str(project),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    reviewer = loaded.stdout.splitlines()[-1]
    return None if reviewer == 'None' else reviewer


def claude_on_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a PATH whose claude answers every review with no findings."""
    bin_dir = tmp_path_factory.mktemp('bin')
    (bin_dir / 'claude').write_text(
        f"#!/bin/sh\ncat > /dev/null\necho '{json.dumps({'findings': []})}'\n", encoding='utf-8'
    )
    (bin_dir / 'claude').chmod(stat.S_IRWXU)
    # the script needs cat
    return os.pathsep.join([str(bin_dir), '/usr/bin', '/bin'])


_FINDING = {'rule': 'R05', 'file': 'reports/report.py', 'line': 3, 'why': 'loops over rows', 'fix': 'use group_by()'}


class TestLint:
    def test_checklist(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a reviewer, the command prints the project's Python files and the rules, and exits 0; named paths
        restrict the files, and LINT_REVIEWER=none prints the checklist even with a reviewer in the project
        configuration."""
        rules = guide_paths()['review'].read_text(encoding='utf-8')
        lint.run([])
        out = capsys.readouterr().out
        assert 'Files:\n  app.py\n  reports/report.py\n\n' in out
        assert 'skipped.py' not in out
        assert out.endswith(rules)

        lint.run(['reports', '--json'])
        assert json.loads(capsys.readouterr().out) == {'reviewer': None, 'files': ['reports/report.py'], 'rules': rules}

        with open(project / 'pixeltable.toml', 'a', encoding='utf-8') as fp:
            fp.write(f'\n[lint]\nreviewer = {json.dumps(reviewer(project, "sys.exit(9)"))}\n')
        monkeypatch.setenv(lint.REVIEWER_ENV_VAR, 'none')
        lint.run(['app.py'])
        assert 'Files:\n  app.py\n\n' in capsys.readouterr().out

    def test_review(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reviewer gets the rules and the numbered lines of each file, and the command prints its findings with the
        rule's title and guide, and exits 2; with no findings, it exits 0."""
        with pytest.raises(SystemExit) as exc:
            monkeypatch.setenv(lint.REVIEWER_ENV_VAR, reviewer(project, answer(_FINDING)))
            lint.run([])
        assert exc.value.code == 2
        prompt = (project / 'prompt.txt').read_text(encoding='utf-8')
        assert guide_paths()['review'].read_text(encoding='utf-8') in prompt
        assert (
            '<file path="reports/report.py">\n    1  import app\n    2  \n    3  rows = app.Signups.collect()' in prompt
        )
        assert capsys.readouterr().out == (
            'reports/report.py:3: R05 A value from other rows computed in a Python loop\n'
            '    Why: loops over rows\n'
            '    Fix: use group_by()\n'
            '    Guide: pxt guide queries\n'
            '1 finding: fix each, or say why it does not apply.\n'
        )

        with pytest.raises(SystemExit) as exc:
            monkeypatch.setenv(lint.REVIEWER_ENV_VAR, reviewer(project, answer(_FINDING, {**_FINDING, 'line': 'n/a'})))
            lint.run(['--json'])
        assert exc.value.code == 2
        assert json.loads(capsys.readouterr().out) == {'findings': [_FINDING, {**_FINDING, 'line': None}]}

        monkeypatch.setenv(lint.REVIEWER_ENV_VAR, reviewer(project, answer()))
        lint.run([])
        assert capsys.readouterr().out == 'No findings.\n'

    def test_reviewer_precedence(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The environment variable comes first, then the project configuration and the home configuration."""
        in_home, in_project, from_env = (
            reviewer(project, answer({**_FINDING, 'rule': rule})) for rule in ('R04', 'R01', 'R02')
        )

        def reported_rule() -> str:
            with pytest.raises(SystemExit):
                lint.run(['--json'])
            return json.loads(capsys.readouterr().out)['findings'][0]['rule']

        (project / 'home' / 'config.toml').write_text(f'[lint]\nreviewer = {json.dumps(in_home)}\n', encoding='utf-8')
        assert reported_rule() == 'R04'
        with open(project / 'pixeltable.toml', 'a', encoding='utf-8') as fp:
            fp.write(f'\n[lint]\nreviewer = {json.dumps(in_project)}\n')
        assert reported_rule() == 'R01'
        monkeypatch.setenv(lint.REVIEWER_ENV_VAR, from_env)
        assert reported_rule() == 'R02'

    def test_reviewer_failures(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reviewer that fails, answers without the JSON, is missing or takes too long is an error, exit 1."""
        cases = [
            (reviewer(project, 'sys.stderr.write("no credits")\nsys.exit(3)'), 'the reviewer exited 3: no credits'),
            (reviewer(project, 'print("Looks fine to me.")'), 'has no {"findings": [...]} object: Looks fine to me.'),
            (reviewer(project, 'print(\'{"findings": [{"rule": "R05"}]}\')'), 'has no {"findings": [...]} object'),
            ('no-such-reviewer --flag', 'the reviewer command was not found'),
        ]
        for command, message in cases:
            monkeypatch.setenv(lint.REVIEWER_ENV_VAR, command)
            with pytest.raises(SystemExit) as exc:
                lint.run([])
            assert exc.value.code == 1
            assert message in capsys.readouterr().err

    def test_reviewer_timeout(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reviewer that does not answer within the timeout is an error, exit 1."""
        monkeypatch.setattr(lint, 'REVIEWER_TIMEOUT_SECS', 1)
        monkeypatch.setenv(lint.REVIEWER_ENV_VAR, reviewer(project, 'time.sleep(10)'))
        with pytest.raises(SystemExit) as exc:
            lint.run([])
        assert exc.value.code == 1
        assert 'did not answer within 1 s' in capsys.readouterr().err

    def test_agent_reviewer(
        self,
        project: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a configured reviewer, a run under Claude Code reviews with claude and writes it to the project
        configuration; under another agent, none, or without claude on PATH, it prints the checklist."""
        config_before = (project / 'pixeltable.toml').read_text(encoding='utf-8')
        monkeypatch.setenv('PATH', str(tmp_path_factory.mktemp('empty')))
        for agent_env in ({}, {'CODEX_SANDBOX': '1'}, {'CLAUDECODE': '1'}):
            with monkeypatch.context() as m:
                for name, value in agent_env.items():
                    m.setenv(name, value)
                lint.run([])
            assert capsys.readouterr().out.startswith('No reviewer ran')
        assert (project / 'pixeltable.toml').read_text(encoding='utf-8') == config_before

        monkeypatch.setenv('CLAUDECODE', '1')
        monkeypatch.setenv('PATH', claude_on_path(tmp_path_factory))
        command = lint.REVIEWER_PRESETS['claude']
        lint.run([])
        captured = capsys.readouterr()
        assert captured.out == 'No findings.\n'
        assert f'reviewing with {command} ([lint] reviewer in pixeltable.toml)' in captured.err
        assert pixeltable_reviewer(project) == command

        # the second run uses the configured reviewer
        lint.run([])
        assert capsys.readouterr() == ('No findings.\n', '')

        (project / 'pixeltable.toml').write_text(config_before + '\n[lint]\nreviewer = "none"\n', encoding='utf-8')
        lint.run([])
        assert capsys.readouterr().out.startswith('No reviewer ran')

    def test_agent_reviewer_config_files(
        self,
        tmp_path_factory: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pyproject.toml with a [tool.pixeltable.pixeltable] table gets [tool.pixeltable.lint]. One with the settings
        of [pixeltable] directly under [tool.pixeltable], as 'pxt init' writes it, has no place for [lint], and neither
        has a directory outside a project: there the review runs without writing a configuration."""
        for marker in [*utils.AGENT_MARKERS, lint.REVIEWER_ENV_VAR, 'PIXELTABLE_CONFIG']:
            monkeypatch.delenv(marker, raising=False)
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path_factory.mktemp('home')))
        monkeypatch.setenv('CLAUDECODE', '1')
        monkeypatch.setenv('PATH', claude_on_path(tmp_path_factory))
        command = lint.REVIEWER_PRESETS['claude']
        this_run_only = (
            f'pxt lint: reviewing with {command} (this run only). Each review runs that model on your account; '
            '[lint] reviewer = "none" in the project configuration turns it off.\n'
        )

        nested = tmp_path_factory.mktemp('nested')
        (nested / 'pyproject.toml').write_text(
            '[project]\nname = "shop"\n\n[[tool.pixeltable.pixeltable.database]]\n', encoding='utf-8'
        )
        (nested / 'app.py').write_text(_APP, encoding='utf-8')
        monkeypatch.chdir(nested)
        lint.run([])
        assert '([lint] reviewer in pyproject.toml)' in capsys.readouterr().err
        assert pixeltable_reviewer(nested) == command

        for flat_layout in ('[[tool.pixeltable.database]]\n', None):
            project = tmp_path_factory.mktemp('flat')
            if flat_layout is not None:
                (project / 'pyproject.toml').write_text(f'[project]\nname = "shop"\n\n{flat_layout}', encoding='utf-8')
            (project / 'script.py').write_text(_APP, encoding='utf-8')
            files_before = {path.name: path.read_text(encoding='utf-8') for path in project.iterdir()}
            monkeypatch.chdir(project)
            lint.run([])
            assert capsys.readouterr() == ('No findings.\n', this_run_only)
            assert {path.name: path.read_text(encoding='utf-8') for path in project.iterdir()} == files_before
            if flat_layout is not None:
                assert pixeltable_reviewer(project) is None

    def test_usage_errors(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing path, no Python files, or more source than a review takes is an error, exit 1."""
        with pytest.raises(SystemExit) as exc:
            lint.run(['missing.py'])
        assert exc.value.code == 1
        assert 'missing.py does not exist' in capsys.readouterr().err

        (project / 'docs').mkdir()
        with pytest.raises(SystemExit) as exc:
            lint.run(['docs'])
        assert exc.value.code == 1
        assert 'no Python files to review' in capsys.readouterr().err

        monkeypatch.setattr(lint, 'MAX_SOURCE_CHARS', 10)
        with pytest.raises(SystemExit) as exc:
            lint.run([])
        assert exc.value.code == 1
        assert 'more than 10 characters; name the files to review' in capsys.readouterr().err
