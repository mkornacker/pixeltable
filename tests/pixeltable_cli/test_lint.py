"""Tests for 'pxt lint', with a script in place of a coding agent as the reviewer."""

import json
import os
import pathlib
import shlex
import stat
import sys
import tomllib

import pytest

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import lint
from pixeltable_cli.client.commands.guide import guide_paths

_APP = 'import pixeltable as pxt\n\nTableModel = pxt.model_base()\n'
_REPORT = 'import app\n\nrows = app.Signups.collect()\nseen = set()\n'


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A project with an application file, a script in a subdirectory, and skipped directories."""
    for marker in [*utils.AGENT_MARKERS, lint.REVIEWER_ENV_VAR]:
        monkeypatch.delenv(marker, raising=False)
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
    """Return a reviewer body that prints findings as JSON, wrapped in prose and a code fence."""
    return f'print("Here is the review:")\nprint("```json")\nprint({json.dumps({"findings": list(findings)})!r})\n'


_FINDING = {'rule': 'R05', 'file': 'reports/report.py', 'line': 3, 'why': 'loops over rows', 'fix': 'use group_by()'}


class TestLint:
    def test_checklist(self, project: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Without a reviewer, the command prints the project's Python files and the rules, and exits 0; named paths
        restrict the files, and '--reviewer none' prints the checklist even with a reviewer configured."""
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
        lint.run(['app.py', '--reviewer', 'none'])
        assert 'Files:\n  app.py\n\n' in capsys.readouterr().out

    def test_review(self, project: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A reviewer gets the rules and the numbered lines of each file, and the command prints its findings with the
        rule's title and guide, and exits 2; with no findings, it exits 0."""
        with pytest.raises(SystemExit) as exc:
            lint.run(['--reviewer', reviewer(project, answer(_FINDING))])
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
            lint.run(['--json', '--reviewer', reviewer(project, answer(_FINDING, {**_FINDING, 'line': 'n/a'}))])
        assert exc.value.code == 2
        assert json.loads(capsys.readouterr().out) == {'findings': [_FINDING, {**_FINDING, 'line': None}]}

        lint.run(['--reviewer', reviewer(project, answer())])
        assert capsys.readouterr().out == 'No findings.\n'

    def test_reviewer_precedence(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--reviewer comes first, then the environment variable, then the project configuration."""
        configured, from_env, from_flag = (
            reviewer(project, answer({**_FINDING, 'rule': rule})) for rule in ('R01', 'R02', 'R03')
        )
        with open(project / 'pixeltable.toml', 'a', encoding='utf-8') as fp:
            fp.write(f'\n[lint]\nreviewer = {json.dumps(configured)}\n')
        for env, argv, rule in ((None, [], 'R01'), (from_env, [], 'R02'), (from_env, ['--reviewer', from_flag], 'R03')):
            if env is not None:
                monkeypatch.setenv(lint.REVIEWER_ENV_VAR, env)
            with pytest.raises(SystemExit):
                lint.run(['--json', *argv])
            assert json.loads(capsys.readouterr().out)['findings'][0]['rule'] == rule

    def test_reviewer_failures(self, project: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A reviewer that fails, answers without the JSON, is missing or takes too long is an error, exit 1."""
        cases = [
            (reviewer(project, 'sys.stderr.write("no credits")\nsys.exit(3)'), 'the reviewer exited 3: no credits'),
            (reviewer(project, 'print("Looks fine to me.")'), 'did not answer with the JSON it was asked for'),
            (reviewer(project, 'print(\'{"findings": [{"rule": "R05"}]}\')'), 'did not answer with the JSON'),
            ('no-such-reviewer --flag', 'the reviewer command was not found'),
        ]
        for command, message in cases:
            with pytest.raises(SystemExit) as exc:
                lint.run(['--reviewer', command])
            assert exc.value.code == 1
            assert message in capsys.readouterr().err

    def test_reviewer_timeout(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lint, 'REVIEWER_TIMEOUT_SECS', 1)
        with pytest.raises(SystemExit) as exc:
            lint.run(['--reviewer', reviewer(project, 'time.sleep(10)')])
        assert exc.value.code == 1
        assert 'did not answer within 1 s' in capsys.readouterr().err

    def test_setup(
        self,
        project: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under Claude Code, --setup writes its reviewer command to the project configuration once; under another
        agent, or none, it is an error."""
        with pytest.raises(SystemExit) as exc:
            lint.run(['--setup'])
        assert exc.value.code == 1
        assert 'has a reviewer command for these coding agents only: claude' in capsys.readouterr().err

        monkeypatch.setenv('CODEX_SANDBOX', 'seatbelt')
        with pytest.raises(SystemExit) as exc:
            lint.run(['--setup'])
        assert exc.value.code == 1
        monkeypatch.delenv('CODEX_SANDBOX')

        monkeypatch.setenv('CLAUDECODE', '1')
        monkeypatch.setenv('PATH', str(tmp_path_factory.mktemp('empty')))
        with pytest.raises(SystemExit) as exc:
            lint.run(['--setup'])
        assert exc.value.code == 1
        assert 'claude is not on PATH' in capsys.readouterr().err

        bin_dir = tmp_path_factory.mktemp('bin')
        (bin_dir / 'claude').write_text('#!/bin/sh\n', encoding='utf-8')
        (bin_dir / 'claude').chmod(stat.S_IRWXU)
        monkeypatch.setenv('PATH', str(bin_dir))
        lint.run(['--setup'])
        command = lint.REVIEWER_PRESETS['claude']
        assert capsys.readouterr().out == f'Wrote the reviewer to pixeltable.toml: {command}\n'
        config = tomllib.loads((project / 'pixeltable.toml').read_text(encoding='utf-8'))
        assert config['lint'] == {'reviewer': command}
        lint.run(['--setup'])
        assert capsys.readouterr().out == f'The project already has a reviewer: {command}\n'

    def test_setup_pyproject(
        self,
        tmp_path: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In a project configured in pyproject.toml, --setup writes [tool.pixeltable.lint], and a review uses it."""
        for marker in [*utils.AGENT_MARKERS, lint.REVIEWER_ENV_VAR]:
            monkeypatch.delenv(marker, raising=False)
        (tmp_path / 'pyproject.toml').write_text('[project]\nname = "shop"\n\n[tool.pixeltable]\n', encoding='utf-8')
        (tmp_path / 'app.py').write_text(_APP, encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        bin_dir = tmp_path_factory.mktemp('bin')
        (bin_dir / 'claude').write_text(
            f"#!/bin/sh\ncat > /dev/null\necho '{json.dumps({'findings': []})}'\n", encoding='utf-8'
        )
        (bin_dir / 'claude').chmod(stat.S_IRWXU)
        # the script needs cat
        monkeypatch.setenv('PATH', os.pathsep.join([str(bin_dir), '/usr/bin', '/bin']))
        monkeypatch.setenv('CLAUDECODE', '1')
        lint.run(['--setup', '--json'])
        assert json.loads(capsys.readouterr().out) == {
            'reviewer': lint.REVIEWER_PRESETS['claude'],
            'written': str(tmp_path / 'pyproject.toml'),
        }
        config = tomllib.loads((tmp_path / 'pyproject.toml').read_text(encoding='utf-8'))
        assert config['tool']['pixeltable']['lint'] == {'reviewer': lint.REVIEWER_PRESETS['claude']}
        lint.run([])
        assert capsys.readouterr().out == 'No findings.\n'

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
