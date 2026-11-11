"""Tests for 'pxt guide'."""

import pathlib

import pytest

import pixeltable
from pixeltable_cli.client.commands import guide

_GUIDES = sorted(
    path
    for path in (pathlib.Path(pixeltable.__file__).parent / 'agent_guides').glob('*.md')
    if path.name != 'README.md'
)


class TestGuide:
    def test_guide(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Without an area, the command lists each guide with the line after its title; with one, it prints that
        guide; an unknown area is a usage error that lists the areas."""
        guide.run([])
        listing = capsys.readouterr().out.splitlines()
        assert [line.split()[0] for line in listing] == [path.stem for path in _GUIDES]
        for line, path in zip(listing, _GUIDES):
            summary = next(text for text in path.read_text(encoding='utf-8').splitlines()[1:] if text.strip() != '')
            assert line.endswith(summary)

        guide.run(['tables'])
        assert capsys.readouterr().out == (_GUIDES[0].parent / 'tables.md').read_text(encoding='utf-8')

        with pytest.raises(SystemExit) as exc:
            guide.run(['tabels'])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "invalid choice: 'tabels'" in err
        assert 'tables' in err
