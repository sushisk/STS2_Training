"""A repeated CLI option is refused rather than silently last-wins.

argparse keeps the last occurrence and says nothing. A duplicated ``--rl-root`` therefore
evaluates a different checkout than the one the reader of the command believes, which is
exactly what happened: two whole-run evaluations measured a stale worktree, and their
failures were investigated as engine bugs before anyone checked which tree had served
them. No option here means anything when passed twice, so it is a hard stop.
"""

from __future__ import annotations

import pytest

from tools.evaluate_whole_run import _parse_args


def test_a_repeated_option_is_refused_and_names_itself():
    with pytest.raises(SystemExit) as excinfo:
        _parse_args([
            "--character-id", "IRONCLAD",
            "--rl-root", "C:/one",
            "--rl-root", "C:/two",
        ])
    message = str(excinfo.value)
    assert "--rl-root" in message
    assert "x2" in message


def test_equals_form_counts_as_the_same_option():
    with pytest.raises(SystemExit) as excinfo:
        _parse_args(["--character-id", "IRONCLAD", "--num-runs", "3", "--num-runs=4"])
    assert "--num-runs" in str(excinfo.value)


def test_a_repeated_value_is_not_mistaken_for_a_repeated_option():
    # Two options may legitimately share a value; only the flags are counted.
    args = _parse_args([
        "--character-id", "IRONCLAD",
        "--output-dir", "same",
        "--detailed-log-dir", "same",
    ])
    # argparse converts these to Path; compare as such rather than as text.
    assert args.output_dir.name == "same"
    assert args.detailed_log_dir.name == "same"


def test_an_ordinary_command_still_parses():
    args = _parse_args(["--character-id", "IRONCLAD", "--num-runs", "10", "--rl-root", "C:/rl"])
    assert args.character_id == "IRONCLAD"
    assert args.num_runs == 10
    assert str(args.rl_root).replace(chr(92), "/") == "C:/rl"
