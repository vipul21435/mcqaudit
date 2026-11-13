import json
from pathlib import Path

import pytest

from mcqaudit.cli import main

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample.jsonl"


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def small(tmp_path):
    rows = [
        {
            "question": f"q{i}",
            "subject": "s" if i % 2 else "t",
            "choices": ["a", "bb", "c"],
            "answer": i % 3,
        }
        for i in range(12)
    ]
    return write_jsonl(tmp_path / "small.jsonl", rows)


def test_audit_markdown(small, capsys):
    assert main(["audit", str(small)]) == 0
    out = capsys.readouterr().out
    assert "# Answer-key audit: small.jsonl" in out
    assert "exact multinomial" in out
    assert "## By subject" in out


def test_audit_json_to_file(small, tmp_path):
    out = tmp_path / "audit.json"
    assert main(["audit", str(small), "--format", "json", "-o", str(out), "--no-subjects"]) == 0
    data = json.loads(out.read_text())
    assert data["n_items"] == 12
    assert data["positions"]["strata"][0]["counts"] == [4, 4, 4]
    assert data["subjects"] == []


def test_cues_json(small, capsys):
    assert main(["cues", str(small), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in data["cues"]}
    assert {"always A", "longest option", "shortest option"} <= names


def test_cues_markdown_on_the_example(capsys):
    assert main(["cues", str(SAMPLE)]) == 0
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("| ")][1:]
    assert any(r.startswith("| longest option |") for r in rows[:2])


def test_bad_file_exits_with_2(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "q"}\n')
    assert main(["audit", str(bad)]) == 2
    assert "no options" in capsys.readouterr().err
    assert main(["audit", str(tmp_path / "missing.jsonl")]) == 2


def test_answer_base_flag_and_note(tmp_path, capsys):
    rows = [{"question": "q", "choices": ["a", "b"], "answer": a} for a in (1, 2, 2, 1)]
    path = write_jsonl(tmp_path / "one.jsonl", rows)
    assert main(["audit", str(path), "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert "1-based" in captured.err
    assert json.loads(captured.out)["positions"]["strata"][0]["counts"] == [2, 2]
    assert main(["audit", str(path), "--answer-base", "0"]) == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "mcqaudit" in capsys.readouterr().out


def test_choices_only_command(small, capsys):
    assert main(["choices-only", str(small), "--folds", "2", "--features", "64"]) == 0
    out = capsys.readouterr().out
    assert "# Choices-only classifier" in out
    assert main(["choices-only", str(small), "--folds", "3", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_items"] == 12 and data["n_folds"] == 3


def test_hygiene_command_with_against(small, tmp_path, capsys):
    other = write_jsonl(
        tmp_path / "other.jsonl",
        [{"question": "q3", "choices": ["x", "y"], "answer": 0}],
    )
    assert main(["hygiene", str(small), "--against", str(other)]) == 0
    out = capsys.readouterr().out
    assert "## Overlap with the other file" in out
    assert "1 match a question of the other file" in out
    assert main(["hygiene", str(small), "--against", str(other), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["overlap"]["exact"] == 1
    assert main(["hygiene", str(small), "--against", str(tmp_path / "nope.jsonl")]) == 2


def test_shuffle_command(small, tmp_path, capsys):
    out = tmp_path / "shuffled.jsonl"
    assert main(["shuffle", str(small), "-o", str(out), "--seed", "1"]) == 0
    assert "wrote 12 items" in capsys.readouterr().err
    original = [json.loads(line) for line in small.read_text().splitlines()]
    shuffled = [json.loads(line) for line in out.read_text().splitlines()]
    for a, b in zip(original, shuffled, strict=True):
        assert a["choices"][a["answer"]] == b["choices"][b["answer"]]
    bal = tmp_path / "balanced.jsonl"
    assert main(["shuffle", str(small), "-o", str(bal), "--balanced"]) == 0
    assert main(["shuffle", str(small)]) == 2


@pytest.mark.parametrize("command", ["audit", "cues", "choices-only", "hygiene"])
def test_bad_answer_letter_is_an_error_line(tmp_path, capsys, command):
    path = write_jsonl(
        tmp_path / "x.jsonl", [{"question": "q", "choices": ["a", "b"], "answer": "É"}]
    )
    assert main([command, str(path)]) == 2
    assert "not A to Z" in capsys.readouterr().err


def test_fullwidth_letter_is_read(tmp_path, capsys):
    path = write_jsonl(
        tmp_path / "x.jsonl", [{"question": "q", "choices": ["a", "b"], "answer": "Ｂ"}]
    )
    assert main(["audit", str(path), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["positions"]["strata"][0]["counts"] == [0, 1]


def test_choices_only_on_tiny_files_exits_with_2(tmp_path, capsys):
    one = write_jsonl(
        tmp_path / "one.jsonl", [{"question": "q", "choices": ["a", "b"], "answer": 0}]
    )
    assert main(["choices-only", str(one)]) == 2
    assert "folds" in capsys.readouterr().err
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert main(["choices-only", str(empty)]) == 2


def test_more_than_26_options(tmp_path, capsys):
    opts = [f"option {j}" for j in range(27)]
    path = write_jsonl(tmp_path / "wide.jsonl", [{"question": "q", "choices": opts, "answer": 26}])
    assert main(["cues", str(path)]) == 0
    assert "always 27" in capsys.readouterr().out
    assert main(["audit", str(path)]) == 0
    assert "| 27 |" in capsys.readouterr().out
