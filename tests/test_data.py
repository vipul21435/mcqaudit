import json

import pytest

from mcqaudit.data import Item, LoadError, items_from_records, load


def test_list_layout_with_index_answer():
    ds = items_from_records(
        [{"question": "2 + 2?", "subject": "maths", "choices": ["3", "4", "5", "6"], "answer": 1}]
    )
    (item,) = ds.items
    assert item.options == ("3", "4", "5", "6")
    assert item.answer == 1
    assert item.answer_text == "4"
    assert item.subject == "maths"
    assert ds.layouts == {"list": 1}


def test_list_layout_with_letter_answer():
    ds = items_from_records([{"question": "q", "choices": ["a", "b", "c"], "answer": "C"}])
    assert ds.items[0].answer == 2


def test_text_label_layout():
    rec = {
        "id": "x1",
        "question": "Which is a mammal?",
        "choices": {"text": ["trout", "whale", "crow"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    item = items_from_records([rec]).items[0]
    assert item.id == "x1"
    assert item.answer_text == "whale"


def test_text_label_layout_with_numeric_labels():
    rec = {
        "question": "q",
        "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]},
        "answerKey": "1",
    }
    ds = items_from_records([rec])
    assert ds.items[0].answer == 0
    assert ds.answer_base == 0


def test_stem_layout():
    rec = {
        "id": "s1",
        "question": {
            "stem": "Where do fish live?",
            "choices": [{"text": "trees", "label": "A"}, {"text": "water", "label": "B"}],
        },
        "answerKey": "B",
    }
    ds = items_from_records([rec])
    assert ds.items[0].question == "Where do fish live?"
    assert ds.items[0].answer == 1
    assert ds.layouts == {"stem": 1}


def test_dict_layout_prefers_answer_idx():
    rec = {
        "question": "q",
        "options": {"A": "red", "B": "green", "C": "blue"},
        "answer": "blue",
        "answer_idx": "C",
    }
    item = items_from_records([rec]).items[0]
    assert item.answer == 2


def test_dict_layout_answer_as_text():
    rec = {"question": "q", "options": {"A": "red", "B": "green"}, "answer": "green"}
    assert items_from_records([rec]).items[0].answer == 1


def test_column_layout_opa():
    rec = {"question": "q", "opa": "w", "opb": "x", "opc": "y", "opd": "z", "cop": 3}
    ds = items_from_records([rec], answer_base=1)
    assert ds.items[0].answer_text == "y"


def test_auto_base_detects_one_based():
    recs = [
        {"question": "q", "options": ["a", "b", "c"], "answer": 3},
        {"question": "q", "options": ["a", "b", "c"], "answer": 1},
    ]
    ds = items_from_records(recs)
    assert ds.answer_base == 1
    assert [it.answer for it in ds.items] == [2, 0]
    assert ds.notes


def test_auto_base_keeps_zero_based_when_zero_seen():
    recs = [
        {"question": "q", "options": ["a", "b", "c"], "answer": 0},
        {"question": "q", "options": ["a", "b", "c"], "answer": 2},
    ]
    ds = items_from_records(recs)
    assert ds.answer_base == 0
    assert [it.answer for it in ds.items] == [0, 2]


def test_out_of_range_answer_is_an_error():
    with pytest.raises(LoadError, match="out of range"):
        items_from_records([{"question": "q", "choices": ["a", "b"], "answer": 5}])


def test_missing_options_is_an_error():
    with pytest.raises(LoadError, match="no options"):
        items_from_records([{"question": "q", "answer": 0}])


def test_jsonl_and_csv_files_give_the_same_items(tmp_path):
    rows = [
        {"question": "First?", "A": "a1", "B": "b1", "C": "c1", "D": "d1", "answer": "D"},
        {"question": "Second?", "A": "a2", "B": "b2", "C": "c2", "D": "d2", "answer": "A"},
    ]
    jsonl = tmp_path / "x.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    csv_path = tmp_path / "x.csv"
    lines = ["question,A,B,C,D,answer"] + [
        ",".join(str(r[k]) for k in ("question", "A", "B", "C", "D", "answer")) for r in rows
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    a = load(jsonl).items
    b = load(csv_path).items
    assert [(i.question, i.options, i.answer) for i in a] == [
        (i.question, i.options, i.answer) for i in b
    ]
    assert a[0] == Item("item-1", "First?", ("a1", "b1", "c1", "d1"), 3, None)


def test_csv_choices_as_json_string(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text('question,choices,answer\nq,"[""x"", ""y"", ""z""]",2\n', encoding="utf-8")
    item = load(path).items[0]
    assert item.options == ("x", "y", "z")
    assert item.answer == 2


def test_bad_jsonl_line_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "q", "choices": ["a", "b"], "answer": 0}\n{oops\n')
    with pytest.raises(LoadError, match=":2:"):
        load(path)


def test_parquet_roundtrip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"question": ["q"], "choices": [["a", "b", "c"]], "answer": [2]})
    path = tmp_path / "t.parquet"
    pq.write_table(table, path)
    assert load(path).items[0].answer_text == "c"
