# mcqaudit

Measure what a multiple-choice benchmark gives away before any model reads the question: answer-position
imbalance, option-length cues, and how far simple question-free heuristics score above chance.

Status: early. The answer-key audit and the first heuristics work; the rest of the planned checks are listed
under "Not implemented yet" below.

## Why this is hard to see

Four options suggest a 25 % chance level, but that number assumes the key and the options carry no pattern. Keys
written by hand often favour one position or avoid repeating a letter, and the correct option is often the longest
or most carefully worded one. A model can pick these up during training and score above chance without reading the
question, so the nominal chance level hides how much of a score is earned. Checking for this needs no model, only
counts and the right null distributions, which is what this tool computes.

## Install

```
pip install git+https://github.com/vipul21435/mcqaudit
```

Python 3.10 or newer. No runtime dependencies; `pip install "mcqaudit[parquet] @ git+https://github.com/vipul21435/mcqaudit"`
adds pyarrow for Parquet input.

## Quickstart

`examples/sample.jsonl` is a 20-item file written for this README, with a cue planted on purpose: the correct
option is often the longest one and often at position C.

```
$ mcqaudit audit examples/sample.jsonl --no-subjects
# Answer-key audit: sample.jsonl

Items: 20

## Answer positions

| options | n | A | B | C | D | chi-square | df | p | method |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 20 | 3 (15.0 %) | 3 (15.0 %) | 12 (60.0 %) | 2 (10.0 %) | 13.20 | 3 | 0.0121 | exact multinomial |

## Order of the key (runs test, file order)

| n | runs | expected | z | p |
|---|---|---|---|---|
| 20 | 17 | 12.7 | +2.82 | 0.0048 |

z +2.82, more runs than chance (the key avoids repeating a position).
```

```
$ mcqaudit cues examples/sample.jsonl
# Question-free heuristics: sample.jsonl

| heuristic | covered | accuracy | 95 % interval | chance | lift | z | p (one-sided) |
|---|---|---|---|---|---|---|---|
| longest option | 16 | 78.1 % | 53.7 % to 91.7 % | 25.0 % | +53.1 pt | +5.01 | 2.7e-07 |
| always C | 20 | 60.0 % | 38.7 % to 78.1 % | 25.0 % | +35.0 pt | +3.61 | 0.0002 |
| always A | 20 | 15.0 % | 5.2 % to 36.0 % | 25.0 % | -10.0 pt | -1.03 | 0.8492 |
| always B | 20 | 15.0 % | 5.2 % to 36.0 % | 25.0 % | -10.0 pt | -1.03 | 0.8492 |
| always D | 20 | 10.0 % | 2.8 % to 30.1 % | 25.0 % | -15.0 pt | -1.55 | 0.9393 |
| shortest option | 16 | 8.3 % | 1.8 % to 31.0 % | 25.0 % | -16.7 pt | -1.97 | 0.9756 |
```

Both commands take `--format json` and `-o FILE`.

## What works so far

- Input: JSONL, JSON, CSV/TSV and Parquet. The layout of each record is detected: `question` with a `choices`
  list and an index or letter `answer`; `choices` as `{"text": [...], "label": [...]}` with an `answerKey`; the
  `question.stem` / `question.choices` form; an `options` dictionary; or one column per option (`A`, `B`, ... or
  `opa`, `opb`, ...). Integer answers are read as 0-based, or 1-based when the file never uses 0 and reaches the
  option count (`--answer-base` overrides this).
- Answer positions: counts per position, tested against a uniform key with Pearson's chi-square or, when the
  number of possible tables is small enough to enumerate (`--exact-limit`), the exact multinomial test. Items with
  different option counts are tested separately and the statistics summed. The same table per subject, with
  Holm-adjusted p-values.
- Key order: the k-category runs test (normal approximation) on the sequence of answer positions, overall and per
  subject. With two categories it reduces to the Wald-Wolfowitz test.
- Heuristics: always position k, longest option, shortest option. Accuracy over the items where a heuristic
  applies, a Wilson interval, the chance level for those items, and a one-sided z test against chance. Ties are
  scored by expected credit under random tie-breaking.

The statistics are written on the standard library and tested against scipy (chi-square tail, exact binomial
test, Wilson interval) and against full enumeration (exact multinomial test, runs-test mean and variance). The
tests also check the false-positive rate of the position, runs and heuristic tests on seeded null data.

## Not implemented yet

- The remaining heuristics: word overlap with the question, "all/none of the above" options, the middle numeric
  value, the option most similar to the others and the odd one out.
- The choices-only classifier (logistic regression on hashed n-grams of the options, cross-validated by subject).
- Item hygiene checks and split overlap.
- `shuffle` and position-balanced copies, and the `choices-only`, `hygiene` and `shuffle` commands.
- Results on public benchmarks.

## Licence

MIT. See [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
