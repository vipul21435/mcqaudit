# Changelog

## 0.1.0 (unreleased)

First version.

- Read JSONL, JSON, CSV/TSV and Parquet; detect the list, text/label, stem, options-dictionary and
  one-column-per-option layouts; read integer answers as 0- or 1-based.
- `mcqaudit audit`: answer-position counts per option count with Pearson's chi-square or the exact
  multinomial test, per-subject tables with Holm-adjusted p-values, and a k-category runs test on the
  order of the key.
- `mcqaudit cues`: always position k, longest and shortest option, most and least word overlap with
  the question, "all of the above", "none of the above" and "both" options, the middle numeric value,
  the option most similar to the others and the odd one out; accuracy with Wilson intervals and a
  z test against chance, ties scored by expected credit.
- `mcqaudit choices-only`: multinomial logistic regression (own L-BFGS on numpy) on hashed word
  n-grams of the options alone, cross-validated with folds grouped by subject.
- `mcqaudit hygiene`: duplicate and empty options, duplicated questions, answer text in the question,
  option-count consistency, and exact or near-duplicate questions against a second file.
- `mcqaudit shuffle`: seeded option shuffling and position-balanced copies, written as JSONL.
- Markdown and JSON output for every command.
- `benchmarks/fetch.sh` and `benchmarks/run.py` for the README numbers on MMLU, ARC-Challenge,
  CommonsenseQA and MedMCQA.
