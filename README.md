# mcqaudit

Measure what a multiple-choice benchmark gives away before any model reads the question: answer-position
imbalance, option-length and wording cues, how far question-free heuristics and an options-only classifier score
above chance, and item hygiene. No model and no network are involved.

## Why this is hard to see

Four options suggest a 25 % chance level, but that number assumes the key and the options carry no pattern. Keys
written by hand often favour one position or avoid repeating a letter, and the correct option is often the longest,
the most hedged, or the "all of the above" one. A model can pick these regularities up and score above chance
without reading the question, so the nominal chance level hides how much of a score is earned. Finding them needs
the right null distribution for each check (option counts differ between items, heuristics tie, subjects differ),
not a model.

## Install

```
pip install git+https://github.com/vipul21435/mcqaudit
```

Python 3.10 or newer; the only runtime dependency is numpy. For Parquet input:
`pip install "mcqaudit[parquet] @ git+https://github.com/vipul21435/mcqaudit"`.

## Quickstart

`examples/sample.jsonl` is a 20-item file written for this README, with cues planted on purpose: the correct
option is often at C and is often the longest one.

```
$ mcqaudit audit examples/sample.jsonl --no-subjects
...
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
...
| heuristic | covered | accuracy | 95 % interval | chance | lift | z | p (one-sided) | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| most question overlap | 8 | 100.0 % | 67.6 % to 100.0 % | 25.0 % | +75.0 pt | +4.90 | 1.5e-05 | 0.0002 |
| longest option | 16 | 78.1 % | 53.7 % to 91.7 % | 25.0 % | +53.1 pt | +5.01 | 6.6e-06 | 7.3e-05 |
| always C | 20 | 60.0 % | 38.7 % to 78.1 % | 25.0 % | +35.0 pt | +3.61 | 0.0009 | 0.0084 |
| middle number | 6 | 50.0 % | 18.8 % to 81.2 % | 25.0 % | +25.0 pt | +2.45 | 0.0156 | 0.1250 |
...
```

```
$ mcqaudit choices-only examples/sample.jsonl --folds 2
...
| items | correct | accuracy | 95 % interval | chance | lift | z | p (one-sided) | majority position |
|---|---|---|---|---|---|---|---|---|
| 20 | 12 | 60.0 % | 38.7 % to 78.1 % | 25.0 % | +35.0 pt | +3.61 | 0.0002 | 60.0 % |

Majority position: accuracy of always picking the training folds' most common position.
0 held-out items had their option set in another fold; those copies were left out of training.

$ mcqaudit shuffle examples/sample.jsonl --balanced -o balanced.jsonl
wrote 20 items to balanced.jsonl; position test p 0.0121 before, 1 after
```

`mcqaudit hygiene FILE [--against OTHER_SPLIT]` prints the hygiene table. `audit`, `cues`, `choices-only` and
`hygiene` take `--format json` and `-o FILE`; `shuffle` always writes JSONL to its `-o FILE` and ignores
`--format`. `python examples/quickstart.py` runs the same checks through the Python API.

## How it works

**Input.** JSONL, JSON, CSV/TSV or Parquet. Each record's layout is detected: `question` with a `choices` list and
an index or letter `answer` (MMLU on the Hugging Face hub); `choices` as `{"text": [...], "label": [...]}` with an
`answerKey` (ARC, CommonsenseQA); the `question.stem` / `question.choices` form of the original ARC files; an
`options` dictionary; or one column per option (`A`, `B`, ... or `opa`, `opb`, ...; MedMCQA). Integer answers are
read as 0-based, or as 1-based when a file never uses 0 and reaches the option count; `--answer-base` overrides it.

**Answer positions (`audit`).** Items are grouped by option count, because an item with three options cannot have
its answer at D. Within a group of n items with k options the counts are tested against a uniform key with
Pearson's statistic, sum (O - E)^2 / E with E = n / k and k - 1 degrees of freedom. When the number of possible
tables, C(n + k - 1, k - 1), is at most `--exact-limit` (200,000) the exact multinomial test is used instead: the
p-value is the total probability of all tables no more likely than the observed one, which with two cells is the
two-sided exact binomial test. Groups are combined by summing statistics and degrees of freedom. The same table is
given per subject, with Holm-adjusted p-values. The chi-square tail is the regularized upper incomplete gamma
function, written here (series below a + 1, continued fraction above) and tested against scipy.

**Key order (`audit`).** A run is a maximal block of consecutive items with the same answer position. Given counts
n_j, S2 = sum n_j^2 and S3 = sum n_j^3, a random order gives E[R] = (n(n + 1) - S2) / n and
Var[R] = (S2 (S2 + n(n + 1)) - 2 n S3 - n^3) / (n^2 (n - 1)); with two categories this is the Wald-Wolfowitz runs
test. The z score uses the normal approximation. Negative z means answers cluster; positive z means the key avoids
repeating a position. The tests check both moments against full enumeration of permutations.

**Heuristics (`cues`).** Always position k; longest and shortest option (characters); most and least content-word
overlap with the question; an "all of the above", "none of the above" or "both" option; the median value when every
option is a single number; the option most similar to the others and the odd one out (mean Jaccard similarity of
character trigrams). A heuristic that names several options is scored by its expected credit under random
tie-breaking, 1/|C| when the answer is in the tied set C; it abstains when it names every option or none. Accuracy
over covered items comes with a Wilson score interval. If the key were independent of the cue, an item with k
options would give credit with mean 1/k and variance 1/(k|C|) - 1/k^2. The one-sided p-value against chance is exact
when a heuristic covers at most 500 items (the per-item credit distributions are convolved on a common grid) and
uses the normal approximation to the summed credit above that. With 14 or more heuristics in one table, each p-value
also gets a Holm adjustment across the heuristics, and that column is the one to read before calling a heuristic
above chance.

**Choices-only classifier (`choices-only`).** The idea follows partial-input baselines (hypothesis-only models for
NLI, choices-only prompts for LLMs, see Background): give a model everything except the question. Here the model is
a multinomial logistic regression over answer positions. For each position j, the word unigrams and bigrams of
option j are hashed (CRC-32, signed) into their own block of 4,096 columns; the question is never read. The loss is
the mean cross-entropy plus (1 / 2Cn) ||W||^2 with an unpenalised intercept, the same optimum as scikit-learn's
`LogisticRegression(C)` (the tests compare the two), fitted by a small L-BFGS written on numpy with up to 300
iterations per fold (`--max-iter`); the report warns when a fold stops at that cap rather than at the optimum.
Positions an item lacks are masked. Folds are grouped by subject when there are at least as many subject groups as
folds (items without a subject form one group), so the model is tested on subjects it has never seen; otherwise
they are a seeded random split. An item whose normalised option set also occurs in the held-out fold is left out of
training, so repeated items, which can span subjects, are not answered from memory; the report counts the held-out
items affected. Features are built in a fixed order, so the result does not depend on the Python hash seed. The
pooled out-of-fold accuracy with a Wilson interval is a lower bound on what the options reveal (its p-value uses
the normal approximation). The report also gives the accuracy of always picking
the training folds' most common position, which separates key imbalance from wording cues.

**Hygiene (`hygiene`).** After normalisation (NFKC, case-folded, collapsed whitespace, no trailing period): items
with duplicate options, with an empty option, duplicated question texts, repeated items (same question and same
option set), duplicates that give different answers to the same options, answer text that appears verbatim in the
question (and whether no distractor does), and items off the usual option count. `--against` counts questions that
also occur in another file, exactly or with word 3-gram Jaccard similarity at or above `--threshold` (0.8).

**Copies (`shuffle`).** A seeded per-item shuffle of the options, or with `--balanced` a copy in which answer
positions are as even as possible within each option count (the answer is moved, distractors keep their order).
The answer text always follows the key.

## Results on public benchmarks

Commands (measured on a loaded laptop with one thread: 84 s of CPU, 3 min 11 s wall clock, plus about 95 MB of
downloads; nothing is committed except the numbers):

```
bash benchmarks/fetch.sh data
python benchmarks/run.py data > benchmarks/results.md
```

The full output, including every heuristic for every dataset, is in
[benchmarks/results.md](benchmarks/results.md). The files come from the Hugging Face hub. Licences are those stated
in each dataset card's metadata; the ARC and MedMCQA cards leave their "Licensing Information" sections empty.

| dataset | file used | items | licence | hub id |
|---|---|---|---|---|
| MMLU (Hendrycks et al., arXiv 2009.03300) | `all/test` | 14,042 | MIT | `cais/mmlu` |
| ARC-Challenge (Clark et al., arXiv 1803.05457) | `ARC-Challenge/test` | 1,172 | CC BY-SA 4.0 | `allenai/ai2_arc` |
| CommonsenseQA (Talmor et al., arXiv 1811.00937) | `validation` (test has no answers) | 1,221 | MIT | `tau/commonsense_qa` |
| MedMCQA (Pal et al., arXiv 2203.14371) | `validation` | 4,183 | Apache-2.0 | `openlifescienceai/medmcqa` |

Answer positions and order:

| dataset | A | B | C | D | E | position test | runs test z (p) |
|---|---|---|---|---|---|---|---|
| MMLU test | 22.9 % | 24.7 % | 25.5 % | 26.9 % | | chi-square 45.9, df 3, p 5.9e-10; 6 of 57 subjects Holm p < 0.05 | -1.81 (0.070) |
| ARC-Challenge test | 22.6 % | 26.5 % | 26.5 % | 24.4 % | | summed chi-square 7.6, df 9, p 0.58; four-option stratum alone p 0.17 | +0.62 (0.53) |
| CommonsenseQA val. | 19.6 % | 20.9 % | 19.7 % | 20.6 % | 19.2 % | chi-square 1.2, df 4, p 0.88 | -1.98 (0.048) |
| MedMCQA val. | 32.2 % | 25.9 % | 22.1 % | 19.7 % | | chi-square 149.4, df 3, p 3.6e-32; 4 of 21 subjects Holm p < 0.05 | +1.68 (0.092) |

ARC shares are over its 1,165 four-option items; it also has 4 three-option and 3 five-option items, tested in
their own strata. Summing chi-square over strata is conservative when some strata are this small, which is why the
largest stratum's own p-value is printed beside the combined one.

Question-free scores. "Best heuristic" is the one with the largest lower interval bound above chance among those
that apply to at least 30 items. It is picked after looking at 14 or more heuristics, so its interval is optimistic;
the Holm-adjusted p-value accounts for the number of heuristics, not for the choice of the best one.

| dataset | best heuristic (items covered) | its accuracy (95 % CI) | chance | p (Holm) | choices-only accuracy (95 % CI) | folds | majority position |
|---|---|---|---|---|---|---|---|
| MMLU test | all of the above (344) | 64.5 % (59.3 to 69.4) | 25.0 % | 1.4e-52 | 26.0 % (25.3 to 26.7) | grouped by subject | 24.9 % |
| ARC-Challenge test | least question overlap (377) | 33.5 % (28.9 to 38.4) | 25.0 % | 5.7e-08 | 27.2 % (24.7 to 29.8) | random | 25.2 % |
| CommonsenseQA val. | longest option (1,220) | 22.2 % (20.0 to 24.6) | 20.0 % | 0.24 | 21.0 % (18.8 to 23.3) | random | 18.8 % |
| MedMCQA val. | all of the above (113) | 46.9 % (38.0 to 56.1) | 25.0 % | 4.6e-06 | 28.9 % (27.5 to 30.3) | grouped by subject | 32.2 % |

Held-out items whose option set was also in another fold, and so left out of training: 382 for MMLU, 12 for ARC,
2 for CommonsenseQA and 62 for MedMCQA. No fold stopped at the iteration cap. The MMLU count is reproducible: a
second run in a new process, `PYTHONHASHSEED=7 mcqaudit choices-only data/mmlu/test.parquet`, gave the same 3,649
correct answers.

Hygiene, and overlap with the training or validation split:

| dataset | duplicate options | duplicated question texts (items; texts whose items all differ in options) | repeated item groups | answer in question (only the answer) | overlap exact / near |
|---|---|---|---|---|---|
| MMLU test | 14 | 155 (331; 46) | 110 | 92 (41) | 48 / 190 against validation |
| ARC-Challenge test | 1 | 2 (4; 2) | 0 | 5 (0) | 7 / 8 against train |
| CommonsenseQA val. | 24 | 0 | 0 | 0 | 0 / 0 against train |
| MedMCQA val. | 21 | 0 | 0 | 7 (5) | 0 / 25 against train |

No dataset had empty options or duplicates with conflicting answers.

Read plainly: the MMLU test key leans towards D and away from A, and an "all of the above" option is the answer in
64.5 % of the 344 items that have one. With repeated items kept out of training, the choices-only classifier scores
26.0 % on MMLU, one point above chance and above the majority position (24.9 %). MedMCQA's validation key puts
32.2 % of answers at A; there the classifier (28.9 %) does worse than always answering the training folds' most
common position (32.2 %), so as a lower bound it is loose. ARC-Challenge and CommonsenseQA keys are consistent with
uniform at these sizes, and the classifier is within 2.2 points of chance on both. On CommonsenseQA no heuristic is
above chance after the Holm adjustment (the best, longest option, has unadjusted p 0.022 and Holm p 0.24). Of
MMLU's 155 duplicated question texts, 46 are generic stems ("Which of the following is true?") whose items all have
different options, and 109 hold at least one repeated item (same question and same options; 110 such groups in
all).

## Limitations

- Every test is about the dataset under a "no cue" null, not about any model. A cue found here is a regularity a
  model could use, not proof that one does.
- The runs test and the classifier's z test use normal approximations; for a few dozen items treat their p-values
  as rough. The position test and heuristics covering up to 500 items use exact p-values.
- The choices-only classifier is one fixed linear model with one C, no tuning and 4,096 hashed columns per position
  (collisions are possible); a stronger model would give a higher lower bound. Folds are grouped by subject only
  when the file has a subject field (MMLU and MedMCQA here).
- Heuristics are fixed English patterns; the "both" pattern and the numeric parser miss many phrasings, and the
  stopword list behind the overlap heuristics is English only. Words are runs of letters, digits and combining marks
  in any script, so text written without spaces (Chinese, Japanese) gives one token per run of characters.
- Near-duplicate detection uses word 3-gram Jaccard on question text only, skips shingles that occur in more than
  200 questions, and checks the 20 best candidates per question; it can miss paraphrases.
- A record without an answer stops the load unless `--skip-unanswered` is given; skipped records are counted in
  the report. A 1-based key that never uses its last position is read as 0-based (the "auto" rule cannot tell);
  the report then carries a note, and `--answer-base 1` fixes it.
- `shuffle` writes the list layout in JSONL, not the input layout. Options whose meaning depends on their position
  ("both A and B", "all of the above") are moved like any other option.
- Not implemented: Monte Carlo p-values for the runs test, a permutation null for the classifier, and model-based
  checks such as the option-order sensitivity studied by Zheng et al. and by Pezeshkpour and Hruschka.

## Background

- Gururangan et al., "Annotation Artifacts in Natural Language Inference Data", NAACL 2018 (arXiv 1803.02324), and
  Poliak et al., "Hypothesis Only Baselines in Natural Language Inference", *SEM 2018 (arXiv 1805.01042): models that
  see only the hypothesis classify NLI data well above the majority class, exposing annotation artifacts.
- Kaushik and Lipton, "How Much Reading Does Reading Comprehension Require? A Critical Investigation of Popular
  Benchmarks", EMNLP 2018 (arXiv 1808.04926): question-only and passage-only baselines often do surprisingly well.
- Balepur, Ravichander and Rudinger, "Artifacts or Abduction: How Do LLMs Answer Multiple-Choice Questions Without
  the Question?", ACL 2024 (arXiv 2402.12483): choices-only prompts beat a majority baseline in 11 of their 12
  dataset and model pairs.
- Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024 (arXiv 2309.03882), and
  Pezeshkpour and Hruschka, "Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions"
  (arXiv 2308.11483): LLMs favour some option IDs and change answers when options are reordered, which is
  why an imbalanced key matters and why `shuffle` exists.

## Related tools

- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for evaluation scores, for
  example a model on the original file and on a copy made with `mcqaudit shuffle`.
- [contamscan](https://github.com/vipul21435/contamscan) and [datasieve](https://github.com/vipul21435/datasieve):
  n-gram contamination checks and deduplication at corpus scale, beyond the split-overlap check here.

## Licence

MIT. See [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
