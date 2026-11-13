# mcqaudit on public benchmarks

| benchmark | items | answer positions | position test p | runs z (p) | best heuristic | its accuracy (95 % CI) | chance | p (Holm) | choices-only accuracy (95 % CI) | majority position |
|---|---|---|---|---|---|---|---|---|---|---|
| MMLU test | 14042 | A 22.9 %, B 24.7 %, C 25.5 %, D 26.9 % | 5.9e-10 | -1.81 (0.070) | all of the above (covers 344) | 64.5 % (59.3 % to 69.4 %) | 25.0 % | 1.4e-52 | 26.0 % (25.3 % to 26.7 %), chance 25.0 %, grouped by subject | 24.9 % |
| ARC-Challenge test | 1172 | A 22.6 %, B 26.5 %, C 26.5 %, D 24.4 % | 0.577 (largest stratum 0.166) | +0.62 (0.534) | least question overlap (covers 377) | 33.5 % (28.9 % to 38.4 %) | 25.0 % | 5.7e-08 | 27.2 % (24.7 % to 29.8 %), chance 25.0 %, random (no subject grouping) | 25.2 % |
| CommonsenseQA validation | 1221 | A 19.6 %, B 20.9 %, C 19.7 %, D 20.6 %, E 19.2 % | 0.884 | -1.98 (0.048) | longest option (covers 1220) | 22.2 % (20.0 % to 24.6 %) | 20.0 % | 0.244 | 21.0 % (18.8 % to 23.3 %), chance 20.0 %, random (no subject grouping) | 18.8 % |
| MedMCQA validation | 4183 | A 32.2 %, B 25.9 %, C 22.1 %, D 19.7 % | 3.6e-32 | +1.68 (0.092) | all of the above (covers 113) | 46.9 % (38.0 % to 56.1 %) | 25.0 % | 4.6e-06 | 28.9 % (27.5 % to 30.3 %), chance 25.0 %, grouped by subject | 32.2 % |

## Hygiene

| benchmark | duplicate options | empty options | duplicated questions (items; all options differ) | repeated item groups | conflicting duplicates | answer in question (only answer) | option counts | overlap: exact / near (against) |
|---|---|---|---|---|---|---|---|---|
| MMLU test | 14 | 0 | 155 (331; 46) | 110 | 0 | 92 (41) | 4: 14042 | 48 / 190 (validation.parquet) |
| ARC-Challenge test | 1 | 0 | 2 (4; 2) | 0 | 0 | 5 (0) | 3: 4, 4: 1165, 5: 3 | 7 / 8 (challenge-train.parquet) |
| CommonsenseQA validation | 24 | 0 | 0 (0; 0) | 0 | 0 | 0 (0) | 5: 1221 | 0 / 0 (train.parquet) |
| MedMCQA validation | 21 | 0 | 0 (0; 0) | 0 | 0 | 7 (5) | 4: 4183 | 0 / 25 (train.parquet) |

## Every heuristic

### MMLU test

| heuristic | covered | accuracy | 95 % CI | chance | p | p (Holm) |
|---|---|---|---|---|---|---|
| all of the above | 344 | 64.5 % | 59.3 % to 69.4 % | 25.0 % | 9.8e-54 | 1.4e-52 |
| both ... | 373 | 35.3 % | 30.6 % to 40.3 % | 25.0 % | 1.6e-06 | 1.5e-05 |
| middle number | 1016 | 29.8 % | 27.0 % to 32.7 % | 25.0 % | 8.0e-10 | 9.6e-09 |
| longest option | 13349 | 28.4 % | 27.6 % to 29.2 % | 25.0 % | 1.5e-21 | 2.0e-20 |
| always D | 14042 | 26.9 % | 26.2 % to 27.6 % | 25.0 % | 1.1e-07 | 1.1e-06 |
| most similar to others | 12659 | 26.9 % | 26.1 % to 27.7 % | 25.0 % | 4.5e-08 | 4.9e-07 |
| most question overlap | 5727 | 25.6 % | 24.5 % to 26.8 % | 25.0 % | 0.102 | 0.715 |
| always C | 14042 | 25.5 % | 24.8 % to 26.2 % | 25.0 % | 0.082 | 0.654 |
| always B | 14042 | 24.7 % | 23.9 % to 25.4 % | 25.0 % | 0.828 | 1.000 |
| least question overlap | 5727 | 24.3 % | 23.2 % to 25.5 % | 25.0 % | 0.952 | 1.000 |
| odd one out | 12659 | 23.5 % | 22.8 % to 24.3 % | 25.0 % | 1.000 | 1.000 |
| always A | 14042 | 22.9 % | 22.3 % to 23.6 % | 25.0 % | 1.000 | 1.000 |
| shortest option | 13349 | 22.2 % | 21.5 % to 22.9 % | 25.0 % | 1.000 | 1.000 |
| none of the above | 205 | 21.0 % | 16.0 % to 27.1 % | 25.0 % | 0.923 | 1.000 |

### ARC-Challenge test

| heuristic | covered | accuracy | 95 % CI | chance | p | p (Holm) |
|---|---|---|---|---|---|---|
| middle number | 48 | 35.4 % | 23.4 % to 49.6 % | 25.0 % | 0.003 | 0.033 |
| least question overlap | 377 | 33.5 % | 28.9 % to 38.4 % | 25.0 % | 4.4e-09 | 5.7e-08 |
| both ... | 17 | 28.4 % | 12.6 % to 52.2 % | 25.0 % | 0.390 | 1.000 |
| longest option | 1139 | 27.5 % | 25.0 % to 30.1 % | 25.0 % | 0.021 | 0.228 |
| most similar to others | 1061 | 26.6 % | 24.0 % to 29.3 % | 25.0 % | 0.105 | 1.000 |
| always B | 1172 | 26.5 % | 24.1 % to 29.1 % | 25.0 % | 0.115 | 1.000 |
| always C | 1172 | 26.5 % | 24.0 % to 29.0 % | 25.0 % | 0.128 | 1.000 |
| always D | 1168 | 24.4 % | 22.0 % to 26.9 % | 25.0 % | 0.678 | 1.000 |
| odd one out | 1061 | 22.8 % | 20.4 % to 25.4 % | 25.0 % | 0.960 | 1.000 |
| always A | 1172 | 22.7 % | 20.4 % to 25.2 % | 25.0 % | 0.967 | 1.000 |
| shortest option | 1139 | 19.6 % | 17.4 % to 22.0 % | 25.0 % | 1.000 | 1.000 |
| most question overlap | 377 | 13.5 % | 10.4 % to 17.4 % | 25.0 % | 1.000 | 1.000 |
| always E | 3 | 0.0 % | 0.0 % to 56.1 % | 20.0 % | 1.000 | 1.000 |

### CommonsenseQA validation

| heuristic | covered | accuracy | 95 % CI | chance | p | p (Holm) |
|---|---|---|---|---|---|---|
| most similar to others | 827 | 22.5 % | 19.8 % to 25.5 % | 20.0 % | 0.008 | 0.094 |
| longest option | 1220 | 22.2 % | 20.0 % to 24.6 % | 20.0 % | 0.022 | 0.244 |
| least question overlap | 97 | 21.4 % | 14.4 % to 30.6 % | 20.0 % | 0.137 | 1.000 |
| always B | 1221 | 20.9 % | 18.7 % to 23.3 % | 20.0 % | 0.220 | 1.000 |
| always D | 1221 | 20.6 % | 18.4 % to 22.9 % | 20.0 % | 0.313 | 1.000 |
| always C | 1221 | 19.7 % | 17.6 % to 22.1 % | 20.0 % | 0.591 | 1.000 |
| always A | 1221 | 19.6 % | 17.4 % to 21.9 % | 20.0 % | 0.645 | 1.000 |
| always E | 1221 | 19.2 % | 17.1 % to 21.6 % | 20.0 % | 0.745 | 1.000 |
| odd one out | 827 | 17.7 % | 15.2 % to 20.4 % | 20.0 % | 0.996 | 1.000 |
| shortest option | 1220 | 15.2 % | 13.3 % to 17.3 % | 20.0 % | 1.000 | 1.000 |
| most question overlap | 97 | 14.6 % | 8.9 % to 23.0 % | 20.0 % | 0.933 | 1.000 |
| both ... | 1 | 0.0 % | 0.0 % to 79.3 % | 20.0 % | 1.000 | 1.000 |

### MedMCQA validation

| heuristic | covered | accuracy | 95 % CI | chance | p | p (Holm) |
|---|---|---|---|---|---|---|
| all of the above | 113 | 46.9 % | 38.0 % to 56.1 % | 25.0 % | 3.9e-07 | 4.6e-06 |
| both ... | 81 | 37.7 % | 27.9 % to 48.5 % | 25.0 % | 0.006 | 0.053 |
| always A | 4183 | 32.2 % | 30.8 % to 33.7 % | 25.0 % | 1.9e-27 | 2.6e-26 |
| middle number | 158 | 31.3 % | 24.6 % to 38.9 % | 25.0 % | 0.001 | 0.014 |
| longest option | 3979 | 30.2 % | 28.8 % to 31.7 % | 25.0 % | 4.5e-16 | 5.8e-15 |
| most question overlap | 838 | 28.0 % | 25.1 % to 31.1 % | 25.0 % | 0.009 | 0.072 |
| most similar to others | 3707 | 27.0 % | 25.6 % to 28.5 % | 25.0 % | 6.7e-04 | 0.007 |
| always B | 4183 | 25.9 % | 24.6 % to 27.3 % | 25.0 % | 0.081 | 0.564 |
| least question overlap | 838 | 23.4 % | 20.6 % to 26.4 % | 25.0 % | 0.968 | 1.000 |
| odd one out | 3707 | 22.2 % | 20.9 % to 23.5 % | 25.0 % | 1.000 | 1.000 |
| always C | 4183 | 22.1 % | 20.9 % to 23.4 % | 25.0 % | 1.000 | 1.000 |
| shortest option | 3979 | 21.4 % | 20.1 % to 22.7 % | 25.0 % | 1.000 | 1.000 |
| always D | 4183 | 19.7 % | 18.5 % to 21.0 % | 25.0 % | 1.000 | 1.000 |
| none of the above | 115 | 1.7 % | 0.5 % to 6.1 % | 25.0 % | 1.000 | 1.000 |

