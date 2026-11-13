#!/usr/bin/env bash
# Download the public benchmark files used in the README into a directory (default: ./data).
# Nothing here is committed to the repository; only the derived numbers are.
set -euo pipefail

out="${1:-data}"
hub="https://huggingface.co/datasets"
mkdir -p "$out"/mmlu "$out"/arc "$out"/csqa "$out"/medmcqa

get() { [ -s "$2" ] || curl -sfL -o "$2" "$1"; echo "$2"; }

get "$hub/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet" "$out/mmlu/test.parquet"
get "$hub/cais/mmlu/resolve/main/all/validation-00000-of-00001.parquet" "$out/mmlu/validation.parquet"
get "$hub/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet" "$out/arc/challenge-test.parquet"
get "$hub/allenai/ai2_arc/resolve/main/ARC-Challenge/train-00000-of-00001.parquet" "$out/arc/challenge-train.parquet"
get "$hub/tau/commonsense_qa/resolve/main/data/validation-00000-of-00001.parquet" "$out/csqa/validation.parquet"
get "$hub/tau/commonsense_qa/resolve/main/data/train-00000-of-00001.parquet" "$out/csqa/train.parquet"
get "$hub/openlifescienceai/medmcqa/resolve/main/data/validation-00000-of-00001.parquet" "$out/medmcqa/validation.parquet"
get "$hub/openlifescienceai/medmcqa/resolve/main/data/train-00000-of-00001.parquet" "$out/medmcqa/train.parquet"
