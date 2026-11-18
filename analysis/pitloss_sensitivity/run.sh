#!/bin/bash
# Run from the project root. A season file is published only after the CLI succeeds.
set -eu
mode=${1:?range or clean}
case "$mode" in range|clean) ;; *) exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
dir=analysis/pitloss_sensitivity
mkdir -p "$dir/logs" "$dir/outputs"
run_season() {
    local stage=$1 year=$2 stem
    local args=()
    if [ "$stage" = pre ]; then args+=(--prerace); fi
    stem="$mode-${stage}-$year"
    .venv/bin/python -u scripts/backtest_field.py --season "$year" --sims 300 --draws 200 --top 10 --seed 20261118 --pit-loss-filter "$mode" "${args[@]}" --out "$dir/outputs/$stem.tmp" > "$dir/logs/$stem.log" 2>&1 &&
       mv "$dir/outputs/$stem.tmp" "$dir/outputs/$stem.csv"
}
export -f run_season
export mode dir
printf '%s\n' 'post 2023' 'post 2024' 'post 2025' 'post 2026' 'pre 2024' 'pre 2025' 'pre 2026' |
 xargs -P 4 -n 2 bash -c 'run_season "$1" "$2"' _
