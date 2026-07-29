#!/bin/bash
# run_full_pipeline.sh — run the full experiment chain in dependency order.
#
# Usage (from anywhere):
#   bash experiments/tools/run_full_pipeline.sh
#
# Required order:
#   pilot sanity -> tune (best_params.json) -> main eval -> sensitivity
#   -> drift sweep -> plots -> summary digest
#
# Reference wall time: ~35-45 minutes on 4 workers.
#
# Every summary number reported in the paper is produced by step 6. None is
# typed by hand — see lessons/README.md, pitfall #10.

set -e

# Always run from the repository root (parent of experiments/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
mkdir -p logs
LOG="logs/pipeline-$(date +%Y-%m-%d_%H%M).log"

{
echo "=== [0/7] PILOT SANITY: $(date) ==="
$PY -m experiments.tools.pilot_sanity      # non-zero exit aborts everything (set -e)

echo "=== [1/7] TUNE: $(date) ==="
$PY -m experiments.tune

echo "=== [2/7] MAIN EVAL: $(date) ==="
$PY -m experiments.run_experiments

echo "=== [3/7] SENSITIVITY: $(date) ==="
$PY -m experiments.run_sensitivity_experiments

echo "=== [4/7] DRIFT SWEEP: $(date) ==="
$PY -m experiments.run_drift_frequency_sweep

echo "=== [5/7] PLOTS: $(date) ==="
$PY -m experiments.plot
$PY -m experiments.plot_sensitivity
$PY -m experiments.plot_drift_sweep
$PY -m experiments.plot_latency_distribution

echo "=== [6/7] DIGEST (source of truth for every reported number): $(date) ==="
$PY -m experiments.make_summary_numbers
$PY -m experiments.make_appendix_tables

echo "=== [7/7] PIPELINE DONE: $(date) ==="
} 2>&1 | tee "$LOG"

echo "Log saved to: $LOG"
