#!/usr/bin/env bash
# Regenerate paper analysis figures from tracked accessibility_ensemble products.
# Does NOT re-screen geometries or re-run PFLOTRAN campaigns.
#
#   bash scripts/reproduce_figures.sh
#   bash scripts/reproduce_figures.sh --refresh
#   bash scripts/reproduce_figures.sh --analysis-only
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REFRESH=()
ANALYSIS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --refresh) REFRESH=(--refresh) ;;
    --analysis-only) ANALYSIS_ONLY=1 ;;
    --sync-only)
      echo "Figure sync into LaTeX is not part of this branch (no manuscript/ tree)." >&2
      exit 1
      ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

PLOT=(python3 src/plot_accessibility_ensemble.py)

if [[ ! -d accessibility_ensemble/analysis ]]; then
  echo "ERROR: accessibility_ensemble/analysis missing under ${ROOT}" >&2
  exit 1
fi

echo "==> [1/2] metric tables + solid/outlet figures + dual combined panels"
"${PLOT[@]}" all-metrics "${REFRESH[@]}"

echo "==> [2/2] SI S3–S4 flow validation (skip with --analysis-only)"
if [[ "${ANALYSIS_ONLY}" -eq 0 ]]; then
  if [[ -x src/si_flow/run_tracer.sh ]]; then
    (cd src/si_flow && bash run_tracer.sh plot --flow-only) || \
      echo "WARN: S3 plot failed (check src/si_flow reference data)" >&2
  else
    echo "WARN: src/si_flow/run_tracer.sh missing; skip S3" >&2
  fi
  if [[ -f src/si_flow/plot_darcy_ensemble_perm.py ]]; then
    python3 src/si_flow/plot_darcy_ensemble_perm.py || \
      echo "WARN: S4 plot failed" >&2
  fi
else
  echo "    skipped"
fi

echo ""
echo "Done. Frozen SI S1 was not regenerated."
echo "Guide: src/REPRODUCE.md"
