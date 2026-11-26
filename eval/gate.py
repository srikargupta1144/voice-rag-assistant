"""CI quality gate: fail when Recall@5 drops below the committed baseline.

Flow:
    CI retrieval-gate job (after `vra ingest` and `python -m eval.retrieval_eval`)
        -> python -m eval.gate --results eval/results/retrieval.json
        -> compares recall_at_5 against eval/baseline.json
        -> exit 1 on regression, so a merge that hurts retrieval is blocked

The baseline is updated deliberately (commit a new eval/baseline.json after an
improvement), never automatically. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def check(results: dict, baseline: dict, metric: str, tolerance: float) -> tuple[bool, str]:
    """(passed, message) for one metric."""
    base, cur = baseline.get(metric), results.get(metric)
    if base is None:
        return True, f"gate: baseline has no {metric}; passing"
    if cur is None:
        return False, f"gate FAILED: results have no {metric}"
    if cur + tolerance < base:
        return False, f"gate FAILED: {metric} {cur} < baseline {base} (tolerance {tolerance})"
    return True, f"gate passed: {metric} {cur} >= baseline {base} (tolerance {tolerance})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=REPO / "eval" / "results" / "retrieval.json"
    )
    parser.add_argument("--baseline", type=Path, default=REPO / "eval" / "baseline.json")
    parser.add_argument("--metric", default="recall_at_5")
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    if not args.results.exists():
        print(f"gate FAILED: results file {args.results} missing")
        return 1
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    ok, msg = check(results, baseline, args.metric, args.tolerance)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
