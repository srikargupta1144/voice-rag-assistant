"""
File: eval/gate.py

Description:
    CI quality gate: fail when Recall@5 drops below the committed baseline.

    The baseline is updated deliberately (commit a new eval/baseline.json after an
    improvement), never automatically. Standard library only.

Flow:
    CI retrieval-gate job (after `vra ingest` and `python -m eval.retrieval_eval`)
        -> python -m eval.gate --results eval/results/retrieval.json
        -> compares recall_at_5 against eval/baseline.json
        -> exit 1 on regression, so a merge that hurts retrieval is blocked

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    argparse - standard library; command-line options for the results, baseline, metric
        and tolerance
    json - standard library; reads the results and baseline files
    sys - standard library; sets the process exit code
    pathlib - standard library; builds the default file paths

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def check(results: dict, baseline: dict, metric: str, tolerance: float) -> tuple[bool, str]:
    """
    Function name: check
    Purpose: Compare one metric in the results against the baseline. The gate passes when the
        baseline has no value for the metric, and fails when the results lack the metric or
        when the result plus the tolerance is below the baseline.
    Parameters:
        results (dict): Results document from eval.retrieval_eval.
        baseline (dict): Committed baseline document.
        metric (str): Name of the metric key to compare.
        tolerance (float): Allowed drop below the baseline.
    Returns:
        tuple[bool, str]: Whether the gate passed, and a message describing the outcome.
    """
    base, cur = baseline.get(metric), results.get(metric)
    if base is None:
        return True, f"gate: baseline has no {metric}; passing"
    if cur is None:
        return False, f"gate FAILED: results have no {metric}"
    if cur + tolerance < base:
        return False, f"gate FAILED: {metric} {cur} < baseline {base} (tolerance {tolerance})"
    return True, f"gate passed: {metric} {cur} >= baseline {base} (tolerance {tolerance})"


def main() -> int:
    """
    Function name: main
    Purpose: Parse the command-line options, read the results and baseline files, run the
        check and print its message. A missing results file fails the gate.
    Parameters:
        None
    Returns:
        int: Process exit code, 0 when the gate passes and 1 when it fails.
    """
    parser = argparse.ArgumentParser(
        description="CI quality gate: fail when Recall@5 drops below the committed baseline."
    )
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
