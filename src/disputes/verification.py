"""Independent checks shared by scripts/verify_*.py.

Score is recomputed directly from rows using the rule in config, without the study code,
so a bug in economics.py cannot be hidden by its own check.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from disputes import protocol as proto
from disputes.config import ACTION_COST, BENEFIT_SHARE
from disputes.paths import EXECUTED_NOTEBOOKS


def rule_score(rows: pd.DataFrame, p: pd.Series) -> float:
    """Sum of (0.2·A·won − 6) over disputes where 0.2·A·p > 6."""
    fight = BENEFIT_SHARE * rows.amount_usd * p > ACTION_COST
    return float(((BENEFIT_SHARE * rows.amount_usd * rows.won - ACTION_COST) * fight).sum())


def check_rule_scores(predictions: pd.DataFrame, summary: pd.DataFrame, by: str,
                      flags=(("eligible", "score_usd"), ("observed_only", "observed_score_usd"))) -> int:
    """Recompute Score of every strategy for both amount scenarios; return the number of checks."""
    checks = 0
    for name, frame in predictions.groupby(by):
        assert frame.dispute_id.is_unique and frame.p.between(0, 1).all()
        assert frame.loc[~frame.eligible, "profit"].isna().all()
        for flag, column in flags:
            rows = frame.loc[frame[flag]]
            assert np.isclose(rule_score(rows, rows.p), summary.loc[name, column]), (name, flag)
            checks += 1
    return checks


def check_repeat(output: Path) -> dict[str, bool]:
    """Check that the second (notebook) run produced the same tables, protocol and code as the first."""
    first = proto.read_json(output / "first_run_evidence.json")
    final = proto.read_json(output / "run_evidence.json")
    same = {name: digest == final["result_sha256"][name] for name, digest in first["result_sha256"].items()}
    assert all(same.values()), same
    assert first["protocol_sha256"] == final["protocol_sha256"]
    assert proto.code_hashes_match(final["code_sha256"])
    return same


def check_executed_notebook(name: str) -> int:
    """Check that the executed notebook is valid and error-free; return the number of code cells."""
    notebook = nbformat.read(EXECUTED_NOTEBOOKS / name, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count and not any(o.output_type == "error" for o in cell.outputs) for cell in code), name
    return len(code)


def write_result(output: Path, result: dict) -> None:
    proto.write_json(output / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
