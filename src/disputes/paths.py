"""Locations of data, code and results, relative to the repository root."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[1]
DATASET = PROJECT / "dataset.csv"
OUTPUTS = PROJECT / "outputs"
NOTEBOOKS = PROJECT / "notebooks"
# Executed notebooks contain outputs derived from the data; kept local, not in git.
EXECUTED_NOTEBOOKS = PROJECT / "build" / "notebooks"


def code_file(module: str) -> Path:
    """Path to a package module, e.g. code_file("train_only_study.py")."""
    return PACKAGE / module
