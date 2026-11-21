"""Де лежать дані, код і результати. Усі шляхи — від кореня репозиторію."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[1]
DATASET = PROJECT / "dataset.csv"
OUTPUTS = PROJECT / "outputs"
NOTEBOOKS = PROJECT / "notebooks"
# Виконані копії ноутбуків (з виводами) — лише локально, не в git.
EXECUTED_NOTEBOOKS = PROJECT / "build" / "notebooks"


def code_file(module: str) -> Path:
    """Шлях до модуля пакета, напр. code_file("train_only_study.py")."""
    return PACKAGE / module
