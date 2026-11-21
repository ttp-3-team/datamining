"""Виконати ноутбуки в чистому ядрі (nbclient) і зберегти виконані копії в build/notebooks/.

Ноутбуки в notebooks/ лишаються без виводів: у виводах можуть бути рядки даних, а датасет не публікується.

    uv run python scripts/check_notebooks.py 05_train_only_nested_study [06_catboost_train_only ...]
    uv run python scripts/check_notebooks.py            # усі
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from disputes.paths import EXECUTED_NOTEBOOKS, NOTEBOOKS, PROJECT


def execute(name: str) -> Path:
    source = NOTEBOOKS / f"{Path(name).stem}.ipynb"
    notebook = nbformat.read(source, as_version=4)
    NotebookClient(notebook, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(PROJECT)}}).execute()
    target = EXECUTED_NOTEBOOKS / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"{source.name}: {cells} code cells OK -> {target.relative_to(PROJECT)}", flush=True)
    return target


if __name__ == "__main__":
    for name in sys.argv[1:] or sorted(p.stem for p in NOTEBOOKS.glob("*.ipynb")):
        execute(name)
