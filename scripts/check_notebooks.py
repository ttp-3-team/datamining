"""Виконати ноутбуки без Jupyter і зберегти виконані копії в build/notebooks/.

Кожен ноутбук виконується в окремому процесі. Ноутбуки в notebooks/ лишаються
без виводів (у виводах можуть бути рядки даних, а датасет не публікується).

    uv run python scripts/check_notebooks.py 05_train_only_nested_study [06_...]
"""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from disputes.paths import EXECUTED_NOTEBOOKS, NOTEBOOKS, PROJECT


def execute_notebook(path, target):
    import IPython.display

    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    os.chdir(PROJECT)
    execution_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        execution_count += 1
        outputs = []
        buffer = io.StringIO()

        def flush():
            text = buffer.getvalue()
            if text:
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
                buffer.seek(0)
                buffer.truncate(0)

        def display(*objects, **kwargs):
            flush()
            for obj in objects:
                data = {"text/plain": repr(obj)}
                if hasattr(obj, "_repr_html_"):
                    html = obj._repr_html_()
                    if html:
                        data["text/html"] = html
                outputs.append(nbformat.v4.new_output("display_data", data=data, metadata={}))

        IPython.display.display = display
        with contextlib.redirect_stdout(buffer):
            exec(compile(cell.source, f"{path.name}:cell-{execution_count}", "exec"), namespace)
        flush()
        cell.outputs = outputs
        cell.execution_count = execution_count
    notebook.metadata["verification"] = {
        "method": "sequential Python cells in a fresh subprocess; not a Jupyter or Colab session",
        "python": sys.version.split()[0],
    }
    nbformat.validate(notebook)
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    print(f"{path.name}: {execution_count} code cells OK", flush=True)


def notebook_path(name):
    return NOTEBOOKS / (Path(name).stem + ".ipynb")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in NOTEBOOKS.glob("*.ipynb"))
    if len(names) == 1:
        source = notebook_path(names[0])
        execute_notebook(source, EXECUTED_NOTEBOOKS / source.name)
    else:
        for name in names:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), name], check=True, cwd=PROJECT)
