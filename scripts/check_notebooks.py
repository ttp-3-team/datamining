"""Виконати Python-комірки кожного ноутбука в окремому процесі.

Не потребує запущеного Jupyter. Ноутбуки цього пакета не містять магічних команд.
Результати print/display зберігаються у стандартному форматі ipynb.
"""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys

import nbformat


ROOT = Path(__file__).resolve().parent.parent


def execute_notebook(path):
    import IPython.display

    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    os.chdir(path.parent)
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
    nbformat.write(notebook, path)
    print(f"{path.name}: {execution_count} code cells OK", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        execute_notebook(Path(sys.argv[1]).resolve())
    else:
        for filename in ["01_data_review.ipynb", "02_prepare_data.ipynb"]:
            subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), str(ROOT / "notebooks" / filename)],
                check=True, timeout=90, cwd=ROOT,
            )
