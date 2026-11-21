"""Протоколи, контроль доступу до даних і докази запуску — спільне для всіх досліджень.

Протокол записується ДО читання даних; повторний запуск з іншим протоколом зупиняється.
Під час дослідження аудит-хук Python дозволяє відкривати лише заявлені таблиці.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from disputes.paths import PROJECT, code_file


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    """Значення таким, яким воно буде після запису в JSON (кортежі → списки тощо)."""
    return json.loads(json.dumps(value))


def write_json(path: Path | str, value) -> None:
    """JSON з відступами; скаляри numpy записуються як звичайні числа."""
    text = json.dumps(value, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    Path(path).write_text(text, encoding="utf-8")


def freeze(output: Path | str, protocol: dict, label: str = "Протокол") -> tuple[Path, str]:
    """Записати protocol.json у `output` (або перевірити, що він не змінився)."""
    output = Path(output)
    output.mkdir(exist_ok=True)
    path = output / "protocol.json"
    value = canonical(protocol)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"{label} уже зафіксований і відрізняється; не перезаписувати після результатів")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    return output, sha256(path)


def code_hashes(*modules: str) -> dict[str, str]:
    """SHA-256 модулів пакета; ключ — шлях від кореня репозиторію."""
    return {code_file(m).relative_to(PROJECT).as_posix(): sha256(code_file(m)) for m in modules}


def code_hashes_match(hashes: dict[str, str]) -> bool:
    return all(sha256(PROJECT / name) == digest for name, digest in hashes.items())


def verified_csv(path: Path | str, expected: str, **kwargs) -> pd.DataFrame:
    """Прочитати CSV лише якщо його контрольна сума збігається з очікуваною."""
    if sha256(path) != expected:
        raise ValueError(f"Контрольна сума {Path(path).name} не збігається")
    return pd.read_csv(path, **kwargs)


def write_tables(output: Path, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Записати таблиці як CSV і повернути їхні контрольні суми."""
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    return {name: sha256(output / f"{name}.csv") for name in tables}


def runtime_versions() -> dict[str, str]:
    import catboost
    import numpy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "catboost": catboost.__version__}


# --------------------------------------------------------------------------- контроль доступу
_POLICIES: list[dict] = []
_HOOK_INSTALLED = False


def _audit(event: str, args) -> None:
    if event != "open" or not _POLICIES or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.suffix != ".csv":
        return
    policy = _POLICIES[-1]
    if path.name in policy["forbidden_names"] or path.name.startswith(policy["forbidden_prefixes"]):
        raise PermissionError(f"Протокол забороняє відкривати {path.name}")
    allowed = policy["allowed"].get(path.parent)
    if allowed is not None:
        if path.name not in allowed:
            raise PermissionError(f"Протокол забороняє відкривати {path.parent.name}/{path.name}")
        policy["opened"].append(path.name)


@contextmanager
def data_access(allowed: dict[Path, set[str]], forbidden_names=(), forbidden_prefixes=()):
    """Обмежити, які CSV можна відкрити всередині блоку; повертає журнал відкритих таблиць.

    allowed: тека → дозволені в ній файли (інші CSV цієї теки заборонені).
    Аудит-хук Python не можна зняти, тому він один, а політики — стек.
    """
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    policy = {"allowed": {Path(folder).resolve(): set(names) for folder, names in allowed.items()},
              "forbidden_names": set(forbidden_names), "forbidden_prefixes": tuple(forbidden_prefixes),
              "opened": []}
    _POLICIES.append(policy)
    try:
        yield policy["opened"]
    finally:
        _POLICIES.remove(policy)


def train_only_access(root: Path):
    """Дозволено тільки outputs/train_audit.csv; validation, test і весь датасет закриті."""
    return data_access({Path(root) / "outputs": {"train_audit.csv"}},
                       forbidden_names={"dataset.csv", "validation_predictions.csv"},
                       forbidden_prefixes=("test_", "validation_", "unknown_time_"))


def main(freeze_fn, run_fn) -> None:
    """Спільний CLI досліджень: `python -m disputes.<study> [--freeze-only]`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true", help="лише зафіксувати протокол")
    if parser.parse_args().freeze_only:
        print("Протокол зафіксовано:", freeze_fn(PROJECT)[1])
    else:
        run_fn(PROJECT)
