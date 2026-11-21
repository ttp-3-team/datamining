"""Фіксація протоколів і контрольні суми — спільне для всіх досліджень.

Протокол записується ДО читання даних. Повторний запуск із іншим протоколом
зупиняється, щоб правила не змінювали після перегляду результатів.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from disputes.paths import PROJECT, code_file


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(output: Path | str, protocol: dict, label: str = "Протокол") -> tuple[Path, str]:
    """Записати protocol.json у `output` (або перевірити, що він не змінився)."""
    output = Path(output)
    output.mkdir(exist_ok=True)
    path = output / "protocol.json"
    canonical = json.loads(json.dumps(protocol))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != canonical:
            raise ValueError(f"{label} уже зафіксований і відрізняється; не перезаписувати після результатів")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(canonical, stream, ensure_ascii=False, indent=2)
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
