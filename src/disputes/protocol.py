"""Protocols, data-access control and run evidence shared by all studies.

A protocol is written BEFORE data are read; a rerun with a different protocol stops.
During a study a Python audit hook allows opening only the declared tables.
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
    """Value as it will look after a JSON round trip (tuples → lists etc.)."""
    return json.loads(json.dumps(value))


def write_json(path: Path | str, value) -> None:
    """Indented JSON; numpy scalars are written as plain numbers."""
    text = json.dumps(value, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    Path(path).write_text(text, encoding="utf-8")


def freeze(output: Path | str, protocol: dict, label: str = "Protocol") -> tuple[Path, str]:
    """Write protocol.json to `output`, or check that it is unchanged."""
    output = Path(output)
    output.mkdir(exist_ok=True)
    path = output / "protocol.json"
    value = canonical(protocol)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"{label} is already frozen and differs; do not overwrite after results")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    return output, sha256(path)


def code_hashes(*modules: str) -> dict[str, str]:
    """SHA-256 of package modules, keyed by path from the repository root."""
    return {code_file(m).relative_to(PROJECT).as_posix(): sha256(code_file(m)) for m in modules}


def code_hashes_match(hashes: dict[str, str]) -> bool:
    return all(sha256(PROJECT / name) == digest for name, digest in hashes.items())


def verified_csv(path: Path | str, expected: str, **kwargs) -> pd.DataFrame:
    """Read a CSV only if its checksum matches the expected one."""
    if sha256(path) != expected:
        raise ValueError(f"Checksum mismatch: {Path(path).name}")
    return pd.read_csv(path, **kwargs)


def write_tables(output: Path, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Write tables as CSV and return their checksums."""
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    return {name: sha256(output / f"{name}.csv") for name in tables}


def runtime_versions() -> dict[str, str]:
    import catboost
    import numpy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "catboost": catboost.__version__}


# --------------------------------------------------------------------------- data access control
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
        raise PermissionError(f"Protocol forbids opening {path.name}")
    allowed = policy["allowed"].get(path.parent)
    if allowed is not None:
        if path.name not in allowed:
            raise PermissionError(f"Protocol forbids opening {path.parent.name}/{path.name}")
        policy["opened"].append(path.name)


@contextmanager
def data_access(allowed: dict[Path, set[str]], forbidden_names=(), forbidden_prefixes=()):
    """Restrict which CSVs may be opened inside the block; yields the log of opened tables.

    allowed: folder → permitted files in it (other CSVs in that folder are forbidden).
    A Python audit hook cannot be removed, so there is one hook and a stack of policies.
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
    """Only outputs/train_audit.csv is allowed; validation, test and the full dataset are closed."""
    return data_access({Path(root) / "outputs": {"train_audit.csv"}},
                       forbidden_names={"dataset.csv", "validation_predictions.csv"},
                       forbidden_prefixes=("test_", "validation_", "unknown_time_"))


def main(freeze_fn, run_fn) -> None:
    """Shared study CLI: `python -m disputes.<study> [--freeze-only]`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true", help="only freeze the protocol")
    if parser.parse_args().freeze_only:
        print("Protocol frozen:", freeze_fn(PROJECT)[1])
    else:
        run_fn(PROJECT)
