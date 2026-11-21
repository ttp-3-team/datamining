"""Reproducible preparation of disputes. No models are trained here."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from disputes.config import TRAIN_LAST_WEEK, VALIDATION_LAST_WEEK
from disputes.economics import check_aligned, payoff
from disputes.paths import DATASET, OUTPUTS
from disputes.protocol import sha256

SOURCE_COLUMNS = [
    "dispute_id", "merchant_id", "psp", "payment_method", "card_brand",
    "reason_category", "reason_code", "currency", "amount", "amount_usd",
    "dispute_week", "dispute_weekday", "won",
]
CATEGORY_COLUMNS = [
    "merchant_id", "psp", "payment_method", "card_brand",
    "reason_category", "reason_code", "currency",
]
SENTINELS = {"amount": 9_999_999, "amount_usd": 999_999}
VERSION = "1.0.0"


@dataclass(frozen=True)
class Settings:
    train_last_week: int = TRAIN_LAST_WEEK
    validation_last_week: int = VALIDATION_LAST_WEEK
    min_reference_amount: float = 10
    min_reference_pairs: int = 30
    reference_margin: float = 0.05
    review_factor: float = 100
    scale_divisor: float = 1000
    review_threshold: float = 10_000
    usd_tolerance: float = 0.01


def normalize_source(raw):
    """Normalise formats; keep the original values in a separate audit table."""
    if set(raw.columns) != set(SOURCE_COLUMNS):
        raise ValueError("Expected exactly 13 columns of the original dataset.csv.")
    data = raw[SOURCE_COLUMNS].drop_duplicates().copy().reset_index(drop=True)
    audit = data.add_prefix("raw__")
    for column in ["dispute_id", *CATEGORY_COLUMNS]:
        data[column] = (
            data[column].astype("string").str.strip()
            .replace({"": pd.NA, "-": pd.NA, "?": pd.NA})
        )
    if data.dispute_id.isna().any() or not data.dispute_id.is_unique:
        raise ValueError("Empty or duplicate dispute_id values require separate handling.")
    audit.insert(0, "dispute_id", data.dispute_id)
    target = pd.to_numeric(data.won, errors="coerce")
    if not target.isin([0, 1]).all():
        raise ValueError("won must contain only 0 or 1 with no missing values.")
    data["won"] = target.astype("int64")

    for column in ["payment_method", "card_brand", "reason_category"]:
        data[column] = data[column].str.lower()
    for column in ["merchant_id", "psp", "reason_code", "currency"]:
        data[column] = data[column].str.upper()
    data["payment_method"] = data.payment_method.replace({"pay pal": "paypal"})
    data["card_brand"] = data.card_brand.replace({"mastercrad": "mastercard", "mc": "mastercard"})
    data["reason_category"] = data.reason_category.str.replace(r"\s+", "_", regex=True)
    audit["currency_dollar_assumed_usd"] = data.currency.eq("$").fillna(False)
    data["currency"] = data.currency.replace({"$": "USD", "US$": "USD", "€": "EUR"})
    paypal = data.payment_method.eq("paypal").fillna(False)
    audit["card_brand_restored"] = paypal & data.card_brand.isna()
    data.loc[audit.card_brand_restored, "card_brand"] = "not_card"
    audit["payment_brand_conflict"] = (
        (data.payment_method.eq("card") & data.card_brand.eq("not_card"))
        | (paypal & data.card_brand.notna() & data.card_brand.ne("not_card"))
    ).fillna(False)
    audit["reason_category_unexpected"] = data.reason_category.notna() & ~data.reason_category.isin(
        ["fraud", "authorization", "processing_error", "consumer_dispute", "other"]
    )

    for column, sentinel in SENTINELS.items():
        tokens = data[column].astype("string").str.strip()
        missing = tokens.isna() | tokens.isin(["", "-", "?"])
        values = pd.to_numeric(tokens.mask(missing), errors="coerce").astype("float64")
        audit[f"source__{column}_missing"] = missing
        audit[f"source__{column}_parse_failed"] = ~missing & values.isna()
        audit[f"source__{column}_nonfinite"] = values.notna() & ~np.isfinite(values)
        audit[f"source__{column}_nonpositive"] = values.le(0)
        audit[f"source__{column}_sentinel"] = values.eq(sentinel)
        data[column] = values

    for column in ["dispute_week", "dispute_weekday"]:
        tokens = data[column].astype("string").str.strip()
        missing = tokens.isna() | tokens.isin(["", "-", "?"])
        values = pd.to_numeric(tokens.mask(missing), errors="coerce").astype("float64")
        valid = np.isfinite(values) & values.mod(1).eq(0)
        if column == "dispute_weekday":
            valid &= values.between(0, 6)
        else:
            valid &= values.ge(0) & ~values.isin([999, 2026])
        audit[f"source__{column}_missing"] = missing
        audit[f"source__{column}_invalid"] = ~missing & ~valid
        data[column] = values.where(valid).astype("Int64")
    return data, audit


def split_by_time(data, settings=Settings()):
    """Split by week: early weeks define the rules, later weeks are held out."""
    week = data.dispute_week
    masks = {
        "train": week.le(settings.train_last_week),
        "validation": week.gt(settings.train_last_week) & week.le(settings.validation_last_week),
        "test": week.gt(settings.validation_last_week),
        "unknown_time": week.isna(),
    }
    parts = {name: data.loc[mask.fillna(False)].copy() for name, mask in masks.items()}
    ids = pd.concat([part.dispute_id for part in parts.values()])
    if len(ids) != len(data) or not ids.is_unique:
        raise ValueError("Split lost or duplicated disputes.")
    if parts["train"].empty:
        raise ValueError("No early data for the reference statistics.")
    return parts


def fit_scale_reference(train, settings=Settings()):
    """Compute the reference on the early part only, without using won."""
    valid = train.currency.notna()
    for column, sentinel in SENTINELS.items():
        valid &= np.isfinite(train[column]) & train[column].ge(settings.min_reference_amount)
        valid &= train[column].ne(sentinel)
    pairs = train.loc[valid, ["currency", "amount", "amount_usd"]].copy()
    pairs["ratio"] = pairs.amount_usd / pairs.amount
    reference = pairs.groupby("currency").ratio.agg(
        pairs="count", minimum="min", p05=lambda x: x.quantile(.05),
        median="median", p95=lambda x: x.quantile(.95), maximum="max",
    )
    reference = reference.loc[reference.pairs.ge(settings.min_reference_pairs)].copy()
    reference["lower"] = reference.p05 * (1 - settings.reference_margin)
    reference["upper"] = reference.p95 * (1 + settings.reference_margin)
    return reference


def prepare_amounts(data, audit, reference, settings=Settings()):
    """Apply the rules once; a repeated call is rejected."""
    if "preparation_version" in audit:
        raise ValueError("These data are already processed. Run preparation from dataset.csv.")
    if not data.dispute_id.equals(audit.dispute_id):
        raise ValueError("Data and audit rows do not match.")
    result, log = data.copy(), audit.copy()
    amount, usd = result.amount, result.amount_usd
    median = result.currency.map(reference["median"])
    lower = result.currency.map(reference["lower"])
    upper = result.currency.map(reference["upper"])
    corrected = usd / settings.scale_divisor
    valid_pair = (
        np.isfinite(amount) & np.isfinite(usd)
        & amount.ge(settings.min_reference_amount) & usd.ge(settings.min_reference_amount)
        & amount.ne(SENTINELS["amount"]) & usd.ne(SENTINELS["amount_usd"])
        & median.notna()
    )
    is_usd = result.currency.eq("USD").fillna(False)
    plausible = (corrected / amount).between(lower, upper)
    plausible = plausible.where(~is_usd, corrected.sub(amount).abs().le(settings.usd_tolerance))
    correction = (valid_pair & ((usd / amount) / median).ge(settings.review_factor) & plausible).fillna(False)
    log["amount_usd_before_scale"] = usd
    log["amount_usd_scale_corrected"] = correction
    result["amount_usd"] = usd.where(~correction, corrected)

    for column, sentinel in SENTINELS.items():
        values = result[column]
        invalid = values.le(0) | values.eq(sentinel) | (values.notna() & ~np.isfinite(values))
        log[f"{column}_invalid_removed"] = invalid
        result[column] = values.mask(invalid)
    usable_amount = result.amount.between(0, settings.review_threshold, inclusive="right")
    usable_usd = result.amount_usd.between(0, settings.review_threshold, inclusive="right")
    restore_usd = is_usd & result.amount_usd.isna() & usable_amount
    restore_amount = is_usd & result.amount.isna() & usable_usd
    result.loc[restore_usd, "amount_usd"] = result.loc[restore_usd, "amount"]
    result.loc[restore_amount, "amount"] = result.loc[restore_amount, "amount_usd"]
    log["amount_usd_restored"] = restore_usd
    log["amount_restored"] = restore_amount
    conflict = (
        is_usd & result.amount.notna() & result.amount_usd.notna()
        & result.amount_usd.sub(result.amount).abs().gt(settings.usd_tolerance)
    )
    log["usd_amount_conflict"] = conflict
    for column in SENTINELS:
        log[f"{column}_before_review_mask"] = result[column]
        log[f"{column}_needs_review"] = result[column].gt(settings.review_threshold) | conflict
        result[column] = result[column].mask(log[f"{column}_needs_review"])
        log[f"current__{column}_missing"] = result[column].isna()

    log["score_eligible"] = result.amount_usd.gt(0) & np.isfinite(result.amount_usd)
    log["score_observed_only"] = log.score_eligible & ~correction & ~restore_usd
    log["amount_usd_origin"] = np.select(
        [log.amount_usd_needs_review, result.amount_usd.isna(), correction, restore_usd],
        ["needs_review", "unavailable", "scale_corrected", "restored_from_usd_amount"],
        default="observed",
    )
    log["preparation_version"] = VERSION
    return result[SOURCE_COLUMNS], log


def prepare_dataset(raw, settings=Settings()):
    data, source_audit = normalize_source(raw)
    parts = split_by_time(data, settings)
    reference = fit_scale_reference(parts["train"], settings)
    clean, audits = {}, {}
    for name, part in parts.items():
        clean[name], audits[name] = prepare_amounts(
            part, source_audit.loc[part.index], reference, settings
        )
    return clean, audits, reference


def quality_summary(clean, audits):
    rows = []
    for name, data in clean.items():
        audit = audits[name]
        rows.append({
            "split": name, "rows": len(data),
            "first_week": data.dispute_week.min(), "last_week": data.dispute_week.max(),
            "amount_missing": int(data.amount.isna().sum()),
            "amount_usd_missing": int(data.amount_usd.isna().sum()),
            "scale_corrected": int(audit.amount_usd_scale_corrected.sum()),
            "amount_restored": int(audit.amount_restored.sum()),
            "amount_usd_restored": int(audit.amount_usd_restored.sum()),
            "amount_usd_needs_review": int(audit.amount_usd_needs_review.sum()),
            "score_eligible": int(audit.score_eligible.sum()),
        })
    return pd.DataFrame(rows)


def calculate_baseline(data, audit, observed_only=False):
    """Always-fight Score on an explicitly specified subset of amounts."""
    check_aligned(data, audit)
    mask = audit["score_observed_only" if observed_only else "score_eligible"].to_numpy(dtype=bool)
    values = data.amount_usd.to_numpy(dtype=float)[mask]
    if not (np.isfinite(values) & (values > 0)).all():
        raise ValueError("Score eligibility flag does not match available amounts.")
    return {"rows": int(mask.sum()), "total_rows": len(data), "score_usd": float(payoff(data)[mask].sum())}


def export_results(clean, audits, reference, source_path, output_dir, settings=Settings()):
    """Write data, audit, read schemas and checksums."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preparation_version": VERSION, "settings": asdict(settings),
        "source_sha256": sha256(Path(source_path)),
        "pandas_version": pd.__version__, "numpy_version": np.__version__, "files": {},
        "notes": [
            "No model was trained. Missing values are kept; no encoding or scaling applied.",
            "The scale reference is computed on train only and applied to all splits.",
            "Test is processed, but no Score or model quality is computed on it.",
        ],
    }
    tables = {f"{name}_clean.csv": value for name, value in clean.items()}
    tables.update({f"{name}_audit.csv": value for name, value in audits.items()})
    tables["dataset_clean.csv"] = pd.concat(clean.values()).sort_index()
    tables["dataset_audit.csv"] = pd.concat(
        [audit.assign(split=name) for name, audit in audits.items()]
    ).sort_index()
    tables["scale_reference.csv"] = reference.reset_index()
    tables["quality_summary.csv"] = quality_summary(clean, audits)
    for filename, table in tables.items():
        path = output / filename
        table.to_csv(path, index=False, encoding="utf-8")
        manifest["files"][filename] = {
            "rows": len(table), "columns": list(table.columns),
            "dtypes": {col: str(dtype) for col, dtype in table.dtypes.items()},
            "sha256": sha256(path),
        }
    features = {
        "target": "won", "identifier": "dispute_id",
        "categorical_candidates": CATEGORY_COLUMNS,
        "numeric_candidates": ["amount", "amount_usd", "dispute_week", "dispute_weekday"],
        "excluded_from_automatic_input": ["dispute_id", "won", "all audit columns"],
        "notes": [
            "Available candidates, not a final feature matrix or model choice.",
            "amount is in different currencies; do not compare it as a single money scale.",
            "Codes and merchant IDs are categorical. Missing values must be handled in the modelling pipeline.",
        ],
    }
    (output / "feature_contract.json").write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_table(output_dir, filename):
    """Read a CSV with restored dtypes and a checksum check."""
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest["files"][filename]
    path = output / filename
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"File {filename} changed after export.")
    # raw__ fields keep literal '?' and '-'; only an empty field means missing.
    return pd.read_csv(path, dtype=spec["dtypes"], keep_default_na=False, na_values=[""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    clean, audits, reference = prepare_dataset(raw)
    export_results(clean, audits, reference, args.input, args.output)
    print(quality_summary(clean, audits).to_string(index=False))


if __name__ == "__main__":
    main()
