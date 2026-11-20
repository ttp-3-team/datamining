"""Перевірки підготовки; запуск: python -m unittest -v test_preparation."""

import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from preparation import (
    SOURCE_COLUMNS, calculate_baseline, export_results, fit_scale_reference,
    load_table, normalize_source, prepare_amounts, prepare_dataset, split_by_time,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "dataset.csv"
OLD_EXPORT = ROOT.parent / "handoff_20261120_005640.zip"


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(SOURCE)
        cls.clean, cls.audits, cls.reference = prepare_dataset(cls.raw)

    def test_row_conservation_and_time_split(self):
        expected = {"train": 7246, "validation": 1683, "test": 1687, "unknown_time": 64}
        self.assertEqual({k: len(v) for k, v in self.clean.items()}, expected)
        ids = pd.concat([x.dispute_id for x in self.clean.values()])
        self.assertTrue(ids.is_unique)
        self.assertEqual(set(ids), set(self.raw.dispute_id))
        self.assertLess(self.clean['train'].dispute_week.max(), self.clean['validation'].dispute_week.min())
        self.assertLess(self.clean['validation'].dispute_week.max(), self.clean['test'].dispute_week.min())

    @unittest.skipUnless(OLD_EXPORT.exists(), "Історичний архів не входить до пакета")
    def test_historical_regression_only_two_intended_amount_changes(self):
        with zipfile.ZipFile(OLD_EXPORT) as archive:
            for name in ["train", "validation"]:
                old = pd.read_csv(io.BytesIO(archive.read(f"{name}_ready.csv")))
                expected = old[SOURCE_COLUMNS].copy()
                expected.loc[old.amount_usd_needs_review, "amount_usd"] = np.nan
                actual = pd.read_csv(io.StringIO(self.clean[name].to_csv(index=False)))
                pd.testing.assert_frame_equal(expected, actual)
            old_ref = pd.read_csv(io.BytesIO(archive.read("scale_reference.csv")), index_col=0)
            pd.testing.assert_frame_equal(old_ref, self.reference, check_dtype=False, check_index_type=False)

    def test_unknown_amounts_are_unavailable_but_auditable(self):
        data, audit = self.clean['train'], self.audits['train']
        review = audit.amount_usd_needs_review
        self.assertEqual(review.sum(), 2)
        self.assertEqual(set(audit.loc[review, 'amount_usd_before_review_mask']), {46000., 11000.})
        self.assertTrue(data.loc[review, 'amount_usd'].isna().all())
        self.assertFalse(audit.loc[review, 'score_eligible'].any())

    def test_reprocessing_guard_preserves_provenance(self):
        with self.assertRaisesRegex(ValueError, 'вже оброблені'):
            prepare_amounts(self.clean['train'], self.audits['train'], self.reference)

    def test_deterministic_and_does_not_mutate_source(self):
        original = self.raw.copy(deep=True)
        clean, audits, reference = prepare_dataset(self.raw)
        pd.testing.assert_frame_equal(original, self.raw)
        for name in clean:
            pd.testing.assert_frame_equal(clean[name], self.clean[name])
            pd.testing.assert_frame_equal(audits[name], self.audits[name])
        pd.testing.assert_frame_equal(reference, self.reference)

    def test_future_values_do_not_change_train_reference(self):
        changed = self.raw.copy()
        late = pd.to_numeric(changed.dispute_week, errors='coerce').gt(17)
        changed.loc[late, 'amount_usd'] = '123456'
        changed.loc[late, 'currency'] = 'GBP'
        clean, _, reference = prepare_dataset(changed)
        pd.testing.assert_frame_equal(reference, self.reference)
        pd.testing.assert_frame_equal(clean['train'], self.clean['train'])

    def test_target_is_not_used_for_cleaning(self):
        changed = self.raw.copy()
        changed['won'] = 1 - changed.won
        clean, audits, reference = prepare_dataset(changed)
        pd.testing.assert_frame_equal(reference, self.reference)
        for name in clean:
            pd.testing.assert_frame_equal(clean[name].drop(columns='won'), self.clean[name].drop(columns='won'))
            pd.testing.assert_frame_equal(audits[name].drop(columns='raw__won'), self.audits[name].drop(columns='raw__won'))

    def test_baseline_uses_explicit_identical_subset(self):
        result = calculate_baseline(self.clean['validation'], self.audits['validation'])
        self.assertEqual(result['rows'], 1666)
        self.assertAlmostEqual(result['score_usd'], 5324.)
        observed = calculate_baseline(self.clean['validation'], self.audits['validation'], observed_only=True)
        self.assertEqual(observed['rows'], 1634)
        self.assertAlmostEqual(observed['score_usd'], 5277.)

    def test_export_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            export_results(self.clean, self.audits, self.reference, SOURCE, temp)
            whole = load_table(temp, 'dataset_clean.csv')
            whole_audit = load_table(temp, 'dataset_audit.csv')
            pd.testing.assert_frame_equal(whole, pd.concat(self.clean.values()).sort_index().reset_index(drop=True))
            self.assertTrue(whole.dispute_id.equals(whole_audit.dispute_id))
            self.assertEqual(set(whole_audit.split), set(self.clean))
            for name in self.clean:
                for suffix, source in [('clean', self.clean), ('audit', self.audits)]:
                    actual = load_table(temp, f'{name}_{suffix}.csv')
                    pd.testing.assert_frame_equal(actual, source[name].reset_index(drop=True))
            with (Path(temp) / 'train_clean.csv').open('a', encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'змінився'):
                load_table(temp, 'train_clean.csv')

    def test_invalid_dates_and_nonfinite_amounts(self):
        raw = self.raw.drop_duplicates().head(4).copy().astype({
            'dispute_week': 'string', 'dispute_weekday': 'string',
            'amount': 'string', 'amount_usd': 'string',
        })
        raw['dispute_week'] = ['1.5', '-3', '999', '0']
        raw['dispute_weekday'] = ['7', '1.5', '?', '0']
        raw['amount'] = ['inf', '10', '10', '10']
        raw['amount_usd'] = ['inf', '10', '10', '10']
        data, audit = normalize_source(raw)
        self.assertEqual(data.dispute_week.isna().sum(), 3)
        self.assertEqual(data.dispute_weekday.isna().sum(), 3)
        clean, log = prepare_amounts(data, audit, self.reference)
        self.assertTrue(clean.loc[0, ['amount', 'amount_usd']].isna().all())
        self.assertFalse(log.loc[0, 'score_eligible'])

    def test_schema_and_id_checks(self):
        bad = self.raw.drop_duplicates().head(2).copy()
        bad.loc[bad.index[1], 'dispute_id'] = bad.iloc[0].dispute_id
        with self.assertRaisesRegex(ValueError, 'dispute_id'):
            normalize_source(bad)
        with self.assertRaisesRegex(ValueError, '13 колонок'):
            normalize_source(self.raw.drop(columns='won'))

    def test_amount_rules_on_independent_examples(self):
        raw = self.raw.drop_duplicates().head(7).copy()
        raw['currency'] = ['USD', 'USD', 'USD', 'USD', 'USD', 'USD', None]
        raw['amount'] = ['50', '20', None, '20', '40', '12000', None]
        raw['amount_usd'] = ['50000', '-20', None, '21', '999999', '12000', '45']
        data, audit = normalize_source(raw)
        clean, log = prepare_amounts(data, audit, self.reference)
        self.assertEqual(clean.loc[0, 'amount_usd'], 50)
        self.assertTrue(log.loc[0, 'amount_usd_scale_corrected'])
        self.assertEqual(clean.loc[1, 'amount_usd'], 20)
        self.assertTrue(log.loc[1, 'amount_usd_restored'])
        self.assertTrue(clean.loc[2, ['amount', 'amount_usd']].isna().all())
        self.assertTrue(log.loc[3, 'usd_amount_conflict'])
        self.assertTrue(clean.loc[3, ['amount', 'amount_usd']].isna().all())
        self.assertEqual(clean.loc[4, 'amount_usd'], 40)
        self.assertTrue(clean.loc[5, ['amount', 'amount_usd']].isna().all())
        self.assertTrue(log.loc[6, 'score_eligible'])
        self.assertEqual(clean.loc[6, 'amount_usd'], 45)


if __name__ == '__main__':
    unittest.main()
