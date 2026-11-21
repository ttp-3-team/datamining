"""Синтетичні train-дані для тестів: 18 тижнів по 16 спорів, без реального датасету."""

import pandas as pd

from disputes.preparation import SOURCE_COLUMNS, normalize_source


def synthetic_source():
    rows = []
    for week in range(18):
        for j in range(16):
            rows.append([f"{week}_{j}", f"M{j % 3}", "P", "card", "visa", "fraud", "10.4",
                         "USD", 40. + j, 40. + j, week, j % 7, int(j % 3 != 0)])
    return normalize_source(pd.DataFrame(rows, columns=SOURCE_COLUMNS))
