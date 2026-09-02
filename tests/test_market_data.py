from datetime import date
from pathlib import Path

from ai_data_detective.market_data import EXPECTED_COLUMNS, load_market_data

DATA_PATH = Path("data/xauusd_daily_sample.parquet")


def test_sample_dataset_schema_and_order() -> None:
    rows = load_market_data(DATA_PATH)

    assert len(rows) == 8
    assert tuple(rows[0]) == EXPECTED_COLUMNS
    assert rows[0]["date"] == date(2025, 1, 2)
    assert rows[-1]["date"] == date(2025, 1, 13)


def test_sample_dataset_contains_known_anomaly() -> None:
    rows = load_market_data(DATA_PATH)
    returns = {
        current["date"]: current["close"] / previous["close"] - 1
        for previous, current in zip(rows, rows[1:], strict=False)
    }

    anomaly_date = date(2025, 1, 9)
    assert returns[anomaly_date] < -0.07
    assert all(abs(value) < 0.02 for day, value in returns.items() if day != anomaly_date)
