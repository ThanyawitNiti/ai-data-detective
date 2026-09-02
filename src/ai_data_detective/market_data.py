"""Utilities for reading deterministic market fixtures."""

from pathlib import Path

import duckdb

EXPECTED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")


def load_market_data(path: str | Path) -> list[dict[str, object]]:
    """Load XAU/USD daily bars from Parquet in stable date order."""
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Market data file not found: {parquet_path}")

    query = """
        SELECT date, symbol, open, high, low, close, volume
        FROM read_parquet(?)
        ORDER BY date
    """
    with duckdb.connect() as connection:
        cursor = connection.execute(query, [str(parquet_path)])
        columns = tuple(item[0] for item in cursor.description)
        rows = cursor.fetchall()

    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected dataset schema: {columns}")

    return [dict(zip(columns, row, strict=True)) for row in rows]
