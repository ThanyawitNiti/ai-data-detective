"""Build the deterministic XAU/USD Parquet fixture."""

from pathlib import Path

import duckdb

OUTPUT_PATH = Path("data/xauusd_daily_sample.parquet")

ROWS = [
    ("2025-01-02", "XAU/USD", 2624.0, 2638.0, 2618.0, 2632.0, 10120),
    ("2025-01-03", "XAU/USD", 2632.0, 2647.0, 2627.0, 2640.0, 10840),
    ("2025-01-06", "XAU/USD", 2641.0, 2650.0, 2631.0, 2637.0, 11210),
    ("2025-01-07", "XAU/USD", 2637.0, 2662.0, 2633.0, 2655.0, 11980),
    ("2025-01-08", "XAU/USD", 2655.0, 2669.0, 2646.0, 2662.0, 11540),
    # Known anomaly: a deterministic -7.96% close-to-close move.
    ("2025-01-09", "XAU/USD", 2660.0, 2664.0, 2435.0, 2450.0, 45200),
    ("2025-01-10", "XAU/USD", 2452.0, 2482.0, 2441.0, 2474.0, 18700),
    ("2025-01-13", "XAU/USD", 2475.0, 2490.0, 2462.0, 2483.0, 14200),
]


def main() -> None:
    """Write the fixture with explicit, reproducible types."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute(
            """
            CREATE TABLE bars (
                date DATE,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
            """
        )
        connection.executemany("INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?)", ROWS)
        connection.execute(
            "COPY bars TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(OUTPUT_PATH)]
        )

    print(f"Wrote {len(ROWS)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
