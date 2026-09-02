import inspect

import pytest

from ai_data_detective.tools import (
    MAX_RESULT_ROWS,
    QueryValidationError,
    get_schema,
    run_sql,
)


def test_get_schema_returns_expected_metadata() -> None:
    schema = get_schema()

    assert schema["table_name"] == "market_bars"
    assert [column["name"] for column in schema["columns"]] == [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert [column["data_type"] for column in schema["columns"]] == [
        "DATE",
        "VARCHAR",
        "DOUBLE",
        "DOUBLE",
        "DOUBLE",
        "DOUBLE",
        "BIGINT",
    ]
    assert schema["row_count"] == 8
    assert schema["min_date"] == "2025-01-02"
    assert schema["max_date"] == "2025-01-13"


def test_run_sql_returns_structured_aggregation() -> None:
    result = run_sql(
        """
        SELECT symbol, count(*) AS trading_days, round(avg(close), 2) AS average_close
        FROM market_bars
        GROUP BY symbol
        """
    )

    assert result == {
        "columns": ["symbol", "trading_days", "average_close"],
        "rows": [
            {
                "symbol": "XAU/USD",
                "trading_days": 8,
                "average_close": 2579.13,
            }
        ],
        "row_count": 1,
        "truncated": False,
    }


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM market_bars",
        "INSERT INTO market_bars VALUES ('2025-01-14', 'XAU/USD', 1, 1, 1, 1, 1)",
        "DROP TABLE market_bars",
        "CREATE TABLE copy AS SELECT * FROM market_bars",
        "COPY market_bars TO '/tmp/market.csv'",
    ],
)
def test_run_sql_rejects_non_read_only_statements(query: str) -> None:
    with pytest.raises(QueryValidationError, match="Only SELECT and WITH"):
        run_sql(query)


def test_run_sql_rejects_multiple_statements() -> None:
    with pytest.raises(QueryValidationError, match="Exactly one"):
        run_sql("SELECT * FROM market_bars; SELECT * FROM market_bars")


def test_run_sql_rejects_other_tables() -> None:
    with pytest.raises(QueryValidationError, match="disallowed tables"):
        run_sql("SELECT * FROM secret_table")


def test_run_sql_rejects_external_file_access() -> None:
    with pytest.raises(QueryValidationError, match="safely bound"):
        run_sql("SELECT * FROM read_csv_auto('/tmp/private.csv')")


def test_run_sql_does_not_accept_a_dataset_path() -> None:
    assert list(inspect.signature(run_sql).parameters) == ["query"]


def test_run_sql_applies_result_limit() -> None:
    result = run_sql(
        f"""
        SELECT market_bars.date, generated.value
        FROM market_bars
        CROSS JOIN generate_series(1, {MAX_RESULT_ROWS}) AS generated(value)
        """
    )

    assert result["row_count"] == MAX_RESULT_ROWS
    assert len(result["rows"]) == MAX_RESULT_ROWS
    assert result["truncated"] is True


def test_run_sql_supports_with_query() -> None:
    result = run_sql(
        """
        WITH daily_returns AS (
            SELECT date, close / lag(close) OVER (ORDER BY date) - 1 AS return
            FROM market_bars
        )
        SELECT date, return
        FROM daily_returns
        WHERE return IS NOT NULL
        ORDER BY return
        LIMIT 1
        """
    )

    assert result["rows"][0]["date"] == "2025-01-09"
    assert result["rows"][0]["return"] < -0.07
