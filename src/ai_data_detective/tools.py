"""Bounded DuckDB tools exposed to the future research agent."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

DATA_PATH = Path("data/xauusd_daily_sample.parquet")
TABLE_NAME = "market_bars"
MAX_RESULT_ROWS = 100
ALLOWED_TABLES = frozenset({TABLE_NAME})


class QueryValidationError(ValueError):
    """Raised when a query falls outside the read-only tool contract."""


def _create_sandboxed_connection() -> duckdb.DuckDBPyConnection:
    """Load the controlled fixture, then query it without external file access."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Market data file not found: {DATA_PATH}")

    with duckdb.connect() as loader:
        rows = loader.execute(
            """
            SELECT date, symbol, open, high, low, close, volume
            FROM read_parquet(?)
            """,
            [str(DATA_PATH)],
        ).fetchall()

    connection = duckdb.connect(config={"enable_external_access": False})
    connection.execute(
        """
        CREATE TABLE market_bars (
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
    connection.executemany("INSERT INTO market_bars VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return connection


def _validate_query(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> str:
    if not query.strip():
        raise QueryValidationError("SQL query cannot be empty.")

    try:
        statements = connection.extract_statements(query)
    except duckdb.Error as error:
        raise QueryValidationError(f"Invalid SQL: {error}") from error

    if len(statements) != 1:
        raise QueryValidationError("Exactly one SQL statement is allowed.")

    statement = statements[0]
    if statement.type != duckdb.StatementType.SELECT:
        raise QueryValidationError("Only SELECT and WITH queries are allowed.")

    try:
        referenced_tables = connection.get_table_names(statement.query)
    except duckdb.Error as error:
        raise QueryValidationError(f"Query cannot be safely bound: {error}") from error

    if not referenced_tables:
        raise QueryValidationError(f"Query must read from {TABLE_NAME}.")

    disallowed_tables = referenced_tables - ALLOWED_TABLES
    if disallowed_tables:
        names = ", ".join(sorted(disallowed_tables))
        raise QueryValidationError(f"Query references disallowed tables: {names}.")

    return statement.query


def _serialize(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def get_schema() -> dict[str, Any]:
    """Return stable metadata for the controlled market table."""
    with _create_sandboxed_connection() as connection:
        column_rows = connection.execute("PRAGMA table_info('market_bars')").fetchall()
        row_count, min_date, max_date = connection.execute(
            "SELECT count(*), min(date), max(date) FROM market_bars"
        ).fetchone()

    return {
        "table_name": TABLE_NAME,
        "columns": [
            {"name": row[1], "data_type": row[2]}
            for row in column_rows
        ],
        "row_count": row_count,
        "min_date": _serialize(min_date),
        "max_date": _serialize(max_date),
    }


def run_sql(query: str) -> dict[str, Any]:
    """Run one bounded, read-only query and return JSON-friendly results."""
    with _create_sandboxed_connection() as connection:
        validated_query = _validate_query(connection, query)
        bounded_query = (
            f"SELECT * FROM ({validated_query}) AS agent_result "
            f"LIMIT {MAX_RESULT_ROWS + 1}"
        )
        try:
            cursor = connection.execute(bounded_query)
            columns = [item[0] for item in cursor.description]
            raw_rows = cursor.fetchall()
        except duckdb.Error as error:
            raise QueryValidationError(f"Query failed: {error}") from error

    truncated = len(raw_rows) > MAX_RESULT_ROWS
    raw_rows = raw_rows[:MAX_RESULT_ROWS]
    rows = [
        {
            column: _serialize(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in raw_rows
    ]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
