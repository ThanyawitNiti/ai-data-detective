"""One-model-tool-call orchestration using the OpenAI Responses API."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from ai_data_detective.tools import get_schema, run_sql

SYSTEM_INSTRUCTIONS = """You are a careful quantitative research assistant.
Choose exactly one provided tool to answer the user's question.
Use get_schema when you need to understand available data.
Use run_sql for questions that require values or calculations.
Base the final answer only on the returned tool evidence.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_schema",
        "description": "Describe the market_bars table, columns, row count, and date range.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_sql",
        "description": (
            "Run one safe, read-only DuckDB SELECT or WITH query against market_bars."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A read-only query that references market_bars.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class ToolCallingError(RuntimeError):
    """Raised when model output violates the fixed orchestration contract."""


@dataclass(frozen=True)
class ToolCallingResult:
    """Final answer plus the evidence trail used to produce it."""

    answer: str
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_result: dict[str, Any]
    request_response_id: str
    answer_response_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ToolCallingError(f"Tool arguments are not valid JSON: {error}") from error

    if not isinstance(arguments, dict):
        raise ToolCallingError("Tool arguments must be a JSON object.")
    return arguments


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and execute one registered local tool."""
    if name == "get_schema":
        if arguments:
            raise ToolCallingError("get_schema does not accept arguments.")
        return get_schema()

    if name == "run_sql":
        if set(arguments) != {"query"}:
            raise ToolCallingError("run_sql requires exactly one 'query' argument.")
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise ToolCallingError("run_sql 'query' must be a non-empty string.")
        return run_sql(query)

    raise ToolCallingError(f"Unknown tool requested: {name}")


def _default_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ToolCallingError(
            "The OpenAI SDK is not installed. Install the project dependencies."
        ) from error
    return OpenAI()


def _extract_api_error_code(error: Exception) -> str | None:
    code = getattr(error, "code", None)
    if isinstance(code, str):
        return code

    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    payload = body.get("error", body)
    if not isinstance(payload, dict):
        return None
    body_code = payload.get("code")
    return body_code if isinstance(body_code, str) else None


def _create_response(api_client: Any, **kwargs: Any) -> Any:
    try:
        return api_client.responses.create(**kwargs)
    except Exception as error:
        code = _extract_api_error_code(error)
        status_code = getattr(error, "status_code", None)

        if code in {"credit_balance_exhausted", "insufficient_quota"}:
            raise ToolCallingError(
                "OpenAI API credits are exhausted. Add credits at "
                "https://platform.openai.com/settings/organization/billing/ "
                "and then run the command again."
            ) from error
        if status_code == 401:
            raise ToolCallingError(
                "OpenAI API authentication failed. Check OPENAI_API_KEY and create "
                "a new key if necessary."
            ) from error
        if status_code == 429:
            raise ToolCallingError(
                "OpenAI API rate limit reached. Wait briefly and try again."
            ) from error
        raise


def answer_question(
    question: str,
    *,
    client: Any | None = None,
    model: str | None = None,
) -> ToolCallingResult:
    """Answer one question through exactly one model-selected local tool."""
    if not question.strip():
        raise ToolCallingError("Question cannot be empty.")

    selected_model = model or os.getenv("OPENAI_MODEL")
    if not selected_model:
        raise ToolCallingError("Set OPENAI_MODEL or pass model explicitly.")

    api_client = client or _default_client()
    request_response = _create_response(
        api_client,
        model=selected_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=question,
        tools=TOOL_DEFINITIONS,
        tool_choice="required",
        parallel_tool_calls=False,
    )

    function_calls = [
        item for item in request_response.output if item.type == "function_call"
    ]
    if len(function_calls) != 1:
        raise ToolCallingError(
            f"Expected exactly one tool call, received {len(function_calls)}."
        )

    function_call = function_calls[0]
    arguments = _parse_arguments(function_call.arguments)
    tool_result = dispatch_tool(function_call.name, arguments)
    tool_output = json.dumps(tool_result, ensure_ascii=False, default=str)

    answer_response = _create_response(
        api_client,
        model=selected_model,
        instructions=SYSTEM_INSTRUCTIONS,
        previous_response_id=request_response.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": tool_output,
            }
        ],
    )
    if not answer_response.output_text.strip():
        raise ToolCallingError("Model returned an empty final answer.")

    return ToolCallingResult(
        answer=answer_response.output_text,
        tool_name=function_call.name,
        tool_arguments=arguments,
        tool_result=tool_result,
        request_response_id=request_response.id,
        answer_response_id=answer_response.id,
    )
