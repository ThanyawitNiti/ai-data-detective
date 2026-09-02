from types import SimpleNamespace
from typing import Any

import pytest

from ai_data_detective.tool_calling import (
    TOOL_DEFINITIONS,
    ToolCallingError,
    answer_question,
    dispatch_tool,
)


class FakeResponses:
    def __init__(self, first_response: Any, final_text: str = "Evidence-based answer") -> None:
        self.first_response = first_response
        self.final_text = final_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return self.first_response
        return SimpleNamespace(id="response_answer", output_text=self.final_text)


class FakeClient:
    def __init__(self, first_response: Any, final_text: str = "Evidence-based answer") -> None:
        self.responses = FakeResponses(first_response, final_text)


def function_response(
    name: str,
    arguments: str,
    *,
    call_id: str = "call_123",
) -> Any:
    return SimpleNamespace(
        id="response_request",
        output=[
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=arguments,
                call_id=call_id,
            )
        ],
    )


def test_tool_definitions_use_strict_json_schemas() -> None:
    assert {tool["name"] for tool in TOOL_DEFINITIONS} == {"get_schema", "run_sql"}
    for tool in TOOL_DEFINITIONS:
        assert tool["strict"] is True
        assert tool["parameters"]["additionalProperties"] is False


def test_schema_question_dispatches_get_schema_and_returns_trace() -> None:
    client = FakeClient(function_response("get_schema", "{}"))

    result = answer_question(
        "What data is available?",
        client=client,
        model="test-model",
    )

    assert result.tool_name == "get_schema"
    assert result.tool_arguments == {}
    assert result.tool_result["table_name"] == "market_bars"
    assert result.answer == "Evidence-based answer"
    assert result.request_response_id == "response_request"
    assert result.answer_response_id == "response_answer"


def test_quant_question_dispatches_run_sql() -> None:
    query = "SELECT min(close) AS minimum_close FROM market_bars"
    client = FakeClient(function_response("run_sql", f'{{"query": "{query}"}}'))

    result = answer_question(
        "What was the minimum close?",
        client=client,
        model="test-model",
    )

    assert result.tool_name == "run_sql"
    assert result.tool_result["rows"] == [{"minimum_close": 2450.0}]


def test_tool_output_is_returned_with_original_call_id() -> None:
    client = FakeClient(function_response("get_schema", "{}", call_id="call_evidence"))

    answer_question("Describe the data.", client=client, model="test-model")

    second_request = client.responses.calls[1]
    assert second_request["previous_response_id"] == "response_request"
    assert second_request["input"][0]["type"] == "function_call_output"
    assert second_request["input"][0]["call_id"] == "call_evidence"
    assert '"table_name": "market_bars"' in second_request["input"][0]["output"]


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ToolCallingError, match="Unknown tool"):
        dispatch_tool("read_secret", {})


def test_invalid_json_arguments_are_rejected() -> None:
    client = FakeClient(function_response("run_sql", "{not-json"))

    with pytest.raises(ToolCallingError, match="not valid JSON"):
        answer_question("Find the minimum close.", client=client, model="test-model")


def test_invalid_registered_tool_arguments_are_rejected() -> None:
    with pytest.raises(ToolCallingError, match="requires exactly"):
        dispatch_tool("run_sql", {"sql": "SELECT * FROM market_bars"})


def test_multiple_tool_calls_are_rejected() -> None:
    first_response = SimpleNamespace(
        id="response_request",
        output=[
            SimpleNamespace(
                type="function_call",
                name="get_schema",
                arguments="{}",
                call_id="call_1",
            ),
            SimpleNamespace(
                type="function_call",
                name="get_schema",
                arguments="{}",
                call_id="call_2",
            ),
        ],
    )
    client = FakeClient(first_response)

    with pytest.raises(ToolCallingError, match="exactly one tool call"):
        answer_question("Describe the data.", client=client, model="test-model")


def test_tests_do_not_require_model_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = FakeClient(function_response("get_schema", "{}"))

    result = answer_question("Describe the data.", client=client, model="test-model")

    assert result.answer == "Evidence-based answer"
