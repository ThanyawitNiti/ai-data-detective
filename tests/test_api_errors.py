from types import SimpleNamespace
from typing import Any

import pytest

from ai_data_detective import cli
from ai_data_detective.tool_calling import ToolCallingError, answer_question


class FakeAPIError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str | None = None,
    ) -> None:
        super().__init__("API request failed")
        self.status_code = status_code
        self.code = code
        self.body = {"code": code} if code else {}


class FailingResponses:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **kwargs: Any) -> Any:
        raise self.error


def failing_client(error: Exception) -> Any:
    return SimpleNamespace(responses=FailingResponses(error))


def test_exhausted_credits_return_billing_message() -> None:
    client = failing_client(
        FakeAPIError(status_code=429, code="credit_balance_exhausted")
    )

    with pytest.raises(ToolCallingError, match="credits are exhausted") as raised:
        answer_question("Find the largest drop.", client=client, model="test-model")

    assert "platform.openai.com/settings/organization/billing" in str(raised.value)


def test_invalid_key_returns_authentication_message() -> None:
    client = failing_client(FakeAPIError(status_code=401))

    with pytest.raises(ToolCallingError, match="authentication failed"):
        answer_question("Find the largest drop.", client=client, model="test-model")


def test_rate_limit_returns_retry_message() -> None:
    client = failing_client(FakeAPIError(status_code=429))

    with pytest.raises(ToolCallingError, match="Wait briefly"):
        answer_question("Find the largest drop.", client=client, model="test-model")


def test_unexpected_error_is_not_hidden() -> None:
    client = failing_client(ConnectionError("network unavailable"))

    with pytest.raises(ConnectionError, match="network unavailable"):
        answer_question("Find the largest drop.", client=client, model="test-model")


def test_cli_exits_cleanly_for_expected_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_question: str) -> None:
        raise ToolCallingError("OpenAI API credits are exhausted.")

    monkeypatch.setattr(cli, "answer_question", fail)

    exit_code = cli.main(["Find the largest drop."])

    assert exit_code == 1
    assert capsys.readouterr().out == "Error: OpenAI API credits are exhausted.\n"
