"""Command-line entrypoint for manually testing one research question."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ai_data_detective.tool_calling import ToolCallingError, answer_question


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the XAU/USD research assistant.")
    parser.add_argument("question", help="Research question to answer")
    args = parser.parse_args(argv)

    try:
        result = answer_question(args.question)
    except ToolCallingError as error:
        print(f"Error: {error}")
        return 1

    print(result.answer)
    print("\nExecution trace")
    print(f"Tool: {result.tool_name}")
    print(f"Arguments: {result.tool_arguments}")
    print(f"Result: {result.tool_result}")
    return 0
