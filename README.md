# AI Data Detective

A learning project for building a quant research agent from first principles.

## Current milestone

Issue #2 establishes a deterministic XAU/USD daily dataset and the smallest
Python foundation needed to read and test it. There is intentionally no LLM or
agent framework yet.

The sample contains eight daily bars. The 2025-01-09 bar has a known abnormal
close-to-close return, giving later agent experiments a stable fact to discover.

## Setup

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Verify

```bash
python scripts/build_sample_data.py
pytest -q
ruff check .
```
