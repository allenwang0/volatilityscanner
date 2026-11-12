.PHONY: setup test lint

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/playwright install chromium

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/mypy engine.py svi.py analysis.py report.py

test:
	.venv/bin/pytest
