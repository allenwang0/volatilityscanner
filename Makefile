PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup test e2e lint run report

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/playwright install chromium

test:
	$(BIN)/pytest

e2e:
	$(BIN)/pytest -m e2e

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy engine.py report.py

run:
	$(BIN)/python main.py

report:
	$(BIN)/python report.py --seed 7 --card site/card.json
	$(BIN)/python scripts/build_report.py --readme README.md
