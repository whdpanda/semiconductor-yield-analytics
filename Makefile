# Semiconductor Yield & Process Analytics Platform
# Usage: make <target>
# On Windows: install GNU Make via choco (choco install make) or run commands directly.

PYTHON := python
PIP    := pip
SRC    := src
TEST   := tests

.PHONY: install install-dev lint format test test-cov \
        generate-data train-wafer run-spc dashboard clean help

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

# ── Code quality ───────────────────────────────────────────────────────────────

lint:
	ruff check $(SRC) $(TEST)

format:
	black $(SRC) $(TEST) scripts app

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	pytest $(TEST)

test-cov:
	pytest $(TEST) --cov=$(SRC)/semiconductor_yield --cov-report=html --cov-report=term-missing

# ── Data pipeline ──────────────────────────────────────────────────────────────

generate-data:
	$(PYTHON) scripts/generate_synthetic_data.py

# ── Training ───────────────────────────────────────────────────────────────────

train-wafer:
	$(PYTHON) scripts/train_module_a.py --config configs/module_a.yaml

run-spc:
	$(PYTHON) scripts/run_module_b_pipeline.py --config configs/module_b.yaml --stage full

# ── Dashboard ──────────────────────────────────────────────────────────────────

dashboard:
	streamlit run app/main.py

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Available targets:"
	@echo "  install        Install package in editable mode"
	@echo "  install-dev    Install package + dev dependencies"
	@echo "  lint           Run ruff linter"
	@echo "  format         Run black formatter"
	@echo "  test           Run pytest"
	@echo "  test-cov       Run pytest with coverage report"
	@echo "  generate-data  Generate synthetic SPC data"
	@echo "  train-wafer    Train wafer map CNN (Module A)"
	@echo "  run-spc        Run SPC + anomaly detection pipeline (Module B)"
	@echo "  dashboard      Launch Streamlit dashboard"
	@echo "  clean          Remove build artifacts and cache"
	@echo ""
