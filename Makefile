.DEFAULT_GOAL := help
.PHONY: help install up down test lint format typecheck security migrate evidence test-integration eval clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras and git hooks
	pip install -e ".[dev]"
	pre-commit install

up:  ## Start the local stack
	docker compose up --build -d

down:  ## Stop the local stack
	docker compose down -v

test:  ## Run the test suite with coverage
	pytest

lint:  ## Lint and check formatting
	ruff check .
	ruff format --check .

format:  ## Auto-fix lint and format
	ruff check --fix .
	ruff format .

typecheck:  ## Run mypy in strict mode
	mypy

security:  ## Secret scan over full history plus dependency audit
	gitleaks git --config .gitleaks.toml --redact --no-banner --log-opts="--all" .
	pip-audit --strict

migrate:  ## Bring the evidence store to the latest schema
	python -m grounded.migrate

evidence:  ## Validate and load an evidence file (FILE=evidence/private/evidence.yaml)
	python -m grounded.evidence check $${FILE:-evidence/private/evidence.yaml}
	python -m grounded.evidence load $${FILE:-evidence/private/evidence.yaml}

test-integration:  ## Postgres + pgvector tests against the compose stack (throwaway databases)
	GROUNDED_TEST_POSTGRES_URL=postgresql+psycopg://app:app@localhost:5433/app pytest -m integration --no-cov

eval:  ## Run the evaluation harness (AI projects only)
	python -m grounded.eval

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
