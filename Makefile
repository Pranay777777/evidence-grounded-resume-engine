.DEFAULT_GOAL := help
.PHONY: help install up down test lint format typecheck security serve migrate evidence index search ablate calibrate draft test-integration collect label eval clean

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

serve:  ## Run the API and admin at http://127.0.0.1:8000 (loopback only until auth lands)
	python -m grounded

migrate:  ## Bring the evidence store to the latest schema
	python -m grounded.migrate

evidence:  ## Validate and load an evidence file (FILE=evidence/private/evidence.yaml)
	python -m grounded.evidence check $${FILE:-evidence/private/evidence.yaml}
	python -m grounded.evidence load $${FILE:-evidence/private/evidence.yaml}

index:  ## Embed citable evidence (EMBEDDER=bge-small needs the [embeddings] extra)
	python -m grounded.retrieval index

search:  ## Hybrid search: make search Q="job description text"
	python -m grounded.retrieval search "$(Q)"

ablate:  ## Retrieval ablation on the synthetic benchmark -> docs/results/retrieval-ablation.md
	python -m grounded.retrieval ablate benchmarks/retrieval/queries.yaml --corpus benchmarks/retrieval/corpus.yaml --out docs/results/retrieval-ablation.md

calibrate:  ## Measure the entailment verifier -> docs/results/verifier-calibration.md
	python -m grounded.verification calibrate benchmarks/verifier/pairs.yaml --out docs/results/verifier-calibration.md

draft:  ## Draft bullets for a job description: make draft JD=job.txt (needs OPENROUTER_API_KEY)
	python -m grounded.generation draft --jd $(JD)

test-integration:  ## Postgres + pgvector tests against the compose stack (throwaway databases)
	GROUNDED_TEST_POSTGRES_URL=postgresql+psycopg://app:app@localhost:5433/app pytest -m integration --no-cov

collect:  ## Generate golden-set bullets: make collect MODEL=<pinned-id> (needs OPENROUTER_API_KEY)
	python -m grounded.evals collect --model $(MODEL)

label:  ## Label golden-set bullets (resumable): make label BY="Your Name"
	python -m grounded.evals label --by "$(BY)"

eval:  ## Regression gate (as CI runs it): verifier calibration, then the golden set once labelled
	python -m grounded.verification calibrate benchmarks/verifier/pairs.yaml --out docs/results/verifier-calibration.md --max-false-accept 0 --max-false-reject 0
	python -m grounded.evals run --out docs/results/eval.md --check benchmarks/golden/limits.yaml --skip-if-unlabelled

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
