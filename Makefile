# SPDX-License-Identifier: AGPL-3.0-or-later
.PHONY: test lint fmt docs bench clean build bridge preflight bandit sast install-hooks docker-build docker-run

test:
	pytest tests/ -v --cov=sc_neurocore --cov-report=term --cov-fail-under=100

test-rust:
	cargo test --manifest-path engine/Cargo.toml

test-all: test test-rust

lint:
	ruff format --check src/ tests/
	ruff check src/ tests/
	mypy --strict src/sc_neurocore/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/
	cargo fmt --manifest-path engine/Cargo.toml

install:
	pip install -e ".[dev]"

bandit:
	bandit -r src/sc_neurocore/ -c pyproject.toml -q

sast: bandit

preflight:
	python tools/preflight.py

preflight-fast:
	python tools/preflight.py --no-tests

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

bench:
	python benchmarks/benchmark_suite.py

bench-rust:
	cargo bench --manifest-path engine/Cargo.toml

bridge:
	cd bridge && maturin develop --release

build:
	python -m build

install-hooks:
	git config core.hooksPath .githooks
	@echo "Git hooks installed (.githooks/pre-push)"

docker-build:
	docker build -f deploy/Dockerfile -t sc-neurocore:latest .

docker-run:
	docker run --rm -it sc-neurocore:latest

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	cargo clean --manifest-path engine/Cargo.toml
