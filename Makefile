PY      := .venv/bin/python
PIP     := .venv/bin/pip
CPU_IDX := https://download.pytorch.org/whl/cpu

.PHONY: help venv install lint type test gate fixtures vendor lock wheelhouse bundle \
        selftest airgap demo clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

venv: ## create .venv
	python3 -m venv .venv && $(PIP) install -q --upgrade pip

install: venv ## install CPU-only torch FIRST, then everything else (§5.7)
	$(PIP) install --index-url $(CPU_IDX) torch torchvision
	$(PIP) install -e ".[dev,seal]"
	@$(PIP) list | grep -i '^nvidia' && { echo "CUDA wheel leaked — see §5.7"; exit 1; } || true

lint: ## ruff
	.venv/bin/ruff check cva attacklab tests

type: ## mypy, strict on core/
	.venv/bin/mypy cva/core

test: ## the fast gate
	$(PY) -m pytest -q -m "not slow and not corpus"

gate: lint type test ## everything CI runs

fixtures: ## deterministic demo + selftest corpora (committed as a generator, not as bytes)
	$(PY) -m cva.fixtures

vendor: ## Mode A only: fetch the pinned backbone artefacts and verify their SHA-256
	$(PY) -m cva.features.vendor fetch

lock: ## resolve the CPU-pinned lockfile (extras data, seal, network)
	$(PIP) install -q pip-tools
	.venv/bin/pip-compile -q --extra-index-url $(CPU_IDX) -c constraints-cpu.txt \
		--extra data --extra seal --extra network --allow-unsafe \
		--output-file requirements.lock pyproject.toml build-tools.in
	@! grep -Eiq '^(nvidia|triton)' requirements.lock || { echo "CUDA wheel in lock — see §5.7"; exit 1; }

# --extra-index-url, not --index-url: the CPU index carries torch and a few of its
# dependencies only, so making it the SOLE index fails on scipy, onnxruntime and the rest.
wheelhouse: lock ## the exact bytes the air-gapped host installs from
	$(PIP) wheel --only-binary=:all: --extra-index-url $(CPU_IDX) \
		-r requirements.lock -w wheelhouse

bundle: ## cva-bundle.tar — wheelhouse, weights, fixtures, generated manifest (< 4 GB)
	$(PY) -m cva.bundle

selftest: ## process-level egress guard, armed programmatically
	$(PY) -m cva.cli selftest

airgap: ## V11 — the OS-level guard; the only one that proves ABSENCE
	unshare -rn $(PY) -m cva.cli selftest

demo: fixtures ## the <10 min cold path, step by step
	$(PY) -m cva.cli scan --dataset artifacts/fixtures/demo_coco \
		--model artifacts/fixtures/demo_model.onnx --profile baseline --out reports

clean:
	rm -rf .pytest_cache **/__pycache__ wheelhouse cva-bundle.tar
