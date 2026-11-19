.PHONY: install lint test theory synthetic data prepare causal fit main main-smoke main-full clean

install:
	python -m pip install -e ".[dev,hmm,causal]"

lint:
	ruff check .
	ruff format --check src/cli.py tests/test_release_tooling.py

test:
	pytest -q

theory:
	dlcssm theory-checks

synthetic:
	dlcssm synthetic --scenario shared_low_rank --output outputs/checkpoints/synthetic_demo.npz

data:
	dlcssm download --config config/data.yaml

prepare:
	dlcssm prepare --config config/data.yaml

causal:
	dlcssm causal --data data/processed/dataset.npz --config config/causal.yaml --output outputs/checkpoints/causal_support.npz --jobs 8

fit:
	dlcssm fit --data data/processed/dataset.npz --model-config config/dlcssm.yaml --support outputs/checkpoints/causal_support.npz --output outputs/checkpoints/dlcssm_train.npz

main:
	python -B scripts/run_main.py --config config/main_runner.json --output outputs/main_run

main-smoke:
	python -B scripts/run_main.py --config config/main_runner_smoke.json --output outputs/main_smoke --allow-incomplete

main-full:
	bash scripts/run_main_full.sh config/main_runner.json outputs/main_run

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
