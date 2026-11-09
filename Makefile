.PHONY: install install-dev train predict test lint format ci clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m src.rail_eta.pipeline

predict:
	python -m src.rail_eta.cli predict --distance $(DISTANCE) --stations $(STATIONS)

test:
	pytest tests -v

lint:
	ruff check src tests
	black --check src tests

format:
	ruff format src tests
	black src tests

typecheck:
	mypy src

ci: install-dev lint typecheck test

clean:
	rm -rf .venv build dist *.egg-info results/figures models/*.joblib results/metrics.json results/features.csv
