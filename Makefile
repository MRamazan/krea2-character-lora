install:
	python -m pip install -e ".[dev]"

notebook:
	python scripts/build_notebook.py

lint:
	ruff check .
	ruff format --check .

test:
	pytest

check:
	ruff check .
	ruff format --check .
	pytest
	python scripts/build_notebook.py --check
