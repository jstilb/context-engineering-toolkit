.PHONY: install test lint typecheck demo clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=src --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

demo:
	python -m src.cli demo

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/ .hypothesis/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
