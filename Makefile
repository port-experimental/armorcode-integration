.PHONY: help install install-dev test test-unit test-integration lint format clean build dist upload

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test:  ## Run all tests
	pytest tests/ -v

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

test-coverage:  ## Run tests with coverage report
	pytest tests/ --cov=src/armorcode_integration --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 src/armorcode_integration tests/
	mypy src/armorcode_integration

format:  ## Format code with black and isort
	black src/armorcode_integration tests/
	isort src/armorcode_integration tests/

format-check:  ## Check code formatting without changing files
	black --check src/armorcode_integration tests/
	isort --check-only src/armorcode_integration tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python -m build

dist: clean build  ## Create distribution packages
	@echo "Distribution packages created in dist/"

upload-test:  ## Upload to TestPyPI
	python -m twine upload --repository testpypi dist/*

upload:  ## Upload to PyPI
	python -m twine upload dist/*

run:  ## Run the application
	python -m armorcode_integration

run-help:  ## Show application help
	python -m armorcode_integration --help 