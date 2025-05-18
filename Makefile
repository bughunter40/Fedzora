# Makefile for Federated Learning Framework

# Python interpreter
PYTHON = python

# Project directories
SRC_DIR = src
TEST_DIR = tests
DOCS_DIR = docs

# Virtual environment
VENV = venv
VENV_BIN = $(VENV)/Scripts
PIP = $(VENV_BIN)/pip

.PHONY: all setup clean test lint docs run-server

all: setup test

# Setup virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Clean up generated files and virtual environment
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage

# Run tests
test:
	$(VENV_BIN)/pytest $(TEST_DIR) --cov=$(SRC_DIR)

# Run linting
lint:
	$(VENV_BIN)/flake8 $(SRC_DIR)
	$(VENV_BIN)/mypy $(SRC_DIR)

# Generate documentation
docs:
	$(VENV_BIN)/sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html

# Run federated learning server
run-server:
	$(VENV_BIN)/python -m $(SRC_DIR).server.federated_server

# Install development dependencies
dev-setup: setup
	$(PIP) install -r requirements-dev.txt

# Run all checks (linting and tests)
check: lint test

# Create distribution package
dist: clean
	$(VENV_BIN)/python setup.py sdist bdist_wheel

# Help target
help:
	@echo "Available targets:"
	@echo "  setup      : Set up virtual environment and install dependencies"
	@echo "  clean      : Remove generated files and virtual environment"
	@echo "  test       : Run tests with coverage"
	@echo "  lint       : Run code linting and type checking"
	@echo "  docs       : Generate documentation"
	@echo "  run-server : Start the federated learning server"
	@echo "  dev-setup  : Install development dependencies"
	@echo "  check      : Run all checks (linting and tests)"
	@echo "  dist       : Create distribution package"
	@echo "  help       : Show this help message"