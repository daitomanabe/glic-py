# GLIC Makefile
# Common development and distribution tasks

.PHONY: help install install-dev install-gui clean test lint format build publish gui demo

# Default target
help:
	@echo "GLIC - GLitch Image Codec"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Installation:"
	@echo "  install      Install package in current environment"
	@echo "  install-dev  Install with development dependencies"
	@echo "  install-gui  Install with GUI (Gradio) support"
	@echo "  install-all  Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test         Run tests"
	@echo "  lint         Run linter (flake8)"
	@echo "  format       Format code (black)"
	@echo "  clean        Remove build artifacts"
	@echo ""
	@echo "Distribution:"
	@echo "  build        Build package (wheel and sdist)"
	@echo "  publish      Publish to PyPI (requires credentials)"
	@echo ""
	@echo "Running:"
	@echo "  gui          Launch web GUI"
	@echo "  demo         Run demo"
	@echo "  list-presets List all available presets"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gui:
	pip install -e ".[gui]"

install-all:
	pip install -e ".[all]"

# Development
test:
	python -m pytest tests/ -v

lint:
	flake8 glic/ --max-line-length=100 --ignore=E501,W503

format:
	black glic/ examples/ --line-length=100

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf glic/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf glic/__pycache__/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Distribution
build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# For test PyPI
publish-test: build
	python -m twine upload --repository testpypi dist/*

# Running
gui:
	python -m glic.gui

demo:
	python -c "import glic; glic.demo()"

list-presets:
	python -m glic --list-presets

# Quick test with sample image
test-quick:
	@if [ -f "sample.png" ]; then \
		python -c "import glic; glic.glitch('sample.png', 'sample_glitched.png')"; \
		echo "Created sample_glitched.png"; \
	else \
		echo "No sample.png found. Place an image named sample.png in this directory."; \
	fi
