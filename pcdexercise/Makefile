# Makefile for Point Cloud Tracking Exercise

# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
PLAYBACK_SCRIPT = playback.py

# Default target
.PHONY: help
help:
	@echo "Point Cloud Human Tracking - Available Commands:"
	@echo ""
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install dependencies"
	@echo "  make run         - Run the point cloud playback visualization"
	@echo "  make clean       - Remove generated files"
	@echo "  make clean-venv  - Remove virtual environment"
	@echo ""

# Create virtual environment
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR)/"
	@echo ""
	@echo "To activate, run: source $(VENV_DIR)/bin/activate"
	@echo ""

# Install dependencies
.PHONY: install
install: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install open3d numpy scipy scikit-learn matplotlib
	@echo ""
	@echo "Dependencies installed successfully!"
	@echo ""

# Run the playback script
.PHONY: run
run:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make venv' first."; \
		exit 1; \
	fi
	@echo "Starting point cloud playback..."
	@$(PYTHON) $(PLAYBACK_SCRIPT)

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Clean complete!"

# Remove virtual environment
.PHONY: clean-venv
clean-venv:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed!"

# Full clean (including venv)
.PHONY: clean-all
clean-all: clean clean-venv
	@echo "Full cleanup complete!"
