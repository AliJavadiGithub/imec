# ==========================================
# Makefile for 3D Point Cloud Human Tracking
# ==========================================

# ---- Variables ----
VENV_DIR := .venv
PYTHON   := $(VENV_DIR)/bin/python
PIP      := $(VENV_DIR)/bin/pip

TRACKING_SCRIPT        := tracking-reid.py
PLAYBACK_SCRIPT        := tracking-playback.py
VISUALIZATION_SCRIPT   := tracking-visualization.py
EVALUATION_SCRIPT      := tracking-evaluate-mot.py

# ---- Default Target ----
.PHONY: help
help:
	@echo "3D Point Cloud Human Tracking - Available Commands:"
	@echo ""
	@echo "  make venv            - Create virtual environment"
	@echo "  make install         - Install core dependencies"
	@echo "  make install-reid    - Install optional ReID (CLIP) dependencies"
	@echo "  make track           - Run human tracking (generate JSON)"
	@echo "  make playback        - Visualize tracking & record video"
	@echo "  make visualize       - Plot trajectories & speed"
	@echo "  make evaluate        - Run multi-human MOT evaluation"
	@echo "  make clean           - Remove generated cache files"
	@echo "  make clean-venv      - Remove virtual environment"
	@echo "  make clean-all       - Full cleanup"
	@echo ""

# ---- Virtual Environment ----
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR)/"
	@echo "Activate with: source $(VENV_DIR)/bin/activate"
	@echo ""

# ---- Install Core Dependencies ----
.PHONY: install
install: venv
	@echo "Installing core dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install \
		open3d \
		numpy \
		scipy \
		scikit-learn \
		matplotlib \
		filterpy \
		opencv-python
	@echo "Core dependencies installed successfully!"
	@echo ""

# ---- Install Optional ReID Dependencies (CLIP) ----
.PHONY: install-reid
install-reid:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Installing optional ReID (CLIP) dependencies..."
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install open_clip_torch Pillow
	@echo "Optional ReID dependencies installed!"
	@echo ""

# ---- Run Tracking ----
.PHONY: track
track:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Running human tracking..."
	@$(PYTHON) $(TRACKING_SCRIPT)

# ---- Playback Visualization ----
.PHONY: playback
playback:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Starting tracking playback..."
	@$(PYTHON) $(PLAYBACK_SCRIPT)

# ---- Trajectory & Speed Plots ----
.PHONY: visualize
visualize:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Visualizing tracking results..."
	@$(PYTHON) $(VISUALIZATION_SCRIPT)

# ---- Evaluation Metrics (Multi-Human MOT) ----
.PHONY: evaluate
evaluate:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Evaluating multi-human tracking performance..."
	@$(PYTHON) $(EVALUATION_SCRIPT)

# ---- Clean Cache Files ----
.PHONY: clean
clean:
	@echo "Cleaning generated cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Clean complete!"

# ---- Remove Virtual Environment ----
.PHONY: clean-venv
clean-venv:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed!"

# ---- Full Cleanup ----
.PHONY: clean-all
clean-all: clean clean-venv
	@echo "Full cleanup complete!"
