# nvjpeg-torch Makefile
# Build Python bindings for NVIDIA nvJPEG library with PyTorch support

PYTHON ?= python3
CUDA_PATH ?= /usr/local/cuda

.PHONY: all build wheel bundle clean test help

all: wheel

help:
	@echo "nvjpeg-torch build targets:"
	@echo "  make build   - Build the extension in-place"
	@echo "  make wheel   - Build a wheel package"
	@echo "  make bundle  - Build wheel with bundled CUDA libraries"
	@echo "  make test    - Run tests"
	@echo "  make clean   - Clean build artifacts"

build:
	$(PYTHON) setup.py build_ext --inplace

wheel:
	$(PYTHON) -m build --wheel $(WHEEL_BUILD_ARGS)

bundle: wheel
	@echo "Bundling CUDA libraries into wheel..."
	@for whl in dist/*.whl; do \
		$(PYTHON) bundle_libs.py "$$whl" -o dist/; \
	done
	@echo "Done! Bundled wheel is in dist/"

test: build
	$(PYTHON) tests/test.py
	$(PYTHON) tests/test-with-multiprocessing.py

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/
	rm -f *.so nvjpeg*.so
	rm -rf nvjpeg_libs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Development install
dev-install: build
	$(PYTHON) -m pip install -e .

# Install patchelf if needed for bundling
install-patchelf:
	@which patchelf > /dev/null 2>&1 || (echo "Installing patchelf..." && sudo apt-get install -y patchelf)
