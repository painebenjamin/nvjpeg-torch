"""
nvjpeg-torch: GPU-accelerated JPEG encoding/decoding with PyTorch support.

This module provides hardware-accelerated JPEG operations using NVIDIA's nvJPEG library,
with native support for PyTorch tensors.
"""

# Import torch first to ensure its libraries are loaded before our C extension
# This is necessary because our extension links against libc10.so and other torch libraries
import torch  # noqa: F401

# Now import the C extension
from _nvjpeg import NvJpeg

__all__ = ["NvJpeg"]

