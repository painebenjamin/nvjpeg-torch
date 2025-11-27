# nvjpeg-torch

GPU-accelerated JPEG encoding and decoding using NVIDIA's nvJPEG library with native PyTorch tensor support.

## Features

- **GPU Acceleration**: Hardware-accelerated JPEG encoding/decoding using nvJPEG
- **PyTorch Integration**: Direct support for PyTorch tensors in CHW format
- **Flexible Input**: Accepts NumPy arrays (HWC, BGR) and PyTorch tensors (CHW, RGB)
- **Auto Range Detection**: Automatically handles float tensors in `[0, 1]` or `[-1, 1]` ranges
- **Multiple dtypes**: Supports `uint8` and all floating point types (`float16`, `float32`, `float64`, `bfloat16`, `float8`, etc.)
- **Import Order Safe**: Can be imported before or after `torch` - dependencies are handled automatically

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit >= 10.2 (CUDA 12.x recommended)
- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.17

## Supported Platforms

- Linux (x86_64)
- Windows (x86_64)

## Installation

### From PyPI

```shell
pip install nvjpeg-torch
```

### From Source

```shell
git clone https://github.com/UsingNet/nvjpeg-python.git
cd nvjpeg-python
pip install .
```

### Building Wheels with Bundled Libraries

To create a wheel that includes CUDA libraries (for distribution):

```shell
# Install build dependencies
pip install build numpy torch

# Build wheel with bundled CUDA libraries
make bundle
```

This creates a wheel in `dist/` that includes the necessary CUDA runtime libraries.

## Usage

### Initialize NvJpeg

```python
from nvjpeg import NvJpeg

nj = NvJpeg()
```

### Read JPEG File to NumPy Array

```python
img = nj.read("image.jpg")
# Returns BGR numpy array (HWC format), similar to cv2.imread()
```

### Write to JPEG File

Works with both NumPy arrays and PyTorch tensors:

```python
# NumPy array (HWC format)
nj.write("output.jpg", numpy_img)

# PyTorch tensor (CHW format)
nj.write("output.jpg", torch_tensor)

# With quality parameter (0-100, default 70)
nj.write("output.jpg", img, 90)
```

### Decode JPEG Bytes to NumPy Array

```python
with open("image.jpg", "rb") as f:
    jpeg_bytes = f.read()

img = nj.decode(jpeg_bytes)
# Returns HWC numpy array, similar to cv2.imdecode()
```

### Encode to JPEG Bytes

Accepts both NumPy arrays and PyTorch tensors:

```python
# NumPy array (HWC format)
jpeg_bytes = nj.encode(numpy_img)

# PyTorch tensor (CHW format)
jpeg_bytes = nj.encode(torch_tensor)

# With quality parameter
jpeg_bytes = nj.encode(img, 90)
```

### PyTorch Tensor Support

The encoder automatically handles PyTorch tensors with various formats.

**Note**: PyTorch tensors are expected in **RGB** format (standard for torchvision), while NumPy arrays are expected in **BGR** format (standard for OpenCV). The conversion is handled automatically:

```python
import torch
from nvjpeg import NvJpeg

nj = NvJpeg()

# uint8 tensor in CHW format
tensor_uint8 = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
jpeg_bytes = nj.encode(tensor_uint8)

# float tensor in [0, 1] range (e.g., from torchvision transforms)
tensor_01 = torch.rand(3, 480, 640, dtype=torch.float32)
jpeg_bytes = nj.encode(tensor_01)

# float tensor in [-1, 1] range (e.g., normalized for neural networks)
tensor_neg11 = torch.rand(3, 480, 640) * 2 - 1
jpeg_bytes = nj.encode(tensor_neg11)

# 4D tensor with batch size 1 (NCHW)
tensor_batch = torch.rand(1, 3, 480, 640)
jpeg_bytes = nj.encode(tensor_batch)

# GPU tensors are automatically moved to CPU for encoding
tensor_gpu = torch.rand(3, 480, 640, device='cuda')
jpeg_bytes = nj.encode(tensor_gpu)
```

### Integration with torchvision

```python
import torch
from torchvision import transforms
from PIL import Image
from nvjpeg import NvJpeg

nj = NvJpeg()

# Load and process image with torchvision
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to CHW float [0, 1]
])

img = Image.open("input.jpg")
tensor = transform(img)

# Encode directly with nvjpeg-torch
jpeg_bytes = nj.encode(tensor, 95)
```

## Version String

The package version includes CUDA and PyTorch version information:

```
0.1.0+cu124torch2.5
```

This indicates:
- Base version: `0.1.0`
- CUDA version: `12.4`
- PyTorch version: `2.5`

## Performance

nvJPEG provides significant speedups over CPU-based JPEG libraries, especially for:
- Batch processing of images
- High-resolution images
- Real-time video/image pipelines

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/UsingNet/nvjpeg-python)
- [Issue Tracker](https://github.com/UsingNet/nvjpeg-python/issues)
- [NVIDIA nvJPEG Documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html)
