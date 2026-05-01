# Distributed Parallel Neural Network Training Engine

C++/CUDA kernels, a simple MLP engine, MPI utilities, and a PyBind11 bridge for training on CIFAR-10.

## Dependencies

- Python 3.10+
- CUDA Toolkit 12.x (GTX 1650 supported)
- PyBind11
- NumPy
- TorchVision (for CIFAR-10 download)
- mpi4py (for MPI training)
- MPI runtime (MS-MPI on Windows)

Python packages:

```
pip install numpy torchvision pybind11 mpi4py pytest
```

## Build the C++ bridge

Set CUDA path (Windows examples):

PowerShell:
```
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
```

Git Bash:
```
export CUDA_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
```

Build/install:

```
pip install -e .
```

## Run tests

CUDA kernel tests:
```
pytest tests/test_kernels.py
```

PyBind bridge tests:
```
pytest tests/test_bridge.py
```

MPI tests:
```
mpiexec -n 2 ./test_mpi
```

## Train on CIFAR-10

Single rank:
```
mpirun -n 1 python python/train.py
```

Two ranks:
```
mpirun -n 2 python python/train.py
```

Useful speed flags:
```
mpirun -n 1 python python/train.py --batch-size 512 --limit-samples 20000 --epochs 5
```

## Notes

- CIFAR-10 is downloaded to the `data/` directory on first run.
- Training logs include per-epoch time and throughput.
- The `GpuBuffer` destructor prints are throttled to reduce log spam.
