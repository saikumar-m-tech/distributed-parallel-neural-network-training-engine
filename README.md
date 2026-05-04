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

## Workflow and project stage

Current workflow:

1. Build the bridge (`build.bat` on Windows or `pip install -e .`).
2. Run the kernel and bridge tests.
3. Train with CSV logging enabled for experiments:

```
python python/train.py --epochs 3 --limit-samples 5000 --batch-size 128 --log-csv plots/run.csv
```

4. Use `python/plot_results.py` to generate convergence and scaling charts.
5. Benchmark Ring-AllReduce with `mpiexec -n 4 build/Debug/ring_allreduce.exe`.

Current stage:

- CUDA kernels include matmul, activations, SGD update, and batch norm (forward/backward).
- The engine uses Dense -> BatchNorm -> Dense with GPU-side SGD updates.
- MPI gradient sync is supported (host-side allreduce), with timing instrumentation for compute vs sync.
- Training supports per-epoch CSV logging (epoch, loss, accuracy, time_per_epoch, throughput, compute_ms, sync_ms).
- A step decay learning rate schedule is applied (halves every 20 epochs).
- Ring-AllReduce benchmark is implemented for dissertation comparison with MPI_Allreduce.

## Notes

- CIFAR-10 is downloaded to the `data/` directory on first run.
- Training logs include per-epoch time and throughput.
- The `GpuBuffer` destructor is silent unless a cudaFree error occurs.
