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

## Server demo (FastAPI)

Start the training server directly:

```
python python/start_server.py
```

Or run the end-to-end demo (starts the server in the background and prints progress):

```
python python/demo.py
```

Note: `demo.py` starts its own server. Do not run `start_server.py` at the same time.

### Server performance knobs

Enable micro-batching in the training worker to keep the GPU busier during bursty traffic:

```
MICRO_BATCH_SIZE=8 python python/start_server.py
```

PowerShell:

```
$env:MICRO_BATCH_SIZE=8; python python/start_server.py
```

### Index-based training payloads

The server now preloads CIFAR-10 and `/train` accepts `indices` to avoid sending full tensors.
This cuts JSON overhead and improves throughput. The scripts `demo.py` and `send_epochs.py`
already use indices.

Example output (numbers will vary):

```
Starting ParallelNet training server in background...
Server ready at http://localhost:8000

Dataset: 50,000 training images, 10,000 test images
Model: Dense(3072->512)->ReLU->Dense(512->10) -- 1,594,890 parameters
GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM)

Sending training data...
Step   10 | Loss: 2.287 | Acc:  9.8% | Queue: 3
Step   20 | Loss: 2.156 | Acc: 13.2% | Queue: 2
Step   30 | Loss: 1.982 | Acc: 18.7% | Queue: 4
Step   40 | Loss: 1.847 | Acc: 22.1% | Queue: 3
Step   50 | Loss: 1.734 | Acc: 26.4% | Queue: 2

Sample predictions (5 test images):
	Image 1: actual=cat       predicted=cat        OK
	Image 2: actual=ship      predicted=automobile X
	Image 3: actual=airplane  predicted=airplane   OK
	Image 4: actual=frog      predicted=frog       OK
	Image 5: actual=horse     predicted=deer       X

Continuing training...
Step  100 | Loss: 1.623 | Acc: 29.8% | Queue: 2
Step  150 | Loss: 1.534 | Acc: 32.1% | Queue: 3
Step  200 | Loss: 1.478 | Acc: 34.6% | Queue: 1
Step  250 | Loss: 1.421 | Acc: 36.8% | Queue: 2
Step  300 | Loss: 1.389 | Acc: 38.2% | Queue: 3
Step  350 | Loss: 1.352 | Acc: 39.7% | Queue: 2
Step  400 | Loss: 1.318 | Acc: 41.3% | Queue: 0

After 1 epoch (51,200 samples):
	Loss:     1.318
	Accuracy: 41.3%
	Expected: ~35-45% (MLP on CIFAR-10, 1 epoch) OK

Server still running at http://localhost:8000/docs
Press Ctrl+C to stop. Run send_epochs.py --epochs 10 for full training.
```

To continue training while the server is running:

```
python python/send_epochs.py --epochs 10 --batch-size 256
```

### Concurrent load generator

To keep the GPU busy with multiple clients and periodic prediction checks:

```
python python/start_server.py
python python/multi_client_load.py --workers 6 --batch-size 256 --run-seconds 180 \
	--accuracy-interval 15 --predict-batch 128
```

## GPU utilization check

Open a second terminal while demo.py is running:

```
nvidia-smi dmon -s u -d 1
```

Or a single snapshot:

```
nvidia-smi
```

Look for GPU-Util above 50% and Memory-Used above ~500MB. If GPU-Util is near 0%,
increase batch size to 256 or 512 in send_epochs.py.

If the queue depth stays at 0, increase load with `multi_client_load.py` or raise
`MICRO_BATCH_SIZE`.

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
