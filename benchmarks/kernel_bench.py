def configure_windows_cuda_dlls():
	import os
	import sys
	from pathlib import Path

	if sys.platform != "win32":
		return

	# Allow manual override for custom CUDA/CuPy DLL locations.
	manual_paths = os.environ.get("CUPY_DLL_PATHS", "")
	for entry in manual_paths.split(os.pathsep):
		entry = entry.strip()
		if entry:
			try:
				os.add_dll_directory(entry)
			except (FileNotFoundError, OSError):
				pass

	# Prefer bundled NVIDIA Python packages if present.
	venv_root = Path(sys.prefix)
	nvidia_bins = [
		venv_root / "Lib" / "site-packages" / "nvidia" / "curand" / "bin",
		venv_root / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
	]
	for path in nvidia_bins:
		if path.exists():
			try:
				os.add_dll_directory(str(path))
			except (FileNotFoundError, OSError):
				pass

	# Fall back to default CUDA install path if present.
	default_cuda = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin")
	if default_cuda.exists():
		try:
			os.add_dll_directory(str(default_cuda))
		except (FileNotFoundError, OSError):
			pass


configure_windows_cuda_dlls()

import argparse
import json
import math
import os
import sys
import subprocess
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

def load_matmul_module():
	repo_root = Path(__file__).resolve().parents[1]
	kernels_dir = repo_root / "kernels"
	src_path = kernels_dir / "matmul.cu"
	source = src_path.read_text(encoding="utf-8")
	return cp.RawModule(
		code=source,
		options=("--std=c++17", f"-I{kernels_dir}"),
		name_expressions=("naive_matmul_kernel", "tiled_matmul_kernel"),
	)


def gflops_for_gemm(m, n, k, seconds):
	ops = 2.0 * m * n * k
	return ops / (seconds * 1e9)


def benchmark_kernel(kernel, a, b, c, m, n, k, block, grid, warmup=2, runs=100):
	for _ in range(warmup):
		kernel(grid, block, (a, b, c, m, n, k))
	cp.cuda.Stream.null.synchronize()

	timings = []
	for _ in range(runs):
		start = cp.cuda.Event()
		end = cp.cuda.Event()
		start.record()
		kernel(grid, block, (a, b, c, m, n, k))
		end.record()
		end.synchronize()
		elapsed_ms = cp.cuda.get_elapsed_time(start, end)
		timings.append(elapsed_ms / 1000.0)
	return float(cp.median(cp.asarray(timings)))


def benchmark_cublas(a, b, warmup=2, runs=100):
	for _ in range(warmup):
		_ = a @ b
	cp.cuda.Stream.null.synchronize()

	timings = []
	for _ in range(runs):
		start = cp.cuda.Event()
		end = cp.cuda.Event()
		start.record()
		_ = a @ b
		end.record()
		end.synchronize()
		elapsed_ms = cp.cuda.get_elapsed_time(start, end)
		timings.append(elapsed_ms / 1000.0)
	return float(cp.median(cp.asarray(timings)))


def make_random_matrix(shape):
	try:
		return cp.random.random(shape, dtype=cp.float32)
	except ImportError:
		# Fall back to NumPy when cuRAND DLLs are unavailable.
		host = np.random.random_sample(shape).astype(np.float32)
		return cp.asarray(host)


def load_results(path):
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def save_results(path, payload):
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2)


def get_default_label():
	try:
		output = subprocess.check_output(
			["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
			stderr=subprocess.STDOUT,
			text=True,
			timeout=5,
		).strip()
		return output.splitlines()[0].strip() if output else "current"
	except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
		return "current"


def get_runtime_metadata():
	def format_cuda_version(version):
		major = version // 1000
		minor = (version % 1000) // 10
		return f"{major}.{minor}"

	def get_nvrtc_version():
		try:
			import cupy.cuda.nvrtc as nvrtc
			major, minor = nvrtc.getVersion()
			return f"{major}.{minor}"
		except Exception:
			return "unknown"

	device = cp.cuda.Device()
	props = cp.cuda.runtime.getDeviceProperties(device.id)
	device_name = props.get("name", "unknown")
	if isinstance(device_name, (bytes, bytearray)):
		device_name = device_name.decode("utf-8", errors="replace")
	return {
		"device_id": device.id,
		"device_name": device_name,
		"compute_capability": f"{props.get('major', 0)}.{props.get('minor', 0)}",
		"total_memory_bytes": int(props.get("totalGlobalMem", 0)),
		"cupy_version": cp.__version__,
		"nvrtc_version": get_nvrtc_version(),
		"runtime_version": format_cuda_version(cp.cuda.runtime.runtimeGetVersion()),
		"driver_version": format_cuda_version(cp.cuda.runtime.driverGetVersion()),
	}


def warmup_device(iterations=5):
	# Reduce first-run compilation/initialization noise.
	for _ in range(iterations):
		a = cp.random.random((64, 64), dtype=cp.float32)
		b = cp.random.random((64, 64), dtype=cp.float32)
		_ = a @ b
	cp.cuda.Stream.null.synchronize()


def main():
	configure_windows_cuda_dlls()
	parser = argparse.ArgumentParser(description="Benchmark tiled GEMM vs cuBLAS.")
	parser.add_argument("--label", default=get_default_label(), help="Label for this GPU run.")
	parser.add_argument("--overlay", action="append", default=[], help="Path to overlay JSON results.")
	parser.add_argument("--out", default="benchmarks/matmul_vs_cublas.png", help="Output plot path.")
	parser.add_argument("--save", default="benchmarks/matmul_results.json", help="Output JSON path.")
	parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
	parser.add_argument("--warmup-iters", type=int, default=5, help="GPU warmup iterations before timing.")
	args = parser.parse_args()

	cp.random.seed(args.seed)
	np.random.seed(args.seed)
	if args.warmup_iters > 0:
		warmup_device(args.warmup_iters)

	sizes = [64, 128, 256, 512, 1024, 2048, 4096]
	module = load_matmul_module()
	naive_kernel = module.get_function("naive_matmul_kernel")
	tiled_kernel = module.get_function("tiled_matmul_kernel")

	naive_gflops = []
	tiled_gflops = []
	cublas_gflops = []

	print("Benchmarking CUDA matmul kernels (float32)\n")

	for size in sizes:
		m = n = k = size
		a = make_random_matrix((m, k))
		b = make_random_matrix((k, n))
		c = cp.empty((m, n), dtype=cp.float32)

		block = (16, 16, 1)
		grid = (math.ceil(n / 16), math.ceil(m / 16), 1)

		naive_time = benchmark_kernel(naive_kernel, a, b, c, m, n, k, block, grid)
		tiled_time = benchmark_kernel(tiled_kernel, a, b, c, m, n, k, block, grid)
		cublas_time = benchmark_cublas(a, b)

		naive_perf = gflops_for_gemm(m, n, k, naive_time)
		tiled_perf = gflops_for_gemm(m, n, k, tiled_time)
		cublas_perf = gflops_for_gemm(m, n, k, cublas_time)

		naive_gflops.append(naive_perf)
		tiled_gflops.append(tiled_perf)
		cublas_gflops.append(cublas_perf)

		percent = (tiled_perf / cublas_perf) * 100.0 if cublas_perf > 0 else 0.0
		print(f"Size {size:4d}: tiled_matmul is {percent:6.2f}% of cuBLAS")

	results = {
		"label": args.label,
		"sizes": sizes,
		"naive_gflops": naive_gflops,
		"tiled_gflops": tiled_gflops,
		"cublas_gflops": cublas_gflops,
		"seed": args.seed,
		"warmup_iters": args.warmup_iters,
		"metadata": get_runtime_metadata(),
	}
	save_results(args.save, results)

	overlays = [load_results(path) for path in args.overlay]

	plt.figure(figsize=(9, 5.5))
	plt.plot(sizes, tiled_gflops, marker="o", label=f"{args.label} tiled")
	plt.plot(sizes, cublas_gflops, marker="o", label=f"{args.label} cuBLAS")

	for overlay in overlays:
		label = overlay.get("label", "overlay")
		plt.plot(overlay["sizes"], overlay["tiled_gflops"], marker="o", label=f"{label} tiled")
		plt.plot(overlay["sizes"], overlay["cublas_gflops"], marker="o", label=f"{label} cuBLAS")

	plt.axhline(4400.0, color="tab:red", linestyle="--", linewidth=1.2, label="GTX 1650 peak")
	plt.title("GEMM Performance (GFLOPs)")
	plt.xlabel("Matrix size (N = M = K)")
	plt.ylabel("GFLOPs")
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.legend()
	plt.tight_layout()

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
	main()
