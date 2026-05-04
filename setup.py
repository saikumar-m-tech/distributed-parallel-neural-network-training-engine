from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools import setup


ROOT = Path(__file__).parent

cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
if not cuda_path:
	raise RuntimeError(
		"CUDA_PATH/CUDA_HOME is not set. Please set it to your CUDA toolkit directory."
	)

cuda_include = str(Path(cuda_path) / "include")
cuda_lib = str(Path(cuda_path) / "lib" / "x64")

windows_sdk_include_dirs = []
if os.name == "nt":
	kits_root = Path(r"C:\Program Files (x86)\Windows Kits\10\Include")
	if kits_root.exists():
		versions = sorted([p for p in kits_root.iterdir() if p.is_dir()])
		if versions:
			sdk = versions[-1]
			windows_sdk_include_dirs = [
				str(sdk / "ucrt"),
				str(sdk / "shared"),
				str(sdk / "um"),
				str(sdk / "winrt"),
			]

extra_compile_args = []
if os.name == "nt":
    extra_compile_args = ["/std:c++17"]
else:
    extra_compile_args = ["-std=c++17"]


def _parse_mpi_flags(flags: list[str]) -> tuple[list[str], list[str], list[str]]:
	include_dirs: list[str] = []
	library_dirs: list[str] = []
	libraries: list[str] = []
	for flag in flags:
		if flag.startswith("-I"):
			include_dirs.append(flag[2:])
		elif flag.startswith("-L"):
			library_dirs.append(flag[2:])
		elif flag.startswith("-l"):
			libraries.append(flag[2:])
	return include_dirs, library_dirs, libraries


def _get_mpi_build_info() -> tuple[list[str], list[str], list[str]]:
	try:
		mpi_compile = subprocess.check_output(["mpicc", "--showme:compile"]).decode().split()
		mpi_link = subprocess.check_output(["mpicc", "--showme:link"]).decode().split()
		inc_dirs, lib_dirs, libs = _parse_mpi_flags(mpi_compile + mpi_link)
		return inc_dirs, lib_dirs, libs
	except (OSError, subprocess.CalledProcessError):
		return [], [], []

class build_ext_cuda(build_ext_orig):
	def build_extensions(self):
		if os.name == "nt":
			self._build_cuda_objects()
		super().build_extensions()

	def _build_cuda_objects(self):
		nvcc = Path(cuda_path) / "bin" / "nvcc.exe"
		if not nvcc.exists():
			raise RuntimeError(f"nvcc not found at {nvcc}")

		build_temp = Path(self.build_temp)
		build_temp.mkdir(parents=True, exist_ok=True)
		nvcc_env = os.environ.copy()
		nvcc_env.setdefault("TMP", str(build_temp))
		nvcc_env.setdefault("TEMP", str(build_temp))
		nvcc_env.setdefault("CUDA_CACHE_PATH", str(build_temp / "cuda_cache"))

		for ext in self.extensions:
			cuda_sources = [src for src in ext.sources if src.endswith(".cu")]
			if not cuda_sources:
				continue

			ext.sources = [src for src in ext.sources if not src.endswith(".cu")]

			include_args = [f"-I{inc}" for inc in ext.include_dirs]
			if os.name == "nt":
				sdk_root = r"C:\Program Files (x86)\Windows Kits\10\Include"
				if os.path.isdir(sdk_root):
					versions = sorted(
						d for d in os.listdir(sdk_root)
						if os.path.isdir(os.path.join(sdk_root, d))
					)
					if versions:
						sdk = os.path.join(sdk_root, versions[-1])
						include_args += [
							f"-I{os.path.join(sdk, 'ucrt')}",
							f"-I{os.path.join(sdk, 'shared')}",
							f"-I{os.path.join(sdk, 'um')}",
							f"-I{os.path.join(sdk, 'winrt')}",
						]

			define_args = [f"-D{name}={value}" if value is not None else f"-D{name}"
						   for name, value in ext.define_macros]
			cuda_objects = []

			for src in cuda_sources:
				obj_name = Path(src).with_suffix(".obj").name
				obj_path = build_temp / obj_name
				cmd = [
					str(nvcc),
					"-allow-unsupported-compiler",
					"-c",
					"-std=c++17",
					"-Xcompiler",
					"/MD",
					"-Xcompiler",
					"/EHsc",
					"-o",
					str(obj_path),
					str(Path(src)),
				] + include_args + define_args
				subprocess.check_call(cmd, env=nvcc_env)
				cuda_objects.append(str(obj_path))

			ext.extra_objects = (ext.extra_objects or []) + cuda_objects


mpi_include_dirs, mpi_library_dirs, mpi_libraries = _get_mpi_build_info()
if os.name == "nt":
	msmpi_inc = os.environ.get("MSMPI_INC", "C:/Program Files/Microsoft MPI/Inc")
	if msmpi_inc not in mpi_include_dirs:
		mpi_include_dirs.append(msmpi_inc)
	msmpi_lib = os.environ.get("MSMPI_LIB64", "C:/Program Files/Microsoft MPI/Lib/x64")
	if msmpi_lib not in mpi_library_dirs:
		mpi_library_dirs.append(msmpi_lib)
	if "msmpi" not in mpi_libraries:
		mpi_libraries.append("msmpi")

ext_modules = [
	Pybind11Extension(
		"parallelnet_cpp",
		sources=[
			"bridge/bindings.cpp",
			"engine/dense_layer.cpp",
			"engine/batch_norm_layer.cpp",
			"engine/network.cpp",
			"mpi/gradient_sync.cpp",
			"kernels/matmul.cu",
			"kernels/activations.cu",
			"kernels/batch_norm.cu",
			"kernels/sgd_update.cu",
		],
		include_dirs=[
    str(ROOT / "engine"),
    str(ROOT / "kernels"),
    str(ROOT / "mpi"),
    str(ROOT / "bridge"),
    cuda_include,
    *windows_sdk_include_dirs,
    *mpi_include_dirs,
],
		library_dirs=[cuda_lib, *mpi_library_dirs],
		libraries=["cudart", *mpi_libraries],
		define_macros=[],
		extra_compile_args=extra_compile_args,
	),
]

setup(
	name="parallelnet_cpp",
	version="0.0.1",
	packages=[],
	py_modules=[],
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext_cuda},
)
