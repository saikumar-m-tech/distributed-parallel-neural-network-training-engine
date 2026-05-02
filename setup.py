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

extra_compile_args = []
if os.name == "nt":
	extra_compile_args = ["/std:c++17"]
else:
	extra_compile_args = ["-std=c++17"]

class build_ext_cuda(build_ext_orig):
	def build_extensions(self):
		if os.name == "nt":
			self._build_cuda_objects()
		super().build_extensions()

	def _build_cuda_objects(self):
		nvcc = Path(cuda_path) / "bin" / "nvcc.exe"
		if not nvcc.exists():
			raise RuntimeError(f"nvcc not found at {nvcc}")

		for ext in self.extensions:
			cuda_sources = [src for src in ext.sources if src.endswith(".cu")]
			if not cuda_sources:
				continue

			ext.sources = [src for src in ext.sources if not src.endswith(".cu")]
			build_temp = Path(self.build_temp)
			build_temp.mkdir(parents=True, exist_ok=True)

			include_args = [f"-I{inc}" for inc in ext.include_dirs]
			define_args = [f"-D{name}={value}" if value is not None else f"-D{name}"
							for name, value in ext.define_macros]
			cuda_objects = []

			for src in cuda_sources:
				obj_name = Path(src).with_suffix(".obj").name
				obj_path = build_temp / obj_name
				cmd = [
					str(nvcc),
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
				subprocess.check_call(cmd)
				cuda_objects.append(str(obj_path))

			ext.extra_objects = (ext.extra_objects or []) + cuda_objects


ext_modules = [
	Pybind11Extension(
		"parallelnet_cpp",
		sources=[
			"bridge/bindings.cpp",
			"engine/dense_layer.cpp",
			"engine/network.cpp",
			"kernels/matmul.cu",
			"kernels/activations.cu",
		],
		include_dirs=[
			str(ROOT / "engine"),
			str(ROOT / "kernels"),
			str(ROOT / "mpi"),
			str(ROOT / "bridge"),
			cuda_include,
		],
		library_dirs=[cuda_lib],
		libraries=["cudart"],
		define_macros=[("PARALLELNET_NO_MPI", "1")],
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
