from __future__ import annotations

import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
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

ext_modules = [
	Pybind11Extension(
		"parallelnet_cpp",
		sources=[
			"bridge/bindings.cpp",
			"engine/dense_layer.cpp",
			"engine/network.cpp",
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
	cmdclass={"build_ext": build_ext},
)
