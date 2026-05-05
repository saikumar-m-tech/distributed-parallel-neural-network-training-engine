from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def _ensure_cuda_path() -> None:
    if os.name != "nt":
        return
    if "CUDA_PATH" not in os.environ and "CUDA_HOME" not in os.environ:
        return
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not cuda_root:
        return
    cuda_bin = os.path.join(cuda_root, "bin")
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)


def _print_banner() -> None:
    print("ParallelNet Training Server", flush=True)
    print("GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM)", flush=True)
    print("Endpoint: http://localhost:8000", flush=True)
    print("Docs:     http://localhost:8000/docs", flush=True)


def main() -> None:
    _ensure_cuda_path()
    os.chdir(Path(__file__).resolve().parent)
    _print_banner()
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
