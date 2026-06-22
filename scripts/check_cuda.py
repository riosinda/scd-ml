"""Quick sanity check for CUDA availability and GPU info.

Usage (from repo root, with .venv-mask active):
    python scripts/check_cuda.py
"""
import sys
import torch


def main():
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA device found — segmentation will run on CPU (slow).")
        return 1

    n = torch.cuda.device_count()
    print(f"GPU count: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  [{i}] {props.name}  —  {mem_gb:.1f} GB  (compute {props.major}.{props.minor})")

    # Quick tensor round-trip to confirm the device actually works
    print("\nRunning tensor round-trip on GPU 0 …", end=" ")
    x = torch.ones(1024, 1024, device="cuda:0")
    result = (x * 2).sum().item()
    assert result == 1024 * 1024 * 2, "Unexpected result"
    print(f"OK  (sum={result:,.0f})")

    print("\nCUDA is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
