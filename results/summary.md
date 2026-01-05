# Benchmark Summary

Environment:
- Python: 3.12 (or later)
- Engine: shared_memory + barrier + parity toggle
- Kernels: Numba (njit, nogil)

## Memory-bound (heavy=0)
- Real diffusion-like stencil workload
- Typically saturates memory bandwidth quickly
- Scaling beyond a few cores is limited by hardware

## Compute-bound (heavy=20)
- Artificial heavy loop to validate parallel scaling
- Demonstrates the engine can scale when compute dominates

## How to reproduce
```bash
python benchmark.py
