# Hexpylot
A lock-free, zero-copy parallel engine for Python. Pushing CPU to the hardware memory wall.
# HexPylot

A lock-free, zero-copy parallel engine for Python.  
Pushing the CPU to the hardware memory wall.

Best suited for compute-bound workloads.  
When the workload is memory-bandwidth-bound, this project honestly exposes the hardware limit rather than pretending to achieve linear scaling.

## What this repo contains

- `benchmark.py`: one-command benchmark runner
- `results/`: captured benchmark results (optional but recommended)

## Quick start

```bash
python benchmark.py
