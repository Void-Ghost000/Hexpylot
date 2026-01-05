# HexPylot

A lock-free, zero-copy parallel engine for Python.  
Pushing the CPU to the hardware memory wall.

Best suited for compute-bound workloads.  
When the workload is memory-bandwidth-bound, this project honestly exposes the hardware limit rather than pretending to achieve linear scaling.

HexPylot is a minimal parallel engine built around removing software overhead
before touching hardware limits.

```text

## Architecture Overview


┌──────────────────────────────┐
│        User Workload          │
│  (benchmark.py / custom code)│
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   Lock-free Control Layer    │
│  - phase-based execution     │
│  - barrier synchronization  │
│  - no global locks           │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   Shared Memory Data Plane   │
│  - zero-copy ndarray views  │
│  - no pickle / no IPC copy  │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   JIT Compute Kernels        │
│  - numba @njit               │
│  - GIL-free numeric loops   │
└──────────────────────────────┘

## What this repo contains

- `benchmark.py`: one-command benchmark runner
- `results/`: captured benchmark results (optional but recommended)

## Quick start

```bash


python benchmark.py
