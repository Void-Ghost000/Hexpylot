# HexPylot

A lock-free, zero-copy parallel engine for Python.  
Pushing the CPU to the hardware memory wall.

Best suited for compute-bound workloads.  
When the workload is memory-bandwidth-bound, this project honestly exposes the hardware limit rather than pretending to achieve linear scaling.

---

## What is HexPylot?

HexPylot is a **minimal parallel execution engine** built to explore how far Python can be pushed **after removing most software-level overhead**.

Instead of focusing on features, abstractions, or APIs, this project focuses on:

- eliminating unnecessary locks
- avoiding IPC and serialization
- using shared memory directly
- letting hardware limits surface naturally

This is primarily an **educational and exploratory project**, not a production framework.

---

## Architecture Overview

HexPylot uses a deliberately simple, layered architecture.

### 1. User Workload
- benchmark.py
- custom plugins (e.g. Monte Carlo simulation)

This layer defines *what* computation is performed.

---

### 2. Lock-free Control Layer
- phase-based execution
- barrier synchronization
- no fine-grained global locks

This layer coordinates parallel execution while minimizing synchronization overhead.

---

### 3. Shared Memory Data Plane
- zero-copy ndarray views
- no pickle
- no IPC memory copy

All worker processes operate directly on shared memory buffers.

---

### 4. JIT Compute Kernels
- Numba @njit
- GIL-free numeric loops

This layer executes the actual computation at near-native speed.

---

The goal of this design is simple:

Remove software bottlenecks first,  
then observe where the hardware becomes the limiting factor.

---

## Example Plugin: Monte Carlo Engine (Quantitative Finance)

HexPylot is not only a benchmark project.

It also serves as a foundation for **real, compute-bound workloads**.

As a concrete example, this repository includes a Monte Carlo simulation engine,
a common workload in quantitative finance and risk modeling.

This example demonstrates:

- how compute-heavy workloads scale under HexPylot
- how shared memory avoids IPC overhead
- where memory bandwidth becomes the dominant bottleneck

The intent is not to provide a full finance library,
but to showcase a realistic, non-trivial workload running on the engine.
