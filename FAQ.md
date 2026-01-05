# FAQ

This document answers common questions about HexPylot.
The goal is to be transparent about both its strengths and its limitations.

HexPylot is a personal learning and exploration project.
It is not a production-ready framework and is not presented as one.

---

## 1. What problem does HexPylot try to solve?

HexPylot explores how far Python can be pushed **after removing most software-level overhead** in parallel execution.

Instead of focusing on application-level features, this project focuses on:
- lock-free coordination
- zero-copy shared memory
- minimizing Python interpreter overhead before hitting hardware limits

It is primarily an educational and experimental project.

---

## 2. What kind of workloads is HexPylot designed for?

HexPylot works best with **compute-bound workloads**, where:
- each data element involves significant computation
- arithmetic dominates over memory access

Examples include:
- numerical simulations
- Monte Carlo–style computations
- heavy mathematical kernels

---

## 3. What kind of workloads is HexPylot *not* designed for?

HexPylot is **not suitable for memory-bandwidth-bound workloads**, such as:
- simple stencil updates
- light array operations
- workloads dominated by memory reads/writes

In these cases, adding more CPU cores will not result in meaningful speedup.

---

## 4. Why doesn’t memory-bound workload scale with more CPU cores?

Because the hardware memory bandwidth becomes saturated.

Once the memory subsystem is fully utilized:
- additional CPU cores cannot fetch data any faster
- software-level optimizations no longer help

HexPylot intentionally exposes this behavior instead of masking it.

---

## 5. Is this a limitation of HexPylot?

No — this is a **hardware limitation**, not a software bug.

HexPylot removes many common software bottlenecks, which makes the hardware limit visible very quickly.
This is expected behavior on modern CPUs.

---

## 6. Why does compute-bound workload scale much better?

Compute-bound workloads:
- spend more time executing arithmetic instructions
- rely less on memory bandwidth

This allows multiple CPU cores to work in parallel more effectively, resulting in near-linear speedup until other limits are reached.

---

## 7. Is this just Numba being fast?

Numba is an important part of the system, but it is **not the whole story**.

HexPylot also relies on:
- shared memory to avoid IPC copies
- phase-based execution instead of fine-grained locking
- explicit synchronization control

Without these, Numba alone would not achieve the same scaling behavior.

---

## 8. How is this different from standard multiprocessing?

Standard multiprocessing typically involves:
- pickling data
- copying data between processes
- higher synchronization overhead

HexPylot instead:
- shares memory directly between processes
- avoids serialization entirely
- minimizes synchronization to coarse-grained barriers

---

## 9. Does HexPylot bypass the Python GIL?

HexPylot does not “remove” the GIL globally.

However:
- numerical kernels are executed inside Numba-compiled functions
- these kernels run without the GIL
- coordination happens outside hot loops

This allows effective parallelism for numeric workloads.

---

## 10. Why use a hexagonal grid in the benchmark?

Hexagonal grids:
- require more neighbor accesses than square grids
- have less regular memory access patterns

They are intentionally used as a **stress test**, not because they are faster.
If the architecture performs well here, it should perform at least as well on simpler layouts.

---

## 11. Isn’t hex topology more expensive than grid topology?

Yes — and that is the point.

Hex grids are computationally more expensive.
Using them helps demonstrate that the performance gains come from architectural choices, not from choosing an easy benchmark.

---

## 12. Is HexPylot production-ready?

No.

HexPylot is:
- a personal learning project
- an architectural experiment
- a demonstration of concepts

It is not:
- a drop-in replacement for existing frameworks
- a polished library with long-term support

---

## 13. Who is this project for?

HexPylot may be useful for:
- learners interested in parallel computing
- developers curious about Python’s hardware limits
- anyone exploring performance trade-offs honestly

It is not intended to compete with established HPC frameworks.

---

## 14. Why publish this project if it has limitations?

Because understanding *why* something does not scale is just as valuable as achieving speedup.

HexPylot documents:
- where Python performs well
- where it hits physical limits
- what can and cannot be optimized in software

That knowledge is the main purpose of this project.
