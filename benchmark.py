---

## 2) benchmark.py（一鍵跑：memory-bound + compute-bound；Hex vs Grid；輸出表格）

```python
#!/usr/bin/env python3
"""
HexPylot benchmark (Py3.10+; tested with Py3.12/Colab)

Runs two workloads:
  1) memory-bound diffusion (heavy=0)
  2) compute-bound validation (heavy=20 by default)

And compares two topologies under identical engine infrastructure:
  - Hex (odd-r offset, 6-neighbor)
  - Grid (4-neighbor)

Outputs: fps, speedup, checksum
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numba import njit
from multiprocessing import shared_memory, Barrier, get_context


Tile = Tuple[int, int, int, int]  # r0, r1, c0, c1


# -----------------------------
# Kernels (Numba)
# -----------------------------
@njit(nogil=True, fastmath=True)
def _heavy_mix(val: float, heavy_iters: int) -> float:
    x = val
    for _ in range(heavy_iters):
        x = math.sin(x) * math.cos(x) + math.sqrt(abs(x))
    if heavy_iters > 0:
        # keep the state stable; mix a tiny portion of heavy compute
        return 0.999 * val + 0.001 * x
    return val


@njit(nogil=True, fastmath=True)
def hex_step_odd_r(src: np.ndarray, dst: np.ndarray,
                   r0: int, r1: int, c0: int, c1: int,
                   alpha: float, heavy_iters: int) -> None:
    """
    Odd-r offset coordinates hex neighbors (pointy-top)
    Even row neighbors:
      (r-1,c-1) (r-1,c) (r,c-1) (r,c+1) (r+1,c-1) (r+1,c)
    Odd row neighbors:
      (r-1,c) (r-1,c+1) (r,c-1) (r,c+1) (r+1,c) (r+1,c+1)
    """
    H, W = src.shape
    for r in range(r0, r1):
        even = (r & 1) == 0
        for c in range(c0, c1):
            # keep borders untouched; caller should avoid border tiles
            v = src[r, c]
            if even:
                n1 = src[r - 1, c - 1]
                n2 = src[r - 1, c]
                n3 = src[r, c - 1]
                n4 = src[r, c + 1]
                n5 = src[r + 1, c - 1]
                n6 = src[r + 1, c]
            else:
                n1 = src[r - 1, c]
                n2 = src[r - 1, c + 1]
                n3 = src[r, c - 1]
                n4 = src[r, c + 1]
                n5 = src[r + 1, c]
                n6 = src[r + 1, c + 1]

            avg = (n1 + n2 + n3 + n4 + n5 + n6) / 6.0
            val = v + alpha * (avg - v)
            dst[r, c] = _heavy_mix(val, heavy_iters)


@njit(nogil=True, fastmath=True)
def grid_step_4n(src: np.ndarray, dst: np.ndarray,
                 r0: int, r1: int, c0: int, c1: int,
                 alpha: float, heavy_iters: int) -> None:
    """4-neighbor (Von Neumann) grid diffusion."""
    for r in range(r0, r1):
        for c in range(c0, c1):
            v = src[r, c]
            avg = (src[r - 1, c] + src[r + 1, c] + src[r, c - 1] + src[r, c + 1]) / 4.0
            val = v + alpha * (avg - v)
            dst[r, c] = _heavy_mix(val, heavy_iters)


# -----------------------------
# Engine (shared memory + barrier + parity toggle)
# -----------------------------
@dataclass(frozen=True)
class BenchConfig:
    H: int = 2048
    W: int = 2048
    steps: int = 50
    alpha: float = 0.2
    tile: int = 128
    seed_value: float = 100.0  # center impulse


def build_tiles(H: int, W: int, tile: int) -> List[Tile]:
    # Avoid borders to keep neighbor access safe (r-1, r+1, c-1, c+1).
    tiles: List[Tile] = []
    r = 1
    while r < H - 1:
        r1 = min(r + tile, H - 1)
        c = 1
        while c < W - 1:
            c1 = min(c + tile, W - 1)
            tiles.append((r, r1, c, c1))
            c += tile
        r += tile
    return tiles


def _attach_ndarray(shm_name: str, shape: Tuple[int, int]) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    return shm, arr


def worker(pid: int,
           shm_a_name: str,
           shm_b_name: str,
           shape: Tuple[int, int],
           tiles: List[Tile],
           steps: int,
           alpha: float,
           heavy_iters: int,
           barrier: Barrier,
           mode: str,
           nproc: int) -> None:
    # Attach shared memory in child
    shm_a, A = _attach_ndarray(shm_a_name, shape)
    shm_b, B = _attach_ndarray(shm_b_name, shape)
    try:
        for t in range(steps):
            # Parity toggle: zero-copy swap by choosing src/dst
            src, dst = (A, B) if (t & 1) == 0 else (B, A)

            # Each process takes every nproc-th tile (must use nproc, NOT cpu_count)
            for i in range(pid, len(tiles), nproc):
                r0, r1, c0, c1 = tiles[i]
                if mode == "hex":
                    hex_step_odd_r(src, dst, r0, r1, c0, c1, alpha, heavy_iters)
                else:
                    grid_step_4n(src, dst, r0, r1, c0, c1, alpha, heavy_iters)

            barrier.wait()
            barrier.wait()
    finally:
        shm_a.close()
        shm_b.close()


def run_case(cfg: BenchConfig, mode: str, nproc: int, heavy_iters: int, mp_ctx) -> Tuple[float, float]:
    shape = (cfg.H, cfg.W)
    tiles = build_tiles(cfg.H, cfg.W, cfg.tile)

    # init state
    A0 = np.zeros(shape, dtype=np.float64)
    A0[cfg.H // 2, cfg.W // 2] = cfg.seed_value

    shm_a = shared_memory.SharedMemory(create=True, size=A0.nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=A0.nbytes)

    # Important: keep these arrays alive until all children finish
    A = np.ndarray(shape, dtype=np.float64, buffer=shm_a.buf)
    B = np.ndarray(shape, dtype=np.float64, buffer=shm_b.buf)
    A[:] = A0
    B[:] = A0

    barrier = mp_ctx.Barrier(nproc)
    procs = []

    t0 = time.perf_counter()
    try:
        for pid in range(nproc):
            p = mp_ctx.Process(
                target=worker,
                args=(pid, shm_a.name, shm_b.name, shape, tiles,
                      cfg.steps, cfg.alpha, heavy_iters, barrier, mode, nproc),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        t1 = time.perf_counter()

        # checksum from final buffer
        final_arr = A if (cfg.steps & 1) == 0 else B
        chk = float(final_arr.sum())
        fps = cfg.steps / (t1 - t0)
        return fps, chk

    finally:
        # Cleanup processes (best effort)
        for p in procs:
            if p.is_alive():
                p.terminate()

        # Release shared memory
        try:
            shm_a.close()
            shm_b.close()
        except Exception:
            pass
        try:
            shm_a.unlink()
            shm_b.unlink()
        except Exception:
            pass


def clamp_workers(max_workers: int) -> List[int]:
    # Standard set; keep only <= max_workers and unique
    base = [1, 2, 4, 8, 16, 32]
    return [n for n in base if n <= max_workers]


def print_env() -> None:
    print("===== HexPylot Fair Benchmark =====")
    print(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")
    print(f"CPU cores available (os.cpu_count): {os.cpu_count()}")


def run_suite(cfg: BenchConfig, heavy_iters: int) -> None:
    # Colab/Jupyter: use 'fork' on Linux for practicality; fallback to 'spawn'
    # NOTE: running multiprocessing inside notebooks can be finicky.
    # This script is intended to be run as: python benchmark.py
    start_method = "fork"
    try:
        mp_ctx = get_context(start_method)
    except ValueError:
        start_method = "spawn"
        mp_ctx = get_context(start_method)

    max_workers = os.cpu_count() or 1
    workers = clamp_workers(max_workers)

    title = "REAL diffusion (memory-bound)" if heavy_iters == 0 else f"COMPUTE-bound (heavy load), heavy={heavy_iters}"
    print(f"\n=== {title} ===")
    print(f"Config: Grid={cfg.H}x{cfg.W} | Steps={cfg.steps} | Alpha={cfg.alpha} | Tile={cfg.tile} | mp={start_method}")

    for mode in ("hex", "grid"):
        print(f"\n{ 'Hex (6-neighbor)' if mode=='hex' else 'Grid (4-neighbor, fair baseline)' }:")
        base_fps = None
        base_chk = None

        for n in workers:
            fps, chk = run_case(cfg, mode, n, heavy_iters, mp_ctx)
            if base_fps is None:
                base_fps = fps
                base_chk = chk

            speedup = fps / base_fps if base_fps else 1.0

            # checksum sanity: for deterministic init, checksum should be stable vs n
            # (small float differences may happen; here we only print)
            print(f"  n={n:<2d}  fps={fps:8.3f}  speedup={speedup:5.2f}x  chk={chk:.6f}")

        # optional quick note if checksum drifts
        if base_chk is not None:
            pass


def main() -> None:
    print_env()

    cfg = BenchConfig(
        H=2048,
        W=2048,
        steps=50,
        alpha=0.2,
        tile=128,
        seed_value=100.0
    )

    # 1) memory-bound
    run_suite(cfg, heavy_iters=0)

    # 2) compute-bound validation
    run_suite(cfg, heavy_iters=20)


if __name__ == "__main__":
    main()
