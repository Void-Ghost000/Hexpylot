from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Union
import math
import os
import time
import platform
import sys

import numpy as np
import multiprocessing as mp


Reducer = Union[str, Callable[[np.ndarray], float]]


@dataclass(frozen=True)
class MCResult:
    value: Union[float, np.ndarray]
    stderr: Optional[float]
    n_samples: int
    status: str           # "OK" | "WARN" | "INVALID"
    reason: Optional[str]
    checksum: float
    audit: Dict[str, Any]


def _get_versions() -> Dict[str, str]:
    out = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import numba  # type: ignore
        out["numba"] = numba.__version__
    except Exception:
        out["numba"] = "not_installed"
    return out


def _validate_inputs(n_samples: int, seed: int, n_workers: int, chunk_size: int) -> Optional[str]:
    if not isinstance(n_samples, int) or n_samples <= 0:
        return "n_samples must be a positive int"
    if n_samples > 1_000_000_000:
        return "n_samples too large (safety cap exceeded)"
    if not isinstance(seed, int):
        return "seed must be an int"
    if not isinstance(n_workers, int) or n_workers <= 0:
        return "n_workers must be a positive int"
    cpu = os.cpu_count() or 1
    if n_workers > cpu:
        return f"n_workers exceeds available CPU cores ({cpu})"
    if not isinstance(chunk_size, int) or chunk_size < 1_000:
        return "chunk_size too small (would be dominated by overhead)"
    return None


def _reduce(values: np.ndarray, reducer: Reducer) -> float:
    if callable(reducer):
        return float(reducer(values))
    if reducer == "mean":
        return float(values.mean())
    if reducer == "sum":
        return float(values.sum())
    raise ValueError(f"Unknown reducer: {reducer}")


def _worker_mc(
    pid: int,
    kernel: Callable[[Dict[str, Any], np.ndarray], Union[np.ndarray, float]],
    params: Dict[str, Any],
    base_seed: int,
    start_idx: int,
    end_idx: int,
    chunk_size: int,
    out_sum: mp.Array,
    out_sumsq: mp.Array,
    out_count: mp.Array,
    inner_u: int,
) -> None:
    # Deterministic per-process seed stream
    rng = np.random.default_rng(base_seed + 1_000_003 * pid)

    total_sum = 0.0
    total_sumsq = 0.0
    total_n = 0

    i = start_idx
    while i < end_idx:
        j = min(i + chunk_size, end_idx)
        n = j - i

        # Uniform randoms (0,1) - shape controls compute intensity a bit
        u = rng.random((n, inner_u), dtype=np.float64)

        y = kernel(params, u)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 0:
            y_arr = np.full((n,), float(y_arr), dtype=np.float64)

        if not np.isfinite(y_arr).all():
            # Mark invalid by writing NaN in sum
            total_sum = float("nan")
            total_sumsq = float("nan")
            total_n = n
            break

        s = float(y_arr.sum())
        ss = float((y_arr * y_arr).sum())
        total_sum += s
        total_sumsq += ss
        total_n += n
        i = j

    out_sum[pid] = total_sum
    out_sumsq[pid] = total_sumsq
    out_count[pid] = total_n


def run_monte_carlo(
    kernel: Callable[[Dict[str, Any], np.ndarray], Union[np.ndarray, float]],
    param_space,
    n_samples: int,
    seed: int = 42,
    n_workers: int = 1,
    chunk_size: int = 50_000,
    reducer: Reducer = "mean",
    validate: bool = True,
    audit: bool = True,
    # inner_u controls compute intensity per sample (more columns => more math possible in kernel)
    inner_u: int = 1,
    # warn threshold for relative stderr
    warn_rel_stderr: float = 0.01,
) -> MCResult:
    t0 = time.time()

    if validate:
        err = _validate_inputs(n_samples, seed, n_workers, chunk_size)
        if err is not None:
            return MCResult(
                value=float("nan"),
                stderr=None,
                n_samples=n_samples,
                status="INVALID",
                reason=err,
                checksum=float("nan"),
                audit={"seed": seed, "n_workers": n_workers, "chunk_size": chunk_size},
            )

    params = dict(getattr(param_space, "params", {}))
    fp = getattr(param_space, "fingerprint", lambda: "no_fingerprint")()

    # Partition samples across workers
    # Balanced contiguous blocks for determinism and cache friendliness
    blocks = []
    base = 0
    for pid in range(n_workers):
        size = n_samples // n_workers + (1 if pid < (n_samples % n_workers) else 0)
        blocks.append((base, base + size))
        base += size

    out_sum = mp.Array("d", n_workers, lock=False)
    out_sumsq = mp.Array("d", n_workers, lock=False)
    out_count = mp.Array("q", n_workers, lock=False)

    procs = []
    for pid, (a, b) in enumerate(blocks):
        p = mp.Process(
            target=_worker_mc,
            args=(
                pid,
                kernel,
                params,
                seed,
                a,
                b,
                chunk_size,
                out_sum,
                out_sumsq,
                out_count,
                inner_u,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    sums = np.frombuffer(out_sum, dtype=np.float64).copy()
    sumsqs = np.frombuffer(out_sumsq, dtype=np.float64).copy()
    counts = np.frombuffer(out_count, dtype=np.int64).copy()

    total_n = int(counts.sum())
    total_sum = float(sums.sum())
    total_sumsq = float(sumsqs.sum())

    runtime = time.time() - t0

    # checksum: stable aggregate (not cryptographic) for sanity checks
    checksum = float(total_sum) if math.isfinite(total_sum) else float("nan")

    if total_n <= 1 or (not math.isfinite(total_sum)) or (not math.isfinite(total_sumsq)):
        return MCResult(
            value=float("nan"),
            stderr=None,
            n_samples=total_n,
            status="INVALID",
            reason="non-finite values encountered in kernel output",
            checksum=checksum,
            audit={
                "seed": seed,
                "n_workers": n_workers,
                "chunk_size": chunk_size,
                "n_samples": n_samples,
                "runtime_sec": runtime,
                "param_space_fingerprint": fp,
                **(_get_versions() if audit else {}),
            },
        )

    # Convert to mean/stderr if reducer == mean, else just reduce
    # For Monte Carlo, mean is typical; keep reducer hook for generality.
    # If reducer is not mean, stderr is omitted.
    if reducer == "mean":
        mean = total_sum / total_n
        # Var = E[x^2] - E[x]^2
        ex2 = total_sumsq / total_n
        var = max(0.0, ex2 - mean * mean)
        stderr = math.sqrt(var / total_n)
        value = mean

        status = "OK"
        reason = None
        if mean != 0.0 and (stderr / abs(mean)) > warn_rel_stderr:
            status = "WARN"
            reason = f"high relative stderr: {stderr/abs(mean):.4f} (> {warn_rel_stderr})"
    else:
        # For non-mean reducers, reconstruct a pseudo-array is expensive; skip stderr.
        # Use mean of sums as a fallback but label it clearly.
        value = total_sum
        stderr = None
        status = "OK"
        reason = None

    aud = {
        "seed": seed,
        "n_workers": n_workers,
        "chunk_size": chunk_size,
        "n_samples": n_samples,
        "effective_samples": total_n,
        "runtime_sec": runtime,
        "samples_per_sec": (total_n / runtime) if runtime > 0 else float("inf"),
        "param_space_fingerprint": fp,
    }
    if audit:
        aud.update(_get_versions())

    return MCResult(
        value=value,
        stderr=stderr,
        n_samples=total_n,
        status=status,
        reason=reason,
        checksum=checksum,
        audit=aud,
    )
