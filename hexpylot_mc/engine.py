"""  
HexPylot Monte Carlo Engine  
  
Minimal Monte Carlo engine with QOIM-inspired audit trail.  
Uses lock-free shared memory for zero-copy parallel execution.  
  
Supports: "mean" and "sum" reducers.  
Requires: kernel functions compatible with vectorized uniform inputs.  
"""  
  
from __future__ import annotations  
  
import math  
import os  
import sys  
import time  
import platform  
import hashlib  
import json  
import multiprocessing as mp  
from dataclasses import dataclass  
from typing import Callable, Dict, Any, Optional, Tuple  
from multiprocessing.shared_memory import SharedMemory  
  
import numpy as np  
  
  
# ---------------------------------------------------------------------  
# Result & Audit  
# ---------------------------------------------------------------------  
  
@dataclass(frozen=True)  
class MCResult:  
    """Monte Carlo execution result with audit trail."""  
    value: float  
    stderr: Optional[float]  
    n_samples: int  
    status: str  # "OK" | "INVALID"  
    checksum: float  
    audit: Dict[str, Any]  
  
  
def _get_audit_info(  
    seed: Optional[int],  
    n_workers: int,  
    n_samples: int,  
    runtime_sec: float,  
    params: Dict,  
) -> Dict[str, Any]:  
    """Collect minimal audit information for reproducibility."""  
    param_str = json.dumps(params, sort_keys=True)  
    param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]  
      
    return {  
        "seed": seed,  
        "n_workers": n_workers,  
        "n_samples": n_samples,  
        "runtime_sec": round(runtime_sec, 3),  
        "samples_per_sec": int(n_samples / runtime_sec) if runtime_sec > 0 else 0,  
        "param_fingerprint": param_hash,  
        "python_version": sys.version.split()[0],  
        "numpy_version": np.__version__,  
        "platform": platform.platform(),  
    }  
  
  
# ---------------------------------------------------------------------  
# Validation  
# ---------------------------------------------------------------------  
  
def _validate_inputs(  
    n_samples: int,  
    n_workers: int,  
    inner_u: int,  
    reducer: str,  
) -> Optional[str]:  
    """Validate inputs according to QOIM feasible domain."""  
    if n_samples <= 0:  
        return "n_samples must be > 0"  
    if n_samples > 1_000_000_000:  
        return "n_samples exceeds safety cap (1e9)"  
    if n_workers <= 0:  
        return "n_workers must be > 0"  
    cpu_count = os.cpu_count() or 1  
    if n_workers > cpu_count:  
        return f"n_workers ({n_workers}) exceeds CPU count ({cpu_count})"  
    if inner_u < 2:  
        return "inner_u must be >= 2"  
    if reducer not in ("mean", "sum"):  
        return f"unsupported reducer: {reducer}"  
    return None  
  
  
# ---------------------------------------------------------------------  
# Worker  
# ---------------------------------------------------------------------  
  
def _worker(  
    pid: int,  
    shm_u_name: str,  
    shm_out_name: str,  
    u_shape: Tuple[int, int],  
    blocks: Tuple[Tuple[int, int], ...],  
    kernel: Callable[[Dict, np.ndarray], np.ndarray],  
    params: Dict,  
) -> None:  
    """  
    Worker process: compute kernel over assigned block.  
    Writes (sum, sum_of_squares) to shared output array.  
    """  
    shm_u = SharedMemory(name=shm_u_name)  
    shm_out = SharedMemory(name=shm_out_name)  
  
    try:  
        # Attach to shared arrays  
        u = np.ndarray(u_shape, dtype=np.float64, buffer=shm_u.buf)  
        out = np.ndarray((mp.cpu_count(), 2), dtype=np.float64, buffer=shm_out.buf)  
  
        start, end = blocks[pid]  
          
        # Call kernel with slice of uniform randoms  
        y = kernel(params, u[start:end])  
        y = np.asarray(y, dtype=np.float64)  
  
        # Check validity  
        if not np.isfinite(y).all():  
            out[pid, 0] = float("nan")  
            out[pid, 1] = float("nan")  
        else:  
            out[pid, 0] = float(np.sum(y))  
            out[pid, 1] = float(np.sum(y * y))  
  
    finally:  
        shm_u.close()  
        shm_out.close()  
  
  
# ---------------------------------------------------------------------  
# Engine  
# ---------------------------------------------------------------------  
  
def run_monte_carlo(  
    *,  
    kernel: Callable[[Dict, np.ndarray], np.ndarray],  
    params: Dict,  
    n_samples: int,  
    n_workers: int = 1,  
    inner_u: int = 2,  
    reducer: str = "mean",  
    seed: Optional[int] = None,  
) -> MCResult:  
    """  
    Execute Monte Carlo simulation with lock-free parallelism.  
  
    Parameters  
    ----------  
    kernel : callable  
        Vectorized function(params, u) -> array of payoffs.  
        Must accept u.shape = (n, inner_u) and return shape (n,).  
    params : dict  
        Parameters passed to kernel.  
    n_samples : int  
        Total number of Monte Carlo samples.  
    n_workers : int  
        Number of parallel worker processes.  
    inner_u : int  
        Number of uniform randoms per sample (must be >= 2).  
    reducer : str  
        Aggregation method: "mean" or "sum".  
    seed : int, optional  
        Random seed for reproducibility.  
  
    Returns  
    -------  
    MCResult  
        Execution result with value, stderr, audit trail.  
    """  
      
    t0 = time.time()  
      
    # Validation  
    err = _validate_inputs(n_samples, n_workers, inner_u, reducer)  
    if err is not None:  
        return MCResult(  
            value=float("nan"),  
            stderr=None,  
            n_samples=0,  
            status="INVALID",  
            checksum=float("nan"),  
            audit={"error": err},  
        )  
  
    # Generate all uniform randoms upfront  
    rng = np.random.default_rng(seed)  
    u = rng.random((n_samples, inner_u), dtype=np.float64)  
  
    # Partition work across workers  
    blocks = []  
    base = 0  
    for pid in range(n_workers):  
        size = n_samples // n_workers  
        if pid < (n_samples % n_workers):  
            size += 1  
        blocks.append((base, base + size))  
        base += size  
  
    # Create shared memory  
    shm_u = SharedMemory(create=True, size=u.nbytes)  
    shm_out = SharedMemory(create=True, size=mp.cpu_count() * 2 * 8)  
  
    try:  
        # Copy uniform randoms to shared memory  
        u_shared = np.ndarray(u.shape, dtype=u.dtype, buffer=shm_u.buf)  
        u_shared[:] = u  
          
        # Initialize output array  
        out = np.ndarray((mp.cpu_count(), 2), dtype=np.float64, buffer=shm_out.buf)  
        out[:] = 0.0  
  
        # Launch workers  
        procs = []  
        for pid in range(n_workers):  
            p = mp.Process(  
                target=_worker,  
                args=(  
                    pid,  
                    shm_u.name,  
                    shm_out.name,  
                    u.shape,  
                    tuple(blocks),  
                    kernel,  
                    params,  
                ),  
            )  
            p.start()  
            procs.append(p)  
  
        # Wait for completion  
        for p in procs:  
            p.join()  
  
        # Aggregate results  
        total_sum = float(np.sum(out[:n_workers, 0]))  
        total_sumsq = float(np.sum(out[:n_workers, 1]))  
          
        runtime = time.time() - t0  
          
        # Check for invalid results  
        if not math.isfinite(total_sum) or not math.isfinite(total_sumsq):  
            return MCResult(  
                value=float("nan"),  
                stderr=None,  
                n_samples=n_samples,  
                status="INVALID",  
                checksum=float("nan"),  
                audit=_get_audit_info(seed, n_workers, n_samples, runtime, params),  
            )  
  
        # Compute final result based on reducer  
        if reducer == "sum":  
            value = total_sum  
            stderr = None  
        else:  # "mean"  
            value = total_sum / n_samples  
            ex2 = total_sumsq / n_samples  
            var = max(0.0, ex2 - value * value)  
            stderr = math.sqrt(var / n_samples)  
  
        return MCResult(  
            value=value,  
            stderr=stderr,  
            n_samples=n_samples,  
            status="OK",  
            checksum=total_sum,  # Use sum as stable checksum  
            audit=_get_audit_info(seed, n_workers, n_samples, runtime, params),  
        )  
  
    finally:  
        shm_u.close()  
        shm_u.unlink()  
        shm_out.close()  
        shm_out.unlink()
