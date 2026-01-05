from __future__ import annotations
import math
import time
import numpy as np

from hexpylot_mc.engine import run_monte_carlo
from hexpylot_mc.params import ParamSpace


def bs_call_payoff(params: dict, u: np.ndarray) -> np.ndarray:
    """
    European call option via Black-Scholes under risk-neutral measure.
    Uses Box-Muller to convert uniform -> normal.
    u: shape (n, inner_u). We only use first 2 columns for Box-Muller, but you can
       set inner_u larger and add extra compute (inner_heavy) to stress compute scaling.
    """
    S0 = float(params["S0"])
    K = float(params["K"])
    r = float(params["r"])
    sigma = float(params["sigma"])
    T = float(params["T"])
    inner_heavy = int(params.get("inner_heavy", 0))

    # Ensure we have at least 2 uniforms for Box-Muller
    if u.shape[1] < 2:
        u = np.concatenate([u, np.clip(u, 1e-12, 1.0 - 1e-12)], axis=1)

    u1 = np.clip(u[:, 0], 1e-12, 1.0 - 1e-12)
    u2 = np.clip(u[:, 1], 1e-12, 1.0 - 1e-12)

    # Box-Muller
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)

    # Risk-neutral terminal price
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    ST = S0 * np.exp(drift + vol * z)

    payoff = np.maximum(ST - K, 0.0)

    # Optional extra compute to make it more compute-bound
    # (kept deterministic, no extra randomness)
    if inner_heavy > 0:
        x = payoff
        for _ in range(inner_heavy):
            x = np.sin(x) * np.cos(x) + np.sqrt(np.abs(x) + 1e-12)
        payoff = 0.999 * payoff + 0.001 * x

    # Discount
    return np.exp(-r * T) * payoff


def main():
    ps = ParamSpace(
        params={
            "S0": 100.0,
            "K": 100.0,
            "r": 0.03,
            "sigma": 0.2,
            "T": 1.0,
            # Increase this to stress CPU and observe scaling
            "inner_heavy": 0,
        }
    )

    n_samples = 2_000_000
    seed = 42
    chunk_size = 100_000

    print("=== HexPylot-MC demo: Black-Scholes Call (Monte Carlo) ===")
    print(f"params_fp={ps.fingerprint()}")
    print(f"n_samples={n_samples:,} seed={seed} chunk_size={chunk_size:,}")

    for workers in [1, 2, 4, 8]:
        t0 = time.time()
        res = run_monte_carlo(
            kernel=bs_call_payoff,
            param_space=ps,
            n_samples=n_samples,
            seed=seed,
            n_workers=workers,
            chunk_size=chunk_size,
            reducer="mean",
            validate=True,
            audit=True,
            inner_u=2,            # two uniforms per sample
            warn_rel_stderr=0.01
        )
        dt = time.time() - t0

        print(f"\n--- workers={workers} ---")
        print(f"value={res.value:.6f}  stderr={res.stderr:.6f}  status={res.status}")
        if res.reason:
            print(f"reason={res.reason}")
        print(f"checksum={res.checksum:.6f}")
        print(f"runtime={dt:.3f}s  samples/sec={res.audit.get('samples_per_sec', float('nan')):,.0f}")
        print(f"audit(seed/workers/chunk)={(res.audit['seed'], res.audit['n_workers'], res.audit['chunk_size'])}")

    print("\nTip: set inner_heavy=20 to make it compute-bound and observe scaling more clearly.")


if __name__ == "__main__":
    main()
