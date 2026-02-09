import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random


# ─── Ramanujan asymptotic bound (same as before) ─────────────────────────────────

def ramanujan_asymptotic(k: int) -> float:
    """Hardy–Ramanujan approx p(k) ≈ 1/(4 k √3) exp(π √(2k/3))"""
    if k <= 0:
        return 0.0
    pref = 1.0 / (4.0 * k * math.sqrt(3.0))
    expo = math.pi * math.sqrt(2.0 * k / 3.0)
    return pref * math.exp(expo)


def compute_rasf_bound(
    d: int,
    p: float,
    k_max_offset: int = 80,
) -> float:
    """Upper bound P_L ≲ ∑_{k≥d} a_k q^k   with a_k ≈ Ramanujan p(k)"""
    if p <= 0 or p >= 0.5:
        return 1.0 if p >= 0.5 else 0.0

    q = p / (1.0 - p)
    start_k = d
    end_k = start_k + k_max_offset

    log_sum = -np.inf
    for k in range(start_k, end_k + 1):
        term = ramanujan_asymptotic(k) * (q ** k)
        if term > 0:
            log_sum = np.logaddexp(log_sum, math.log(term))

    return math.exp(log_sum) if np.isfinite(log_sum) else 0.0


# ─── Very simple phenomenological Monte Carlo for rotated surface code ──────────
# Assumptions: independent X & Z errors with prob p per data qubit per round
# We simulate d×d data qubits, (d-1)×(d-1) × & z stabilizers
# Logical X = horizontal string of X on data, Logical Z = vertical string of Z

def simple_surface_code_mc(
    d: int,
    p: float,
    shots: int = 200_000,
    rounds: int = 1,
) -> float:
    """
    Toy Monte Carlo: phenomenological noise on rotated surface code memory
    Returns fraction of shots with logical error (X or Z logical flip)
    Very crude — no boundary handling, assumes periodic-like, no measurement error
    """
    if d % 2 == 0:
        raise ValueError("Only odd d supported in this toy version")

    logical_errors = 0

    for _ in range(shots):
        # syndrome accumulation over rounds (simple majority over rounds not done here)
        x_synd = np.zeros((d-1, d-1), dtype=int)   # plaquettes
        z_synd = np.zeros((d-1, d-1), dtype=int)   # stars

        for r in range(rounds):
            # Apply errors to data qubits
            data_x = np.random.random((d, d)) < p
            data_z = np.random.random((d, d)) < p

            # Compute plaquette (X) syndromes
            for i in range(d-1):
                for j in range(d-1):
                    flips = 0
                    flips += data_x[i,   j  ]
                    flips += data_x[i+1, j  ]
                    flips += data_x[i,   j+1]
                    flips += data_x[i+1, j+1]
                    x_synd[i,j] ^= (flips % 2)

            # Compute star (Z) syndromes — similar but dual lattice
            for i in range(d-1):
                for j in range(d-1):
                    flips = 0
                    flips += data_z[i,   j  ]
                    flips += data_z[i+1, j  ]
                    flips += data_z[i,   j+1]
                    flips += data_z[i+1, j+1]
                    z_synd[i,j] ^= (flips % 2)

        # Very crude decoder: assume logical error if total syndrome parity odd
        # (this is WRONG for real decoding — only illustrative!)
        x_total_parity = np.sum(x_synd) % 2
        z_total_parity = np.sum(z_synd) % 2

        if x_total_parity == 1 or z_total_parity == 1:
            logical_errors += 1

    return logical_errors / shots


# ─── Plot comparison ────────────────────────────────────────────────────────────

def plot_comparison(
    distances: List[int] = [3, 5, 7],
    p_min: float = 1e-3,
    p_max: float = 0.12,
    n_p: int = 40,
):
    p_values = np.logspace(np.log10(p_min), np.log10(p_max), n_p)

    plt.figure(figsize=(10, 6))

    for d in distances:
        # RASF bound
        rasf = [compute_rasf_bound(d, p) for p in p_values]

        # Toy MC (takes time — reduce shots for testing)
        mc = []
        for p in p_values:
            perr = simple_surface_code_mc(d, p, shots=80_000 if d<=5 else 40_000)
            mc.append(perr)
            print(f"d={d}, p={p:.4f} → MC {perr:.2e}")

        plt.semilogy(p_values, rasf, '--', lw=2, label=f'RASF bound d={d}')
        plt.semilogy(p_values, mc,   'o-', lw=1.5, alpha=0.8, label=f'Toy MC d={d}')

    plt.xlabel('physical error rate p')
    plt.ylabel('logical error rate P_L')
    plt.title('RASF bound vs toy phenomenological Monte Carlo\n'
              '(note: toy MC decoder is crude → underestimates protection)')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.ylim(1e-8, 1)
    plt.xlim(p_min, p_max)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick test points
    for d in [3,5,7]:
        p = 0.01
        bound = compute_rasf_bound(d, p)
        print(f"d={d:2d}  p={p:.3f} → RASF bound {bound:.2e}")

    # Generate comparison plot (will take several minutes)
    plot_comparison(distances=[3,5], p_max=0.10, n_p=25)