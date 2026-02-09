import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


def ramanujan_asymptotic(k: int) -> float:
    """
    Hardy–Ramanujan asymptotic approximation for the partition function p(k)
    
    a_k ≈ 1/(4 k √3) * exp( π √(2k/3) )
    
    This is used as an estimate of the degeneracy of weight-k logical errors.
    
    Parameters
    ----------
    k : int
        Weight (number of physical errors / flipped qubits)
        
    Returns
    -------
    float
        Approximate number of ways ≈ p(k)
    """
    if k <= 0:
        return 0.0
    prefactor = 1.0 / (4.0 * k * math.sqrt(3.0))
    exponent   = math.pi * math.sqrt(2.0 * k / 3.0)
    return prefactor * math.exp(exponent)


def compute_rasf_bound(
    d: int,
    p: float,
    k_max_offset: int = 40,
    k_min: Optional[int] = None
) -> float:
    """
    Compute the RASF (Ramanujan Asymptotics Surface-code Failure) upper bound
    on logical error probability P_L(d,p) using the truncated sum:

        P_L ≲ ∑_{k ≥ d} a_k(d) q^k
    
    where q = p / (1-p)   and   a_k ≈ Ramanujan asymptotic

    Parameters
    ----------
    d : int
        Surface code distance
        
    p : float
        Physical error probability per qubit (0 ≤ p < 0.5)
        
    k_max_offset : int, optional
        How many terms beyond the minimal weight d to include
        
    k_min : int or None, optional
        Override minimal summation start (usually d)
        
    Returns
    -------
    float
        Approximate upper bound on logical failure probability
    """
    if p <= 0 or p >= 0.5:
        return 1.0 if p >= 0.5 else 0.0
    
    q = p / (1.0 - p)           # Boltzmann weight (q < 1 when p < 0.5)
    
    start_k = max(d, k_min) if k_min is not None else d
    end_k   = start_k + k_max_offset
    
    total = 0.0
    log_total = -np.inf         # more stable for very small numbers
    
    for k in range(start_k, end_k + 1):
        a_k = ramanujan_asymptotic(k)
        term = a_k * (q ** k)
        
        if term > 0:
            log_term = math.log(term)
            log_total = np.logaddexp(log_total, log_term)
    
    if np.isfinite(log_total):
        total = math.exp(log_total)
    else:
        total = 0.0
    
    return total


def plot_rasf_vs_p(
    distances: List[int] = [3, 5, 7, 9],
    p_values: np.ndarray = np.logspace(-3.5, -0.5, 80),
    k_max_offset: int = 60,
    ymin: float = 1e-12,
    ymax: float = 1.0
):
    """
    Quick visualization of RASF bounds for several code distances
    """
    plt.figure(figsize=(10, 6.5))
    
    for d in distances:
        bounds = []
        for p in p_values:
            bound = compute_rasf_bound(d, p, k_max_offset=k_max_offset)
            bounds.append(bound)
        
        plt.semilogy(p_values, bounds, lw=2.1, label=f'd = {d}')
    
    plt.xlabel('Physical error probability  p')
    plt.ylabel('Logical error probability upper bound  P_L')
    plt.title('RASF analytical bound  (Ramanujan asymptotic degeneracy)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.xlim(p_values[0], p_values[-1])
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────
#                  Example usage
# ────────────────────────────────────────────────

if __name__ == "__main__":

    # Single point checks
    print("RASF bounds at p = 0.01:")
    for d in [3,5,7,9,11]:
        pl = compute_rasf_bound(d, p=0.01, k_max_offset=80)
        print(f"  d = {d:2d} → P_L ≲ {pl:.2e}")

    # Generate the typical plot shown in papers
    p_range = np.logspace(-3.2, -1.0, 100)   # p = 0.0006 … 0.1
    plot_rasf_vs_p(
        distances=[3,5,7,9,13],
        p_values=p_range,
        k_max_offset=100,
        ymin=1e-14,
        ymax=5e-1
    )