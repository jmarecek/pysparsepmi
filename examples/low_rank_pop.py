"""Low-rank (CP) polynomial optimization with LRPOP (arXiv:2512.08394).

Minimizes f(x) = sum_{l=1}^r prod_{i=1}^n f_{l,i}(x_i) over [-1, 1]^n, with
the well-conditioned instances of Section 4.2 of the paper: every univariate
factor is given in the Bernstein basis with coefficients b_0 = 1 and
b_j in [1 + delta/n, 1 + 2 delta/n], so each factor has minimum 1 at x = -1
and the global minimum is exactly r.

The polynomial has total degree n*d and is dense (every variable interacts
with every other), so neither correlative nor term sparsity helps. Lifting
the partial products t_{l,i} = t_{l,i-1} f_{l,i}(x_i) gives a chain whose
cliques have r + 2 variables whatever n is: the PSD block size stays
constant while n grows.

SCS, the default first-order solver, needs many iterations on long chains;
a result with status ``optimal_inaccurate`` is not a certified bound (it can
even exceed the true minimum). An interior-point solver such as MOSEK
(``solver="MOSEK"``) is much faster and more accurate at this scale.
"""
import time

import numpy as np

import pysparsepmi as psp


def instance(n, r=2, d=2, delta=1.0, seed=0):
    rng = np.random.RandomState(seed)
    factors = []
    for _ in range(n):
        F = rng.uniform(1 + delta / n, 1 + 2 * delta / n, size=(d + 1, r))
        F[0, :] = 1.0
        factors.append(F)
    return factors


def run(sizes=(5, 10, 20), r=2):
    print("rank r = %d, local degree 2, expected minimum %g" % (r, r))
    for n in sizes:
        factors = instance(n, r=r)
        for method in ("chord", "push"):
            t0 = time.time()
            res = psp.cp_lower_bound(
                factors, box=1.0, basis="bernstein", order=2, method=method,
                solver="SCS", eps=1e-7, max_iters=200000,
            )
            print(
                "n = %3d  %-5s  bound %.5f  (%s)  max PSD block %d, %d blocks, %.1fs"
                % (n, method, res.value, res.status, max(res.block_sizes),
                   len(res.block_sizes), time.time() - t0)
            )


if __name__ == "__main__":
    run()
