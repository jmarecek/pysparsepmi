"""Worst-case gain of a product of parameter-dependent matrices (PMI lifting).

For P(x) = M(x_1) M(x_2) ... M(x_n) with x_i in [-1, 1], bound

    max_x lambda_max(P(x)' P(x)),

the worst-case squared gain of a linear time-varying system. The polynomial
matrix P'P has degree 2n in n variables. It is evaluated through the
symmetric states S_0 = I, S_i = M(x_i)' S_{i-1} M(x_i), so lifting the
three entries of each 2x2 state gives chain structure. The last map returns
the matrix S_n itself, so the relaxations certify a polynomial matrix
inequality (the matrix-valued extension of SL-chord / SL-push in
pysparsepmi). The bounds are compared with sampling on a grid, which gives
a lower estimate of the maximum.
"""
import itertools
import time

import numpy as np

import pysparsepmi as psp


def M(x):  # works for floats and Polynomials
    return [[0.9, 0.5 * x], [0.3 * x, 0.8]]


def congruence(S, x):
    """M(x)' S M(x) for a symmetric 2x2 S given as [s11, s12, s22]."""
    Mx = M(x)
    Sm = [[S[0], S[1]], [S[1], S[2]]]
    return [
        [sum(Mx[c][a] * Sm[c][d] * Mx[d][b] for c in range(2) for d in range(2))
         for b in range(2)]
        for a in range(2)
    ]


def first(s, x):
    R = congruence([1.0, 0.0, 1.0], x)
    return [R[0][0], R[0][1], R[1][1]]


def step(s, x):
    R = congruence(s, x)
    return [R[0][0], R[0][1], R[1][1]]


def last(s, x):
    return congruence(s, x)  # matrix-valued terminal map


def sampled_max(n, pts=5):
    best = 0.0
    for v in itertools.product(np.linspace(-1, 1, pts), repeat=n):
        P = np.eye(2)
        for x in v:
            P = P @ np.array(M(x))
        best = max(best, np.linalg.eigvalsh(P.T @ P).max())
    return best


def run(n=6):
    maps = [first] + [step] * (n - 2) + [last]
    print("n = %d, sampled max lambda_max >= %.5f" % (n, sampled_max(n)))
    for method, order in (("push", 2), ("push", 3), ("chord", 2)):
        t0 = time.time()
        res = psp.composition_lower_bound(
            maps, box=1.0, order=order, method=method, sense="max",
            solver="SCS", eps=1e-7, max_iters=100000,
        )
        print(
            "  %-5s order %d: upper bound %.5f (%s), max PSD block %d, %.1fs"
            % (method, order, res.value, res.status, max(res.block_sizes),
               time.time() - t0)
        )


if __name__ == "__main__":
    run()
