"""Controlled two-state Markov chain via state lifting (arXiv:2604.17563, Sec. 7.1).

The state distribution evolves as v_i = v_{i-1} P(x_i) with a controlled
transition matrix

    P(x) = [[a(x), 1 - a(x)], [b(x), 1 - b(x)]],  a(x) = 0.95 - 0.2 x^2,
                                                   b(x) = 0.05 - 0.05 x^2,

and controls x_i in [-1, 1]. The probability of being in the working state
at time n is a tensor-train polynomial of degree 2n. Its maximum is
1/2 + 0.9^n / 2 (controls off), which lets us check the certified upper
bounds of SL-chord and SL-push. The states are the rank-2 distributions,
so the PSD blocks do not grow with n.
"""
import time

import pysparsepmi as psp


def a(x):
    return 0.95 - 0.20 * x**2


def b(x):
    return 0.05 - 0.05 * x**2


def first(s, x):
    return [a(x), 1 - a(x)]


def step(s, x):
    return [s[0] * a(x) + s[1] * b(x), s[0] * (1 - a(x)) + s[1] * (1 - b(x))]


def last(s, x):
    return s[0] * a(x) + s[1] * b(x)


def run(sizes=(5, 10, 20)):
    for n in sizes:
        maps = [first] + [step] * (n - 2) + [last]
        exact = 0.5 + 0.5 * 0.9**n
        for method in ("chord", "push"):
            t0 = time.time()
            res = psp.composition_lower_bound(
                maps, box=1.0, order=2, sense="max", method=method,
                solver="SCS", eps=1e-7, max_iters=100000,
            )
            print(
                "n = %2d  %-5s  upper bound %.6f  exact %.6f  (%s)  max PSD block %d, %.1fs"
                % (n, method, res.value, exact, res.status, max(res.block_sizes),
                   time.time() - t0)
            )


if __name__ == "__main__":
    run()
