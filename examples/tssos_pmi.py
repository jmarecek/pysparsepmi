"""Ports of the first polynomial-matrix examples from TSSOS's example/pmi.jl.

https://github.com/wangjie212/TSSOS/blob/master/example/pmi.jl

Both problems compute a lower bound on ``inf_x lambda_min(F(x))`` subject to
polynomial matrix inequality constraints ``G(x) >> 0``.
"""
import numpy as np

import pysparsepmi as psp


def example1():
    # inf mineig(F(x)) s.t. G(x) >= 0   (pmi.jl, first block)
    x1, x2 = psp.polyvar(2)
    Q = np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(3), 1 / np.sqrt(6)],
            [0.0, 1 / np.sqrt(3), 2 / np.sqrt(6)],
            [1 / np.sqrt(2), 1 / np.sqrt(3), -1 / np.sqrt(6)],
        ]
    )
    d = [
        -x1**2 - x2**2,
        (x1 + 1) ** 2 * (-0.25) + (x2 - 1) ** 2 * (-0.25),
        (x1 - 1) ** 2 * (-0.25) + (x2 + 1) ** 2 * (-0.25),
    ]
    # F = Q * diag(d) * Q'
    D = psp.PolyMatrix(
        [[d[i] if i == j else psp.Polynomial.zero(2) for j in range(3)] for i in range(3)]
    )
    F = psp.PolyMatrix(Q.astype(object), nvars=2) @ D @ psp.PolyMatrix(Q.T.astype(object), nvars=2)
    G = psp.PolyMatrix([[x1 * x2 * (-4.0) + 1, x1], [x1, x1**2 * (-1) - x2**2 + 4]])
    res = psp.pmi_optimize(F, ineqs=[G], order=2, solver="SCS", eps=1e-6, max_iters=50000)
    print("Example 1: bound on inf mineig(F) =", res.value, "(status: %s)" % res.status)
    return res


def example2():
    # second block of pmi.jl (3x3 F and G in 3 variables)
    x1, x2, x3 = psp.polyvar(3)
    one = psp.Polynomial.constant(3, 1.0)
    F = psp.PolyMatrix(
        [
            [x1**2 + 1, x1**2, x2**2],
            [x1**2, one, x3**2],
            [x2**2, x3**2, one],
        ]
    )
    s = x1 + x2 + x3
    G = psp.PolyMatrix(
        [
            [x1**2 * (-1) + 2, one, s],
            [one, x2**2 * (-1) + 2, one],
            [s, one, x3**2 * (-1) + 2],
        ]
    )
    res = psp.pmi_optimize(F, ineqs=[G], order=2, solver="SCS", eps=1e-6, max_iters=50000)
    print("Example 2: bound on inf mineig(F) =", res.value, "(status: %s)" % res.status)
    return res


if __name__ == "__main__":
    example1()
    example2()
