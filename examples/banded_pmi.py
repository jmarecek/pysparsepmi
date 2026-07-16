"""Banded polynomial matrix inequality in the style of Example 5.1 of

    Y. Zheng, G. Fantuzzi, "Sum-of-squares chordal decomposition of
    polynomial matrix inequalities" (arXiv:2007.11410).

A banded m-by-m homogeneous polynomial matrix P(x, y, z; lambda) depends
affinely on parameters lambda = (lam1, lam2). The dense SOS-matrix test for
``(x^2+y^2+z^2)^nu * P >> 0`` needs one Gram matrix of size m * |basis|;
the chordal decomposition replaces it with m-1 small clique blocks (this is
what the modified ``sos.csp`` option of YALMIP does for the scalarized
problem in the original MATLAB code).

Here we fix lam2 = 1 and maximize lam1 subject to feasibility of the sparse
SOS certificate.
"""
import cvxpy as cp

import pysparsepmi as psp


def run(d=2, nu=1, sparse=True):
    m = 3 * d
    x, y, z = psp.polyvar(3)
    lam1 = cp.Variable(name="lam1")
    lam2 = 1.0

    a = [y**4 + x**4 * lam2, z**4 + y**4 * lam2, x**4 + z**4 * lam2]
    b = [x**2 * y**2, y**2 * z**2, z**2 * x**2]

    zero = psp.Polynomial.zero(3)
    entries = [[zero for _ in range(m)] for _ in range(m)]
    for i in range(m - 1):  # 0-based version of the MATLAB loop
        c = lam1 if i % 2 == 0 else lam2
        i1, i2 = i % 3, (i + 1) % 3
        entries[i][i] = a[i1]
        entries[i + 1][i + 1] = a[i2]
        entries[i][i + 1] = entries[i + 1][i] = b[i1] * c

    P = psp.PolyMatrix(entries, nvars=3)
    con = psp.SOSMatrix(P, sparse=sparse, nu=nu)
    prob = psp.Problem(cp.Maximize(lam1), [con])
    prob.solve(solver="SCS", eps=1e-4, max_iters=50000)
    print(
        "m=%d nu=%d sparse=%s: max lam1 = %s (status %s), cliques=%s, "
        "Gram block sizes=%s"
        % (m, nu, sparse, prob.value, prob.status, con.cliques, con.block_sizes)
    )
    return prob.value


if __name__ == "__main__":
    run(sparse=True)
    run(sparse=False)
