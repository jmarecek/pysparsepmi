"""Term-sparse PMI optimization (TSSOS's ``TS="MD"`` option).

Python port of the TSSOS example

    using DynamicPolynomials, TSSOS
    @polyvar x[1:5]
    F = [...]                          # 5x5 polynomial matrix, see below
    G = [G1, G2]                       # two 2x2 PMI constraints
    opt, sol, data = tssos(F, G, x, 3, TS="MD")   # first TS step
    opt, sol, data = tssos(data, TS="MD")         # higher TS steps

following Miller, Wang & Guo, "Sparse Polynomial Matrix Optimization"
(arXiv:2411.15479), Section 4. ``ts_order=s`` computes the s-th step of the
term-sparsity hierarchy; ``ts_order=None`` iterates until the block
structure stabilizes.
"""
import time

import pysparsepmi as psp

x = psp.polyvar(5)
z = psp.Polynomial.zero(5)

F = psp.PolyMatrix(
    [
        [x[0] ** 4, x[0] ** 2 - x[1] * x[2], x[2] ** 2 - x[3] * x[4], x[0] * x[3], x[0] * x[4]],
        [x[0] ** 2 - x[1] * x[2], x[1] ** 4, x[1] ** 2 - x[2] * x[3], x[1] * x[3], x[1] * x[4]],
        [x[2] ** 2 - x[3] * x[4], x[1] ** 2 - x[2] * x[3], x[2] ** 4, x[3] ** 2 - x[0] * x[1], x[4] ** 2 - x[2] * x[4]],
        [x[0] * x[3], x[1] * x[3], x[3] ** 2 - x[0] * x[1], x[3] ** 4, x[3] ** 2 - x[0] * x[2]],
        [x[0] * x[4], x[1] * x[4], x[4] ** 2 - x[2] * x[4], x[3] ** 2 - x[0] * x[2], x[4] ** 4],
    ]
)
G1 = psp.PolyMatrix(
    [[1 - x[0] ** 2 - x[1] ** 2, x[1] * x[2]], [x[1] * x[2], 1 - x[2] ** 2]], nvars=5
)
G2 = psp.PolyMatrix(
    [[1 - x[3] ** 2, x[3] * x[4]], [x[3] * x[4], 1 - x[4] ** 2]], nvars=5
)

for s in (1, 2, None):
    t0 = time.time()
    res = psp.pmi_optimize(
        F, ineqs=[G1, G2], order=3, ts="MD", ts_order=s,
        solver="SCS", eps=1e-6, max_iters=50000,
    )
    label = "stabilized (s=%d)" % res.ts_order if s is None else "s=%d" % s
    print(
        "TS='MD' %-16s  bound %.6f  (%s, %.1fs)"
        % (label, res.value, res.status, time.time() - t0)
    )
    print("  certificate blocks:", sorted(res.block_sizes, reverse=True))
    for k, sizes in enumerate(res.ts_block_sizes[1:], 1):
        print("  localizer %d blocks:" % k, sorted(sizes, reverse=True))
