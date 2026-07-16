"""Correlative sparsity for scalar polynomial optimization (YALMIP sos.csp).

Minimizes a chained polynomial in n variables. The correlative sparsity
pattern is a path graph, so the Gram basis splits into n-1 cliques of size 2
instead of one dense basis over all n variables — the effect of setting
``opts.sos.csp = 1`` in YALMIP.
"""
import pysparsepmi as psp


def run(n=20):
    x = psp.polyvar(n)
    f = psp.Polynomial.zero(n)
    for i in range(n - 1):
        f = f + (x[i] * x[i + 1] - 1) ** 2 + 0.1 * (x[i] - x[i + 1]) ** 2

    res = psp.sos_lower_bound(f, order=2, solver="SCS", eps=1e-6, max_iters=20000)
    print("n = %d" % n)
    print("cliques:", res.cliques)
    print("Gram block sizes:", res.block_sizes)
    print("sparse SOS lower bound: %.6f (status %s)" % (res.value, res.status))
    print("(true minimum is 0, attained at x = (1, ..., 1))")
    return res


if __name__ == "__main__":
    run()
