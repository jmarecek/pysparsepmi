"""Tests for pysparsepmi. Run with ``pytest`` or ``python tests/test_pysparsepmi.py``."""
import numpy as np
import cvxpy as cp

import pysparsepmi as psp

TOL = 5e-3  # SCS is a first-order solver; keep tolerances loose
SOLVER_OPTS = dict(solver="SCS", eps=1e-6, max_iters=20000)


# ----------------------------------------------------------------------
# chordal / correlative sparsity
# ----------------------------------------------------------------------
def test_chordal_cliques_chain():
    n = 5
    A = np.eye(n, dtype=int)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    cliques = psp.chordal_cliques(A)
    assert cliques == [[i, i + 1] for i in range(n - 1)]


def test_chordal_cliques_star_and_cycle():
    # star: center 0 -- already chordal, cliques {0, i}
    n = 5
    A = np.eye(n, dtype=int)
    A[0, 1:] = A[1:, 0] = 1
    assert psp.chordal_cliques(A) == [[0, i] for i in range(1, n)]
    # 4-cycle: chordal extension adds one chord -> two triangles
    B = np.eye(4, dtype=int)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        B[i, j] = B[j, i] = 1
    cliques = psp.chordal_cliques(B)
    assert len(cliques) == 2 and all(len(c) == 3 for c in cliques)


def test_chordal_cliques_isolated_vertex():
    A = np.zeros((3, 3), dtype=int)
    A[0, 1] = A[1, 0] = 1
    assert psp.chordal_cliques(A) == [[0, 1], [2]]


def test_correlative_sparsity():
    x = psp.polyvar(4)
    f = x[0] * x[1] + x[1] * x[2] + x[2] * x[3]
    C, cliques = psp.correlative_sparsity(f)
    assert C[0, 1] == 1 and C[0, 2] == 0
    assert cliques == [[0, 1], [1, 2], [2, 3]]


# ----------------------------------------------------------------------
# scalar SOS constraints
# ----------------------------------------------------------------------
def test_sos_feasible():
    # classic SOS polynomial: 2x^4 + 2x^3 y - x^2 y^2 + 5 y^4
    x, y = psp.polyvar(2)
    p = 2 * x**4 + 2 * x**3 * y - x**2 * y**2 + 5 * y**4
    prob = psp.Problem(None, [p >> 0])
    prob.solve(**SOLVER_OPTS)
    assert prob.status in ("optimal", "optimal_inaccurate"), prob.status


def test_motzkin_not_sos():
    # Motzkin polynomial is nonnegative but not SOS
    x, y = psp.polyvar(2)
    p = x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1
    prob = psp.Problem(None, [psp.SOS(p, sparse=False)])
    prob.solve(**SOLVER_OPTS)
    assert prob.status not in ("optimal", "optimal_inaccurate"), prob.status


def test_unconstrained_minimum():
    # min x^4 - 3x^2 = -9/4 (univariate, so the SOS bound is exact)
    (x,) = psp.polyvar(1)
    res = psp.sos_lower_bound(x**4 - 3 * x**2, **SOLVER_OPTS)
    assert abs(res.value - (-2.25)) < TOL, res.value


def test_constrained_minimum():
    # min x on [-1, 1] = -1
    (x,) = psp.polyvar(1)
    res = psp.sos_lower_bound(x, ineqs=[1 - x**2], order=1, **SOLVER_OPTS)
    assert abs(res.value - (-1.0)) < TOL, res.value


def test_csp_chain():
    # f = sum (x_i x_{i+1} - 1)^2 has minimum 0; each square lives in a
    # clique of the correlative sparsity pattern, so the sparse bound is exact
    x = psp.polyvar(4)
    f = sum(
        ((x[i] * x[i + 1] - 1) ** 2 for i in range(3)),
        psp.Polynomial.zero(4),
    )
    res = psp.sos_lower_bound(f, order=2, **SOLVER_OPTS)
    assert res.cliques == [[0, 1], [1, 2], [2, 3]]
    assert abs(res.value) < TOL, res.value
    # sparse blocks must all be small
    assert max(res.block_sizes) <= 6


# ----------------------------------------------------------------------
# SOS matrix constraints
# ----------------------------------------------------------------------
def test_sos_matrix_feasible_dense():
    (x,) = psp.polyvar(1)
    P = psp.PolyMatrix([[1 + x**2, x], [x, 1]])
    prob = psp.Problem(None, [P >> 0])
    prob.solve(**SOLVER_OPTS)
    assert prob.status in ("optimal", "optimal_inaccurate"), prob.status


def test_sos_matrix_infeasible():
    (x,) = psp.polyvar(1)
    # [[x^2, 2x^2], [2x^2, x^2]] is not PSD anywhere except x=0
    P = psp.PolyMatrix([[x**2, 2 * x**2], [2 * x**2, x**2]])
    prob = psp.Problem(None, [psp.SOSMatrix(P, sparse=False)])
    prob.solve(**SOLVER_OPTS)
    assert prob.status not in ("optimal", "optimal_inaccurate"), prob.status


def test_sos_matrix_tridiagonal_sparse():
    # tridiagonal P with diag 1 + x^2 and off-diag x/2:
    # decomposes exactly into 2x2 clique SOS blocks
    (x,) = psp.polyvar(1)
    m = 5
    entries = [
        [psp.Polynomial.zero(1) for _ in range(m)] for _ in range(m)
    ]
    for i in range(m):
        entries[i][i] = 1 + x**2
    for i in range(m - 1):
        entries[i][i + 1] = entries[i + 1][i] = 0.5 * x
    con = psp.SOSMatrix(entries, nvars=1)
    prob = psp.Problem(None, [con])
    prob.solve(**SOLVER_OPTS)
    assert prob.status in ("optimal", "optimal_inaccurate"), prob.status
    assert con.cliques == [[i, i + 1] for i in range(m - 1)]
    assert max(con.block_sizes) <= 4  # 2 (clique) x 2 (basis {1, x})


def test_polymatrix_lmi_operators():
    (x,) = psp.polyvar(1)
    t = cp.Variable()
    F = psp.PolyMatrix([[1 + x**2, x], [x, 1]])
    con = F >> psp.eye(2, nvars=1) * t
    assert isinstance(con, psp.SOSMatrix)
    con2 = psp.eye(2, nvars=1) * t << F
    assert isinstance(con2, psp.SOSMatrix)


# ----------------------------------------------------------------------
# PMI optimization (Scherer-Hol certificates)
# ----------------------------------------------------------------------
def test_pmi_scalar_objective():
    # F 1x1 reduces to plain SOS bound: min x^4 - 3x^2 = -9/4
    (x,) = psp.polyvar(1)
    res = psp.pmi_optimize([[x**4 - 3 * x**2]], **SOLVER_OPTS)
    assert abs(res.value - (-2.25)) < TOL, res.value


def test_pmi_min_eig_scalar_constraint():
    # min lambda_min([[1 + x^2, x], [x, 1]]) over x in [-1, 1]
    # = (3 - sqrt(5)) / 2, attained at |x| = 1
    (x,) = psp.polyvar(1)
    F = psp.PolyMatrix([[1 + x**2, x], [x, 1]])
    res = psp.pmi_optimize(F, ineqs=[1 - x**2], order=2, **SOLVER_OPTS)
    expected = (3 - np.sqrt(5)) / 2
    assert abs(res.value - expected) < TOL, (res.value, expected)


def test_pmi_min_eig_matrix_constraint():
    # min x s.t. diag(1 - x^2, 1) >> 0  <=>  min x on [-1, 1]  =  -1
    (x,) = psp.polyvar(1)
    G = psp.PolyMatrix([[1 - x**2, psp.Polynomial.zero(1)], [psp.Polynomial.zero(1), 1]], nvars=1)
    res = psp.pmi_optimize([[x]], ineqs=[G], order=1, **SOLVER_OPTS)
    assert abs(res.value - (-1.0)) < TOL, res.value


def test_pmi_sparse_matches_dense():
    # arrow-patterned F: sparse (clique) relaxation should match the dense one
    x = psp.polyvar(2)
    z = psp.Polynomial.zero(2)
    F = psp.PolyMatrix(
        [
            [1 + x[0] ** 4, z, x[0] * x[1]],
            [z, 1 + x[1] ** 4, x[1]],
            [x[0] * x[1], x[1], 3.0 + z],
        ]
    )
    dense = psp.pmi_optimize(F, ineqs=[1 - x[0] ** 2 - x[1] ** 2], order=2, sparse=False, **SOLVER_OPTS)
    sparse = psp.pmi_optimize(F, ineqs=[1 - x[0] ** 2 - x[1] ** 2], order=2, sparse=True, **SOLVER_OPTS)
    assert len(sparse.cliques) == 2
    # clique decomposition is a restriction, so sparse bound <= dense bound
    assert sparse.value <= dense.value + TOL
    assert abs(sparse.value - dense.value) < 5e-2, (sparse.value, dense.value)


# ----------------------------------------------------------------------
# parametric coefficients (YALMIP-style SOS programming)
# ----------------------------------------------------------------------
def test_parametric_sos_matrix():
    # largest lam such that [[1 - lam*x^2, x],[x, 1]] is an SOS matrix:
    # need (1 - lam x^2) - x^2 >= 0 achievable ... certificate exists iff
    # det = 1 - (1 + lam) x^2 ... for the SOS-matrix cert, lam = -1 gives
    # [[1 + x^2, x],[x, 1]] feasible; lam > 0 makes the (1,1) entry
    # indefinite for large x. Just check monotone feasibility boundary
    # lam* in [0, 1]: at lam = 0, v' P v = (x v1 ... ) -- P = [[1, x],[x, 1]]
    # is not an SOS matrix (det < 0 for |x| > 1), so lam* <= 0... use a
    # quantitative case instead: max lam s.t. [[1 - lam + x^2, x],[x, 1]] >> 0.
    # v'Pv = (x v1 + v2)^2 + (1 - lam) v1^2, so lam* = 1.
    (x,) = psp.polyvar(1)
    lam = cp.Variable()
    P = psp.PolyMatrix([[x**2 + 1 - lam, x], [x, 1]])
    prob = psp.Problem(cp.Maximize(lam), [P >> 0])
    prob.solve(**SOLVER_OPTS)
    assert abs(prob.value - 1.0) < TOL, prob.value


def test_polynomial_evaluation_and_value():
    (x,) = psp.polyvar(1)
    t = cp.Variable()
    p = x**2 + x * t
    t.value = 3.0
    assert abs(p.value()(np.array([2.0])) - 10.0) < 1e-12


if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except Exception:
                failed += 1
                print("FAIL %s" % name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
