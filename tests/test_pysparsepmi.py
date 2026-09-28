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
# term sparsity (TSSOS-style; arXiv:2411.15479)
# ----------------------------------------------------------------------
def test_sos_ts_unconstrained_exact():
    # min x^4 - 3x^2: the TS blocks split {1, x^2} | {x} and stay exact
    (x,) = psp.polyvar(1)
    res = psp.sos_lower_bound(x**4 - 3 * x**2, ts="block", **SOLVER_OPTS)
    assert abs(res.value - (-2.25)) < TOL, res.value
    assert max(res.block_sizes) <= 2, res.block_sizes


def test_sos_ts_blocks():
    x, y = psp.polyvar(2)
    f = 1 + x**4 + y**4 + x * y
    con = psp.SOS(f, ts="block")
    prob = psp.Problem(None, [con])
    prob.solve(**SOLVER_OPTS)
    assert prob.status in ("optimal", "optimal_inaccurate"), prob.status
    # dense Gram basis has 6 monomials; TS splits it into blocks
    assert max(con.block_sizes) <= 4, con.block_sizes
    assert sum(con.block_sizes) >= 6


def test_sos_matrix_ts_example32():
    # Example 3.2 of arXiv:2411.15479: 2x2 bivariate SOS matrix whose TSP
    # graph splits into small components
    x1, x2 = psp.polyvar(2)
    F = psp.PolyMatrix(
        [
            [1 + x1**2 + 2 * x1**2 * x2**2 + x2**2, x1 * x2],
            [x1 * x2, 2 + x1**2 * x2**2 + x2**4],
        ]
    )
    for method in ("block", "MD"):
        con = psp.SOSMatrix(F, ts=method)
        prob = psp.Problem(None, [con])
        prob.solve(**SOLVER_OPTS)
        assert prob.status in ("optimal", "optimal_inaccurate"), (method, prob.status)
        assert max(con.block_sizes) <= 4, (method, con.block_sizes)
    dense = psp.SOSMatrix(F, sparse=False)
    psp.Problem(None, [dense]).compile()
    assert max(dense.block_sizes) == 10


def test_pmi_ts_hierarchy():
    # TS bounds are monotone in the sparse order s and converge to the
    # dense bound at stabilization (Prop. 4.1 of arXiv:2411.15479)
    x1, x2 = psp.polyvar(2)
    F = psp.PolyMatrix([[1 + x1**2, x1], [x1, 1]])
    G = psp.PolyMatrix([[x1 * x2 * (-4.0) + 1, x1], [x1, 4 - x1**2 - x2**2]])
    ineqs = [G, 1 - x2**2]
    dense = psp.pmi_optimize(F, ineqs=ineqs, order=2, sparse=False, **SOLVER_OPTS)
    ts1 = psp.pmi_optimize(F, ineqs=ineqs, order=2, ts="MD", ts_order=1, **SOLVER_OPTS)
    ts_stab = psp.pmi_optimize(F, ineqs=ineqs, order=2, ts="MD", **SOLVER_OPTS)
    assert ts1.ts_order == 1
    assert ts1.value <= ts_stab.value + TOL
    assert ts_stab.value <= dense.value + TOL
    assert abs(ts_stab.value - dense.value) < TOL, (ts_stab.value, dense.value)
    # the first TS step must use smaller certificate blocks than dense
    assert max(ts1.block_sizes) < max(dense.block_sizes)
    assert ts_stab.ts_block_sizes is not None


def test_sos_lower_bound_ts_chain():
    x = psp.polyvar(4)
    f = sum(
        ((x[i] * x[i + 1] - 1) ** 2 for i in range(3)),
        psp.Polynomial.zero(4),
    )
    res = psp.sos_lower_bound(f, order=2, ts="block", **SOLVER_OPTS)
    assert abs(res.value) < TOL, res.value


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


# ----------------------------------------------------------------------
# equality constraints, elimination orders, lifting (LRPOP / SL-chord / SL-push)
# ----------------------------------------------------------------------
def _cp_graph_edges(r, n):
    # x_i -> i, t_{l,i} -> n + l*n + i; h_{l,i} couples t_{l,i}, t_{l,i-1}, x_i
    edges = []
    for l in range(r):
        edges.append((n + l * n, 0))
        for i in range(1, n):
            a, b, c = n + l * n + i, n + l * n + i - 1, i
            edges += [(a, b), (a, c), (b, c)]
    return edges, n + r * n


def test_chordal_cliques_elimination_order():
    r, n = 4, 12
    edges, N = _cp_graph_edges(r, n)
    greedy = max(len(c) for c in psp.chordal_cliques(edges, n=N))
    order = []
    for i in reversed(range(n)):
        order += [n + l * n + i for l in range(r)] + [i]
    cl = psp.chordal_cliques(edges, n=N, order=order)
    assert max(len(c) for c in cl) == r + 2  # Theorem 3.3 of arXiv:2512.08394
    assert greedy > r + 2


def test_embed_and_compose():
    x, y = psp.polyvar(2)
    p = psp.compose(x**2 * y + 3 * x, [x + y, x * y])
    q = (x + y) ** 2 * (x * y) + 3 * (x + y)
    assert p.equals(q)
    e = (x * y**2).embed(4, [3, 1])
    assert e.terms == {(0, 2, 0, 1): 1.0}


def test_sos_lower_bound_equality():
    x, y = psp.polyvar(2)
    res = psp.sos_lower_bound(x + y, eqs=[x**2 + y**2 - 1], **SOLVER_OPTS)
    assert abs(res.value + np.sqrt(2)) < TOL, res.value
    assert len(res.eq_multipliers) == 1


def _lrpop_example():
    # Example 3.1 of arXiv:2512.08394 (rank 2, n = 5), minimum -180 on [-1, 1]^5
    return [
        np.array([[1, -1], [2, 1]]),
        np.array([[-2, 0], [1, 2]]),
        np.array([[0, 1], [-1, 3]]),
        np.array([[3, 0], [1, -1]]),
        np.array([[2, 1], [-3, -1]]),
    ]


def test_lrpop_chord_example31():
    res = psp.cp_lower_bound(_lrpop_example(), box=1.0, order=2, method="chord", **SOLVER_OPTS)
    assert max(len(c) for c in res.cliques) == 4  # r + 2
    assert res.ranks == [2, 2, 2, 2]
    assert abs(res.value + 180) < 180 * TOL, res.value


def test_lrpop_push_example31():
    res = psp.cp_lower_bound(_lrpop_example(), box=1.0, order=2, method="push", **SOLVER_OPTS)
    assert max(len(c) for c in res.cliques) == 3  # r + 1
    assert abs(res.value + 180) < 180 * TOL, res.value


def _markov_maps(n):
    a = lambda x: 0.95 - 0.20 * x**2
    b = lambda x: 0.05 - 0.05 * x**2
    first = lambda s, x: [a(x), 1 - a(x)]
    step = lambda s, x: [s[0] * a(x) + s[1] * b(x), s[0] * (1 - a(x)) + s[1] * (1 - b(x))]
    last = lambda s, x: s[0] * a(x) + s[1] * b(x)
    return [first] + [step] * (n - 2) + [last]


def test_markov_chain_chord_exact():
    # Section 7.1 of arXiv:2604.17563: max Pr(working at n) = 1/2 + 0.9^n / 2
    n = 5
    res = psp.composition_lower_bound(
        _markov_maps(n), box=1.0, order=2, sense="max", **SOLVER_OPTS
    )
    assert abs(res.value - (0.5 + 0.5 * 0.9**n)) < TOL, res.value


def test_markov_chain_push_valid():
    n = 5
    res = psp.composition_lower_bound(
        _markov_maps(n), box=1.0, order=2, sense="max", method="push", **SOLVER_OPTS
    )
    exact = 0.5 + 0.5 * 0.9**n
    assert exact - TOL <= res.value <= exact + 0.05, res.value


def test_tt_matches_vertex_minimum():
    # multilinear TT polynomial: the box minimum is attained at a vertex
    import itertools

    rng = np.random.RandomState(1)
    ranks = [1, 2, 2, 1]
    cores = [rng.randn(ranks[i], 2, ranks[i + 1]) for i in range(3)]

    def val(v):
        row = np.ones((1, 1))
        for i in range(3):
            row = row @ (cores[i][:, 0, :] + cores[i][:, 1, :] * v[i])
        return row[0, 0]

    brute = min(val(v) for v in itertools.product([-1, 1], repeat=3))
    res = psp.tt_lower_bound(cores, box=1.0, order=2, **SOLVER_OPTS)
    assert abs(res.value - brute) < 10 * TOL * max(1.0, abs(brute)), (res.value, brute)


# ----------------------------------------------------------------------
# PMIs with equalities, correlative sparsity, matrix-valued lifting
# ----------------------------------------------------------------------
def test_pmi_equality():
    # eigenvalues of [[x, y], [y, -x]] are +-sqrt(x^2 + y^2) = +-1 on the circle
    x, y = psp.polyvar(2)
    res = psp.pmi_optimize([[x, y], [y, -x]], eqs=[x**2 + y**2 - 1], **SOLVER_OPTS)
    assert abs(res.value + 1.0) < TOL, res.value


def test_pmi_correlative_sparsity_matches_dense():
    n = 5
    v = psp.polyvar(n)
    Z = psp.Polynomial.zero(n)
    F = [
        [
            1 + v[i] ** 2 - v[i] if i == j
            else (v[min(i, j)] * v[max(i, j)] if abs(i - j) == 1 else Z)
            for j in range(n)
        ]
        for i in range(n)
    ]
    ineqs = [1 - vi**2 for vi in v]
    dense = psp.pmi_optimize(F, ineqs=ineqs, order=2, **SOLVER_OPTS)
    sparse = psp.pmi_optimize(F, ineqs=ineqs, order=2, cs=True, **SOLVER_OPTS)
    assert sparse.var_cliques == [[i, i + 1] for i in range(n - 1)]
    assert max(sparse.block_sizes) < max(dense.block_sizes)
    assert abs(sparse.value - dense.value) < TOL, (sparse.value, dense.value)


def test_matrix_cp_lifting():
    # F = diag(p_1, p_2) with rank-one multilinear p_l: lambda_min = min_l min p_l
    import itertools

    factors = [
        np.array([[1.0, -0.5], [0.5, 0.4]]),
        np.array([[0.2, 1.0], [-0.8, 0.3]]),
        np.array([[0.6, 0.5], [0.3, -0.5]]),
        np.array([[-0.4, 0.7], [0.5, 0.2]]),
    ]
    A = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]

    def p(l, v):
        return np.prod([factors[i][0, l] + factors[i][1, l] * v[i] for i in range(4)])

    brute = min(p(l, v) for l in range(2) for v in itertools.product([-1, 1], repeat=4))
    for method in ("chord", "push"):
        res = psp.cp_lower_bound(
            factors, box=1.0, order=2, matrices=A, method=method, **SOLVER_OPTS
        )
        assert res.value <= brute + TOL, (method, res.value, brute)
        assert abs(res.value - brute) < 5 * TOL, (method, res.value, brute)


def test_matrix_terminal_map_push():
    # worst-case gain of M(x1) M(x2) M(x3): symmetric states, PMI last stage
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    import matrix_product_gain as ex

    n = 3
    samp = ex.sampled_max(n, pts=21)
    res = psp.composition_lower_bound(
        [ex.first, ex.step, ex.last], box=1.0, order=2, method="push",
        sense="max", **SOLVER_OPTS
    )
    assert samp - TOL <= res.value <= samp + 0.02, (res.value, samp)

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
