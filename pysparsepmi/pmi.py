"""High-level sparse relaxations for polynomial matrix inequalities (PMIs).

Provides the two entry points that TSSOS exposes for polynomial matrix
optimization (see ``example/pmi.jl`` in TSSOS):

* :func:`pmi_optimize` — a lower bound on ``inf_x lambda_min(F(x))`` subject
  to scalar and/or matrix polynomial inequality constraints, via a sparse
  Scherer–Hol sum-of-squares certificate with one block per maximal clique of
  the chordal extension of the sparsity pattern of ``F``.

* :func:`sos_lower_bound` — a lower bound on ``inf_x f(x)`` for a scalar
  polynomial with scalar constraints, exploiting correlative (variable)
  sparsity in the style of Waki et al. (2006) / YALMIP's ``sos.csp``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Any, List, Optional

import numpy as np
import cvxpy as cp

from .polynomial import Polynomial, _is_number, as_polynomial, poly_matrix
from .chordal import chordal_cliques, correlative_sparsity
from .basis import monomials
from .termsparsity import graph_blocks
from .sos import (
    PolyMatrix,
    Problem,
    SOS,
    SOSMatrix,
    _match_coefficients,
    free_poly_variable,
    sos_matrix_variable,
    sos_poly_variable,
)

__all__ = ["pmi_optimize", "sos_lower_bound", "PMIResult"]


def _half(d):
    """ceil(d / 2)."""
    return (int(d) + 1) // 2


@dataclass
class PMIResult:
    """Result of a sparse SOS relaxation.

    ``block_sizes`` are the PSD block sizes of the SOS certificate. With
    term sparsity, ``ts_order`` is the number of support-extension steps
    actually performed and ``ts_block_sizes[k]`` lists the Gram block sizes
    of constraint ``k`` (``k = 0`` is the certificate itself). For lifted
    (composition) relaxations, ``ranks`` are the state dimensions ``r_i``.
    For matrix problems with correlative sparsity in the variables,
    ``cliques`` are the row cliques and ``var_cliques`` the variable cliques.
    """

    value: Optional[float]
    status: str
    order: int
    cliques: List[List[int]]
    block_sizes: List[int]
    problem: Any = field(repr=False, default=None)
    t: Any = field(repr=False, default=None)
    multipliers: Any = field(repr=False, default=None)
    ts_order: Optional[int] = None
    ts_block_sizes: Optional[List[List[int]]] = None
    eq_multipliers: Any = field(repr=False, default=None)
    ranks: Optional[List[int]] = None
    var_cliques: Optional[List[List[int]]] = None


def _matrix_localizer(nvars, size, basis, G):
    """Localizing term for a matrix constraint ``G(x) >> 0`` (Scherer–Hol).

    Returns ``(L, Q)`` where ``L`` is a ``size x size`` object array of
    Polynomials with ``L[i, j] = trace(S^{(i,j)}(x) G(x))`` for an SOS matrix
    ``S`` of size ``size * q`` (``q`` the size of ``G``) parameterized by the
    PSD CVXPY variable ``Q``.
    """
    from collections import defaultdict

    basis = [tuple(b) for b in basis]
    nB = len(basis)
    q = G.shape[0]
    N = size * q * nB

    def idx(a, i, u):
        return (a * size + i) * q + u

    Q = cp.Variable((N, N), PSD=True)
    terms = [[defaultdict(list) for _ in range(size)] for _ in range(size)]
    for a, b in product(range(nB), repeat=2):
        base = tuple(x + y for x, y in zip(basis[a], basis[b]))
        for u in range(q):
            for v in range(q):
                for gexp, gc in G[u, v].terms.items():
                    if not _is_number(gc):
                        raise ValueError(
                            "matrix constraint data must have numeric coefficients"
                        )
                    gamma = tuple(x + y for x, y in zip(base, gexp))
                    for i in range(size):
                        for j in range(i, size):
                            terms[i][j][gamma].append(gc * Q[idx(a, i, u), idx(b, j, v)])
    L = np.empty((size, size), dtype=object)
    for i in range(size):
        for j in range(i, size):
            L[i, j] = Polynomial(
                nvars, {g: sum(lst) for g, lst in terms[i][j].items()}
            )
            L[j, i] = L[i, j]
    return L, Q


def _pmi_optimize_ts(F, norm_ineqs, d, ts, ts_order, solver, verbose, solver_kwargs):
    """Term-sparse Scherer-Hol relaxation for constrained PMO.

    Implements the sparse order-``s`` relaxation of Miller, Wang & Guo
    (arXiv:2411.15479, Section 4, eqs. (33)-(37)) on the SOS side: the Gram
    matrix of the certificate and of every localizing multiplier is indexed
    by nodes ``(alpha, i, u)`` (basis monomial, row of F, row of the
    constraint), the term sparsity graphs are grown jointly by support
    extension and chordal extension, and each clique of the stabilized
    graphs yields one PSD block.
    """
    p = F.shape[0]
    n = F.nvars

    # constraint matrices; k = 0 is the unit constraint carrying S_0
    Gs = [np.array([[Polynomial.constant(n, 1.0)]], dtype=object)]
    for kind, g in norm_ineqs:
        Gs.append(np.array([[g]], dtype=object) if kind == "scalar" else g.arr)
    for k, G in enumerate(Gs[1:], 1):
        q = G.shape[0]
        for u in range(q):
            for v in range(q):
                if any(not _is_number(c) for c in G[u, v].terms.values()):
                    raise ValueError(
                        "constraint data must have numeric coefficients"
                    )

    bases, nodes_k, gsupps = [], [], []
    for G in Gs:
        q = G.shape[0]
        dg = max(G[u, v].degree() for u in range(q) for v in range(q))
        B = monomials(n, d - _half(dg))
        bases.append(B)
        nodes_k.append(
            [(tuple(a), i, u) for i in range(p) for u in range(q) for a in B]
        )
        gsupps.append(
            {
                (u, v): [tuple(e) for e in G[u, v].terms]
                for u in range(q)
                for v in range(u, q)
            }
        )

    # initial activated supports, eq. (34): supp(F_ij), plus all even
    # monomials of degree <= 2d on the diagonal (this also covers t)
    C0 = {}
    even = {tuple(2 * a for a in alpha) for alpha in bases[0]}
    for i in range(p):
        for j in range(i, p):
            supp = set(F.arr[i, j].terms)
            if i == j:
                supp |= even
            C0[(i, j)] = supp
    C = {key: set(v) for key, v in C0.items()}

    def gsupp(k, u, v):
        return gsupps[k][(u, v) if u <= v else (v, u)]

    # joint support-extension / chordal-extension iteration, eqs. (35)-(36)
    blocks_k = [[[v] for v in range(len(nodes))] for nodes in nodes_k]
    prev_edges = None
    steps = 0
    while True:
        all_edges = []
        for k, nodes in enumerate(nodes_k):
            edges = set()
            for a in range(len(nodes)):
                alpha, ia, ua = nodes[a]
                for b in range(a + 1, len(nodes)):
                    beta, ib, ub = nodes[b]
                    key = (ia, ib) if ia <= ib else (ib, ia)
                    target = C[key]
                    base = tuple(x + y for x, y in zip(alpha, beta))
                    if any(
                        tuple(x + y for x, y in zip(base, g)) in target
                        for g in gsupp(k, ua, ub)
                    ):
                        edges.add((a, b))
            all_edges.append(edges)
        if all_edges == prev_edges:
            break
        blocks_k = [
            graph_blocks(len(nodes), edges, ts)
            for nodes, edges in zip(nodes_k, all_edges)
        ]
        prev_edges = all_edges
        steps += 1
        C = {key: set(v) for key, v in C0.items()}
        for k, (nodes, blocks) in enumerate(zip(nodes_k, blocks_k)):
            for blk in blocks:
                for a_pos in range(len(blk)):
                    alpha, ia, ua = nodes[blk[a_pos]]
                    for b_pos in range(a_pos, len(blk)):
                        beta, ib, ub = nodes[blk[b_pos]]
                        key = (ia, ib) if ia <= ib else (ib, ia)
                        base = tuple(x + y for x, y in zip(alpha, beta))
                        for g in gsupp(k, ua, ub):
                            C[key].add(tuple(x + y for x, y in zip(base, g)))
        if ts_order is not None and steps >= ts_order:
            break

    # assemble the SDP: F - t*I = S_0 + sum_k <S_k, G_k>_p with one PSD
    # Gram block per term-sparsity clique
    t = cp.Variable(name="t")
    lhs = {}
    grams_k = []
    for k, (nodes, blocks, G) in enumerate(zip(nodes_k, blocks_k, Gs)):
        grams = []
        for blk in blocks:
            nblk = len(blk)
            Q = cp.Variable((nblk, nblk), PSD=True)
            grams.append(Q)
            for a_pos in range(nblk):
                alpha, ia, ua = nodes[blk[a_pos]]
                for b_pos in range(nblk):
                    beta, ib, ub = nodes[blk[b_pos]]
                    if ia > ib:
                        continue
                    dest = lhs.setdefault((ia, ib), defaultdict(list))
                    base = tuple(x + y for x, y in zip(alpha, beta))
                    for gexp, gc in G[ua, ub].terms.items():
                        gamma = tuple(x + y for x, y in zip(base, gexp))
                        dest[gamma].append(gc * Q[a_pos, b_pos])
        grams_k.append(grams)

    constraints = []
    for i in range(p):
        for j in range(i, p):
            rhs = F.arr[i, j]
            if i == j:
                rhs = rhs - Polynomial.constant(n, t)
            if (i, j) not in lhs and rhs.is_zero():
                continue
            constraints.extend(
                _match_coefficients(
                    lhs.get((i, j), {}),
                    rhs.terms,
                    "PMI certificate entry (%d, %d)" % (i, j),
                )
            )

    problem = Problem(cp.Maximize(t), constraints)
    problem.solve(solver=solver, verbose=verbose, **solver_kwargs)
    return PMIResult(
        value=problem.value,
        status=problem.status,
        order=d,
        cliques=[list(range(p))],
        block_sizes=[len(b) for b in blocks_k[0]],
        problem=problem,
        t=t,
        multipliers=grams_k[1:],
        ts_order=steps,
        ts_block_sizes=[[len(b) for b in blocks] for blocks in blocks_k],
    )


def _hosts(cliques, vars_, mode):
    """Cliques that receive a multiplier for a constraint on ``vars_``."""
    hosts = [c for c in cliques if vars_ <= set(c)]
    if not hosts:
        return [sorted(set().union(*map(set, cliques)))]
    if mode == "one":
        return [min(hosts, key=len)]
    return hosts


def pmi_optimize(
    F,
    ineqs=(),
    order=None,
    sparse=True,
    nu=0,
    nvars=None,
    eqs=(),
    cs=False,
    var_cliques=None,
    multiplier_hosts="all",
    ts=None,
    ts_order=None,
    solver="SCS",
    verbose=False,
    **solver_kwargs
):
    """Sparse SOS lower bound on ``inf_x lambda_min(F(x))`` over a set.

    Solves ``max t`` such that ``F(x) - t*I`` lies in the (sparse) quadratic
    module generated by the constraints::

        F - t*I = S_0 + sum_j g_j * S_j + sum_j <S_j, G_j>

    where each ``S`` is an SOS matrix and, with ``sparse=True``, every term is
    decomposed over the maximal cliques of the chordal extension of the
    sparsity pattern of ``F`` (Zheng & Fantuzzi 2020; the clique-level
    analogue of TSSOS's PMI relaxations).

    Parameters
    ----------
    F : PolyMatrix or matrix-like of Polynomial
        Symmetric polynomial matrix objective.
    ineqs : sequence
        Constraints ``>= 0``: scalar Polynomials and/or polynomial matrices
        (PolyMatrix / matrix-like) with numeric coefficients.
    order : int, optional
        Relaxation order ``d`` (certificate degree ``2d``). Defaults to the
        smallest valid order.
    nu : int
        Power of the ``(x'x)^nu`` multiplier applied to the certificate
        matrix before chordal decomposition (Theorem 3.4 of the paper).
        Not supported together with ``ts``.
    eqs : sequence of Polynomial
        Scalar equality constraints ``h = 0``, entering through free
        symmetric polynomial-matrix multipliers ``T_h`` (``- h * T_h``).
    cs : bool
        Also exploit correlative sparsity in the variables: every SOS matrix
        and multiplier is split over the maximal cliques of the chordally
        extended correlative sparsity graph of ``F`` and the constraints
        (the variables of each monomial of ``F``, and of each constraint,
        are pairwise adjacent). Combines with the row-clique decomposition.
    var_cliques : list of lists, optional
        Explicit variable cliques (implies ``cs``); every constraint should
        be supported on one of them.
    multiplier_hosts : {"all", "one"}
        With variable cliques, give each constraint one multiplier per
        containing clique, or a single one on the smallest such clique.
    ts : {"block", "MD"}, optional
        Exploit term sparsity (Miller, Wang & Guo, arXiv:2411.15479,
        Section 4; TSSOS's ``TS=`` option). The Gram matrices of the
        certificate and of every localizing multiplier are restricted to
        blocks given by the cliques of the term sparsity pattern graphs,
        grown by joint support extension plus block closure (``"block"``)
        or minimum-degree chordal closure (``"MD"``). When set, ``ts``
        replaces the clique decomposition of ``sparse`` (the term sparsity
        pattern already encodes the zero pattern of ``F``).
    ts_order : int, optional
        Sparse order ``s``: the number of support-extension steps.
        ``ts_order=1`` is the first step of the TS hierarchy (TSSOS's
        ``tssos(F, G, x, d, TS=...)``); larger values mirror repeated
        ``tssos(data, TS=...)`` calls. ``None`` (default) iterates until
        the block structure stabilizes.
    """
    F = F if isinstance(F, PolyMatrix) else PolyMatrix(F, nvars)
    m = F.shape[0]
    n = F.nvars

    norm_ineqs = []
    for g in ineqs:
        if isinstance(g, Polynomial) or _is_number(g):
            norm_ineqs.append(("scalar", as_polynomial(g, n)))
        else:
            G = g if isinstance(g, PolyMatrix) else PolyMatrix(g, n)
            if G.shape == (1, 1):
                norm_ineqs.append(("scalar", G[0, 0]))
            else:
                norm_ineqs.append(("matrix", G))

    degF = max(F.arr[i, j].degree() for i in range(m) for j in range(m))
    deg_ineqs = [
        g.degree()
        if kind == "scalar"
        else max(g.arr[i, j].degree() for i in range(g.shape[0]) for j in range(g.shape[0]))
        for kind, g in norm_ineqs
    ]
    eqs = [as_polynomial(h, n) for h in eqs]
    d_min = max(
        [_half(degF)]
        + [_half(dg) for dg in deg_ineqs]
        + [_half(h.degree()) for h in eqs]
        + [1]
    )
    d = d_min if order is None else int(order)
    if d < d_min:
        raise ValueError("order=%d is below the minimum valid order %d" % (d, d_min))
    if multiplier_hosts not in ("all", "one"):
        raise ValueError("multiplier_hosts must be 'all' or 'one'")

    if ts is not None:
        if nu:
            raise ValueError("nu is not supported together with ts")
        if eqs or cs or var_cliques is not None:
            raise NotImplementedError(
                "eqs, cs and var_cliques are not supported together with ts"
            )
        return _pmi_optimize_ts(
            F, norm_ineqs, d, ts, ts_order, solver, verbose, solver_kwargs
        )

    t = cp.Variable(name="t")

    # certificate matrix C = F - t*I - localizing terms
    C = np.array(
        [[F.arr[i, j] for j in range(m)] for i in range(m)], dtype=object
    )
    for i in range(m):
        C[i, i] = C[i, i] - Polynomial.constant(n, t)

    if var_cliques is None and cs:
        var_cliques = _pmi_var_cliques(F, norm_ineqs, eqs)
    elif var_cliques is not None:
        var_cliques = [sorted(int(v) for v in c) for c in var_cliques]
    cliques = _row_cliques(F, sparse)
    multipliers, eq_multipliers = _pmi_localize(
        C, norm_ineqs, deg_ineqs, eqs, d, cliques, var_cliques, multiplier_hosts
    )

    cert = SOSMatrix(PolyMatrix(C), cliques=cliques, nu=nu, var_cliques=var_cliques)
    problem = Problem(cp.Maximize(t), [cert])
    problem.solve(solver=solver, verbose=verbose, **solver_kwargs)
    return PMIResult(
        value=problem.value,
        status=problem.status,
        order=d,
        cliques=cliques,
        block_sizes=cert.block_sizes,
        problem=problem,
        t=t,
        multipliers=multipliers,
        eq_multipliers=eq_multipliers,
        var_cliques=var_cliques,
    )


def _row_cliques(F, sparse):
    """Maximal cliques of the chordally extended pattern of ``F`` (plus I)."""
    m = F.shape[0]
    if not sparse:
        return [list(range(m))]
    pattern = np.eye(m, dtype=int)
    for i in range(m):
        for j in range(m):
            if not F.arr[i, j].is_zero():
                pattern[i, j] = 1
    return chordal_cliques(pattern)


def _pmi_var_cliques(F, norm_ineqs, eqs, order=None):
    """Correlative sparsity cliques of a PMI problem in the variables."""
    n = F.nvars
    groups = []
    for idx in np.ndindex(F.shape):
        for e in F.arr[idx].terms:
            groups.append([i for i, ei in enumerate(e) if ei])
    for kind, g in norm_ineqs:
        if kind == "scalar":
            groups.append(g.variables())
        else:
            groups.append(sorted(set().union(*(set(q.variables()) for q in g.arr.flat))))
    for h in eqs:
        groups.append(h.variables())
    edges = [(a, b) for grp in groups for a in grp for b in grp if a < b]
    return chordal_cliques(edges, n=n, order=order)


def _pmi_localize(C, norm_ineqs, deg_ineqs, eqs, d, cliques, var_cliques, hosts_mode):
    """Subtract Scherer–Hol localizing terms and equality multipliers from ``C``.

    ``C`` (an ``m x m`` object array of Polynomials) is modified in place:
    for every row clique ``C_k`` and every host variable clique, scalar
    constraints get ``g * S`` (``S`` an SOS matrix), matrix constraints
    ``<S, G>`` and equalities ``h * T`` (``T`` a free symmetric polynomial
    matrix). Returns the Gram variables of the inequality multipliers and
    the coefficient vectors of the equality multipliers.
    """
    n = C[0, 0].nvars
    all_vars = list(range(n))

    def hosts(vars_):
        if var_cliques is None:
            return [None]
        return _hosts(var_cliques, set(vars_), hosts_mode)

    def subtract(Ck, S):
        mk = len(Ck)
        for ii in range(mk):
            for jj in range(mk):
                C[Ck[ii], Ck[jj]] = C[Ck[ii], Ck[jj]] - S[ii, jj]

    multipliers = []
    for (kind, g), dg in zip(norm_ineqs, deg_ineqs):
        d_loc = d - _half(dg)
        if kind == "scalar":
            gvars = g.variables()
        else:
            gvars = sorted(set().union(*(set(q.variables()) for q in g.arr.flat)))
        per_clique = []
        for V in hosts(gvars):
            basis = monomials(n, d_loc, vars=V if V is not None else all_vars)
            for C_k in cliques:
                mk = len(C_k)
                if kind == "scalar":
                    S, Q = sos_matrix_variable(n, mk, basis)
                    subtract(C_k, g * S)
                else:
                    S, Q = _matrix_localizer(n, mk, basis, g.arr)
                    subtract(C_k, S)
                per_clique.append(Q)
        multipliers.append(per_clique)

    eq_multipliers = []
    for h in eqs:
        per_clique = []
        for V in hosts(h.variables()):
            basis = monomials(n, 2 * d - h.degree(), vars=V if V is not None else all_vars)
            for C_k in cliques:
                mk = len(C_k)
                T = np.empty((mk, mk), dtype=object)
                coeffs = []
                for ii in range(mk):
                    for jj in range(ii, mk):
                        T[ii, jj], c = free_poly_variable(n, basis)
                        T[jj, ii] = T[ii, jj]
                        coeffs.append(c)
                subtract(C_k, T * h)
                per_clique.append(coeffs)
        eq_multipliers.append(per_clique)
    return multipliers, eq_multipliers


def sos_lower_bound(
    f,
    ineqs=(),
    order=None,
    sparse=True,
    eqs=(),
    cliques=None,
    multiplier_hosts="all",
    ts=None,
    ts_order=None,
    solver="SCS",
    verbose=False,
    **solver_kwargs
):
    """Sparse SOS lower bound on ``inf_x f(x)`` with scalar constraints.

    Solves ``max t`` such that ``f - t = s_0 + sum_j g_j * s_{j,k}`` with SOS
    multipliers. With ``sparse=True`` the correlative sparsity pattern of the
    variables (from ``f`` and the ``g_j``) is chordally extended; the Gram
    basis of ``s_0`` is split over its maximal cliques (YALMIP ``sos.csp``)
    and each constraint ``g_j`` receives one multiplier per clique containing
    all its variables (Waki et al. 2006).

    Equality constraints ``h = 0`` in ``eqs`` enter through free polynomial
    multipliers ``tau_{h,k}`` of degree ``2*order - deg(h)`` in the variables
    of every clique ``k`` that contains all variables of ``h``, i.e.
    ``f - t = s_0 + sum_j g_j * s_{j,k} + sum_h tau_{h,k} * h``. ``cliques``
    overrides the automatic correlative-sparsity cliques (every constraint
    should then be supported on at least one of them).
    ``multiplier_hosts="one"`` gives each constraint a single multiplier on
    the smallest clique containing it (as in the SL-chord/LRPOP
    hierarchies) instead of one per containing clique (``"all"``), which
    avoids redundant multipliers when many cliques overlap.

    With ``ts`` set ("block" or "MD"), the scalar term-sparsity (TSSOS)
    relaxation is used instead: Gram matrices of ``s_0`` and of every
    multiplier are split into blocks along the cliques of the term sparsity
    pattern graphs (see :func:`pmi_optimize`; ``ts_order`` is the sparse
    order ``s``).
    """
    if not isinstance(f, Polynomial):
        raise TypeError("f must be a Polynomial")
    n = f.nvars
    ineqs = [as_polynomial(g, n) for g in ineqs]
    eqs = [as_polynomial(h, n) for h in eqs]

    if multiplier_hosts not in ("all", "one"):
        raise ValueError("multiplier_hosts must be 'all' or 'one'")
    if ts is not None:
        if eqs or cliques is not None:
            raise NotImplementedError("eqs and cliques are not supported together with ts")
        d_min = max([_half(f.degree())] + [_half(g.degree()) for g in ineqs] + [1])
        d = d_min if order is None else int(order)
        if d < d_min:
            raise ValueError(
                "order=%d is below the minimum valid order %d" % (d, d_min)
            )
        return _pmi_optimize_ts(
            PolyMatrix([[f]], nvars=n),
            [("scalar", g) for g in ineqs],
            d,
            ts,
            ts_order,
            solver,
            verbose,
            solver_kwargs,
        )

    d_min = max(
        [_half(f.degree())]
        + [_half(g.degree()) for g in ineqs]
        + [_half(h.degree()) for h in eqs]
        + [1]
    )
    d = d_min if order is None else int(order)
    if d < d_min:
        raise ValueError("order=%d is below the minimum valid order %d" % (d, d_min))

    if cliques is not None:
        cliques = [sorted(int(v) for v in c) for c in cliques]
        sparse = True
    elif sparse:
        # correlative sparsity from f, plus a clique per constraint
        exponents = list(f.terms)
        for g in list(ineqs) + list(eqs):
            e = [0] * n
            for i in g.variables():
                e[i] = 1
            exponents.append(tuple(e))
        _, cliques = correlative_sparsity(exponents, n)
    else:
        cliques = [list(range(n))]

    t = cp.Variable(name="t")
    cert = f - Polynomial.constant(n, t)

    multipliers = []
    for g in ineqs:
        gv = set(g.variables())
        hosts = _hosts(cliques, gv, multiplier_hosts)
        d_loc = d - _half(g.degree())
        per_clique = []
        for c in hosts:
            basis = monomials(n, d_loc, vars=c)
            s, Q = sos_poly_variable(n, basis)
            cert = cert - g * s
            per_clique.append(Q)
        multipliers.append(per_clique)

    eq_multipliers = []
    for h in eqs:
        hv = set(h.variables())
        hosts = _hosts(cliques, hv, multiplier_hosts)
        deg_tau = 2 * d - h.degree()
        per_clique = []
        for c in hosts:
            tau, coeffs = free_poly_variable(n, monomials(n, deg_tau, vars=c))
            cert = cert - tau * h
            per_clique.append(coeffs)
        eq_multipliers.append(per_clique)

    con = SOS(cert, cliques=cliques if sparse else None, sparse=sparse)
    problem = Problem(cp.Maximize(t), [con])
    problem.solve(solver=solver, verbose=verbose, **solver_kwargs)
    return PMIResult(
        value=problem.value,
        status=problem.status,
        order=d,
        cliques=cliques,
        block_sizes=con.block_sizes,
        problem=problem,
        t=t,
        multipliers=multipliers,
        eq_multipliers=eq_multipliers,
    )
