"""State-lifting relaxations for polynomials with composition structure.

Implements the moment-SOS hierarchies of Balada Gaggioli, Henrion & Korda
for polynomials that are evaluated through a chain of low-dimensional maps

    s_1 = F_1(x_1),   s_i = F_i(s_{i-1}, x_i),   p(x) = F_n(s_{n-1}, x_n),

with states ``s_i`` of dimension ``r_i`` (the ranks / bond dimensions):

* ``method="chord"`` — the state-lifting chordal hierarchy SL-chord
  (arXiv:2604.17563, Section 4, eq. (22)): the states become lifted
  variables tied to ``x`` by equalities ``s_i - F_i(s_{i-1}, x_i) = 0``, and
  the correlative sparsity of the lifted problem is exploited. The chordal
  extension uses the stage-by-stage elimination order of Theorem 4.2 of that
  paper / Theorem 3.3 of arXiv:2512.08394, so PSD blocks involve at most
  ``max_i r_i + r_{i+1} + 1`` variables (``r + 2`` for rank-``r`` CP
  polynomials, the LRPOP hierarchy of arXiv:2512.08394), independently of
  the number of variables ``n``.

* ``method="push"`` — the state-lifting push-forward hierarchy SL-push
  (arXiv:2604.17563, Section 5, eq. (25)): polynomial potentials ``V_i(s_i)``
  with one local certificate per stage,
  ``V_i(F_i(s_{i-1}, x_i)) - V_{i-1}(s_{i-1}) in Q_i``, which telescope to a
  certificate for ``p - t``. Blocks involve only ``r_{i-1} + dim(x_i)``
  variables, at the price of higher degrees.

The last map may be matrix-valued, which extends both hierarchies to
polynomial matrix inequalities (a pysparsepmi extension; the papers treat
scalar objectives). The bounds remain valid, since the certificates
telescope to ``F(x) - t I`` in the quadratic module on the feasible set;
convergence of the matrix versions is not claimed.

:func:`cp_lower_bound` and :func:`tt_lower_bound` build the chain for
polynomials given in canonical polyadic (CP) or tensor-train (TT) format,
using TensorLy's factor/core conventions.

All relaxations produce valid bounds for any relaxation order. When a box is
given, redundant interval constraints on the states (computed by interval
arithmetic) are added, as the paper does for convergence; these can be
replaced by ``state_bounds``.
"""
from __future__ import annotations

import math
import numbers

import cvxpy as cp

from .basis import monomials
from .chordal import chordal_cliques
import numpy as np

from .pmi import (
    PMIResult,
    _half,
    _pmi_localize,
    _row_cliques,
    pmi_optimize,
    sos_lower_bound,
)
from .polynomial import Polynomial, _is_number, as_polynomial, compose, polyvar
from .sos import (
    SOS,
    SOSMatrix,
    PolyMatrix,
    Problem,
    free_poly_variable,
    sos_poly_variable,
)

__all__ = ["composition_lower_bound", "cp_lower_bound", "tt_lower_bound"]


# ----------------------------------------------------------------------
# chain discovery
# ----------------------------------------------------------------------
def _per_stage(value, n, name):
    """Broadcast a scalar-or-list option to one entry per stage."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError("%s must have one entry per stage (%d)" % (name, n))
        return list(value)
    return [value] * n


def _resolve_box(box, n):
    """Per-stage ``(lo, hi)`` intervals for the local variables, or Nones."""
    if box is None:
        return [None] * n
    if isinstance(box, numbers.Real):
        return [(-float(box), float(box))] * n
    if isinstance(box, tuple) and len(box) == 2 and all(
        isinstance(v, numbers.Real) for v in box
    ):
        return [(float(box[0]), float(box[1]))] * n
    box = list(box)
    if len(box) != n:
        raise ValueError("box must be a number, a (lo, hi) pair or one entry per stage")
    return [_resolve_box(b, 1)[0] for b in box]


def _local_constraints(spec, xs, i):
    """Evaluate a ``local_ineqs``/``local_eqs`` entry for stage ``i``."""
    if spec is None:
        return []
    fn = spec[i] if isinstance(spec, (list, tuple)) else spec
    if fn is None:
        return []
    out = fn(xs[0] if len(xs) == 1 else xs)
    if isinstance(out, Polynomial) or _is_number(out):
        out = [out]
    return list(out)


class _Stage:
    """One map ``F_i`` evaluated in its local space ``(s_{i-1}, x_i)``."""

    def __init__(self, r_in, xdim, outputs, ineqs, eqs, box, shape=None):
        self.r_in = r_in
        self.xdim = xdim
        self.nlocal = r_in + xdim
        self.outputs = outputs  # state components, or matrix entries row-major
        self.ineqs = ineqs
        self.eqs = eqs
        self.box = box
        self.shape = shape  # (m, m) for a matrix-valued last map, else None

    @property
    def degree(self):
        return max(p.degree() for p in self.outputs)

    def matrix(self, sign=1.0):
        """The matrix-valued output as an object array (times ``sign``)."""
        m = self.shape[0]
        arr = np.empty((m, m), dtype=object)
        for a in range(m):
            for b in range(m):
                arr[a, b] = self.outputs[a * m + b] * sign
        return arr


def _as_matrix(out):
    """Object array if ``out`` is matrix-like (PolyMatrix / 2-D), else None."""
    if isinstance(out, PolyMatrix):
        return out.arr
    if isinstance(out, np.ndarray) and out.ndim == 2:
        return out
    if isinstance(out, (list, tuple)) and out and isinstance(
        out[0], (list, tuple, np.ndarray)
    ):
        arr = np.empty((len(out), len(out[0])), dtype=object)
        for a, row in enumerate(out):
            for b, q in enumerate(row):
                arr[a, b] = q
        return arr
    return None


def _discover(maps, xdims, box, local_ineqs, local_eqs):
    """Evaluate every map on symbolic local variables."""
    maps = list(maps)
    n = len(maps)
    if n == 0:
        raise ValueError("maps must contain at least one stage")
    xdims = _per_stage(xdims, n, "xdims")
    boxes = _resolve_box(box, n)
    stages = []
    r_prev = 0
    for i, F in enumerate(maps):
        k = int(xdims[i])
        if k < 1:
            raise ValueError("every stage needs at least one local variable")
        L = r_prev + k
        v = polyvar(L)
        s, xs = v[:r_prev], v[r_prev:]
        out = F(s, xs[0] if k == 1 else xs)
        shape = None
        arr = _as_matrix(out) if i == n - 1 else None
        if arr is not None:
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("the last stage must return a square matrix")
            shape = arr.shape
            out = list(arr.flat)
        elif isinstance(out, Polynomial) or _is_number(out):
            out = [out]
        out = [as_polynomial(o, L) for o in out]
        if not out:
            raise ValueError("stage %d returned no state" % i)
        for o in out:
            if o.nvars != L or not o.is_numeric():
                raise ValueError(
                    "stage %d must return numeric polynomials in its (s, x) variables" % i
                )
        if i == n - 1 and shape is None and len(out) != 1:
            raise ValueError(
                "the last stage must return a single polynomial or a symmetric matrix"
            )
        ineqs = [as_polynomial(g, L) for g in _local_constraints(local_ineqs, xs, i)]
        if boxes[i] is not None:
            lo, hi = boxes[i]
            ineqs += [(hi - x) * (x - lo) for x in xs]
        eqs = [as_polynomial(h, L) for h in _local_constraints(local_eqs, xs, i)]
        stages.append(_Stage(r_prev, k, out, ineqs, eqs, boxes[i], shape))
        r_prev = len(out)
    return stages


# ----------------------------------------------------------------------
# interval bounds on the states
# ----------------------------------------------------------------------
def _ipow(iv, k):
    lo, hi = iv
    a, b = lo ** k, hi ** k
    if k % 2 == 1:
        return (a, b)
    if lo >= 0:
        return (a, b)
    if hi <= 0:
        return (b, a)
    return (0.0, max(a, b))


def _imul(u, v):
    prods = [u[0] * v[0], u[0] * v[1], u[1] * v[0], u[1] * v[1]]
    return (min(prods), max(prods))


def _interval(p, ivs):
    """Interval enclosure of a numeric polynomial over a box."""
    lo = hi = 0.0
    for e, c in p.terms.items():
        term = (1.0, 1.0)
        for j, ej in enumerate(e):
            if ej:
                term = _imul(term, _ipow(ivs[j], ej))
        a, b = c * term[0], c * term[1]
        lo += min(a, b)
        hi += max(a, b)
    return (lo, hi)


def _state_bounds(stages, state_bounds):
    """Per-stage lists of ``(lo, hi)`` for every state component (or None)."""
    n = len(stages)
    if state_bounds is None:
        return [None] * (n - 1)
    if isinstance(state_bounds, str):
        if state_bounds != "auto":
            raise ValueError("state_bounds must be 'auto', None or explicit bounds")
        if any(st.box is None for st in stages):
            return [None] * (n - 1)
        out = []
        prev = []
        for st in stages[:-1]:
            ivs = prev + [st.box] * st.xdim
            cur = []
            for p in st.outputs:
                lo, hi = _interval(p, ivs)
                pad = 1e-3 * (hi - lo) + 1e-6
                cur.append((lo - pad, hi + pad))
            out.append(cur)
            prev = cur
        return out
    state_bounds = list(state_bounds)
    if len(state_bounds) != n - 1:
        raise ValueError("state_bounds needs one entry per lifted state (n - 1)")
    out = []
    for i, b in enumerate(state_bounds):
        r = len(stages[i].outputs)
        if b is None:
            out.append(None)
        elif isinstance(b, numbers.Real):
            out.append([(-float(b), float(b))] * r)
        elif isinstance(b, tuple) and len(b) == 2 and all(
            isinstance(v, numbers.Real) for v in b
        ):
            out.append([(float(b[0]), float(b[1]))] * r)
        else:
            b = [tuple(map(float, iv)) for iv in b]
            if len(b) != r:
                raise ValueError("state_bounds[%d] needs %d intervals" % (i, r))
            out.append(b)
    return out


def _rescale(stages, bounds):
    """Rescale states and objective to magnitude ~1 using their bounds.

    Substitutes ``s_i = D_i * u_i`` with ``D_i`` the componentwise bound
    magnitudes, so that the lifted SDP has O(1) data (long chains otherwise
    produce states and constants that first-order solvers cannot handle).
    This is an exact reparametrization. Returns the scale of the objective.
    """
    n = len(stages)
    if n == 1 or any(b is None for b in bounds):
        return 1.0
    scales = [[max(abs(lo), abs(hi)) or 1.0 for lo, hi in b] for b in bounds]
    for i, st in enumerate(stages):
        v = polyvar(st.nlocal)
        subs = [v[a] * scales[i - 1][a] for a in range(st.r_in)] if i else []
        subs += v[st.r_in:]
        outs = [compose(F, subs) for F in st.outputs]
        if i < n - 1:
            outs = [F * (1.0 / c) for F, c in zip(outs, scales[i])]
        st.outputs = outs
    for i in range(n - 1):
        bounds[i] = [(lo / c, hi / c) for (lo, hi), c in zip(bounds[i], scales[i])]
    # objective scale from an interval bound over the (scaled) last stage
    last = stages[-1]
    ivs = (bounds[-1] if n > 1 else []) + [last.box] * last.xdim
    if last.box is None:
        return 1.0
    c = 0.0
    for q in last.outputs:
        lo, hi = _interval(q, ivs)
        c = max(c, abs(lo), abs(hi))
    c = c or 1.0
    last.outputs = [q * (1.0 / c) for q in last.outputs]
    return c


# ----------------------------------------------------------------------
# the two hierarchies
# ----------------------------------------------------------------------
def _chord(stages, bounds, sign, order, solver, verbose, solver_kwargs):
    n = len(stages)
    # global layout: x_1, s_1, x_2, s_2, ..., x_n
    idx_x, idx_s = [], []
    N = 0
    for i, st in enumerate(stages):
        idx_x.append(list(range(N, N + st.xdim)))
        N += st.xdim
        if i < n - 1:
            r = len(st.outputs)
            idx_s.append(list(range(N, N + r)))
            N += r

    def local_map(i):
        return (idx_s[i - 1] if i > 0 else []) + idx_x[i]

    ineqs, eqs = [], []
    for i, st in enumerate(stages):
        m = local_map(i)
        ineqs += [g.embed(N, m) for g in st.ineqs]
        eqs += [h.embed(N, m) for h in st.eqs]
        if i < n - 1:
            for l, F in enumerate(st.outputs):
                eqs.append(Polynomial.variable(N, idx_s[i][l]) - F.embed(N, m))
            if bounds[i] is not None:
                for l, (lo, hi) in enumerate(bounds[i]):
                    s = Polynomial.variable(N, idx_s[i][l])
                    ineqs.append((hi - s) * (s - lo))
    last_map = local_map(n - 1)
    if stages[-1].shape is None:
        objective = stages[-1].outputs[0].embed(N, last_map) * sign
        obj_entries = [objective]
    else:
        arr = stages[-1].matrix(sign)
        for idx in np.ndindex(arr.shape):
            arr[idx] = arr[idx].embed(N, last_map)
        objective = PolyMatrix(arr)
        obj_entries = list(arr.flat)

    groups = []
    for q in obj_entries:
        for e in q.terms:
            groups.append([j for j, ej in enumerate(e) if ej])
    for q in ineqs + eqs:
        groups.append(q.variables())
    edges = [(a, b) for g in groups for a in g for b in g if a < b]
    elim = []
    for i in reversed(range(n)):
        if i < n - 1:
            elim += idx_s[i]
        elim += idx_x[i]
    cliques = chordal_cliques(edges, n=N, order=elim)

    if stages[-1].shape is not None:
        return pmi_optimize(
            objective,
            ineqs=ineqs,
            eqs=eqs,
            order=order,
            var_cliques=cliques,
            multiplier_hosts="one",
            solver=solver,
            verbose=verbose,
            **solver_kwargs
        )
    return sos_lower_bound(
        objective,
        ineqs=ineqs,
        eqs=eqs,
        order=order,
        cliques=cliques,
        multiplier_hosts="one",
        solver=solver,
        verbose=verbose,
        **solver_kwargs
    )


def _push(stages, bounds, sign, order, solver, verbose, solver_kwargs):
    n = len(stages)
    d_min = 1
    for i, st in enumerate(stages):
        d_min = max(
            [d_min, _half(st.degree)]
            + [_half(g.degree()) for g in st.ineqs]
            + [_half(h.degree()) for h in st.eqs]
        )
    k = d_min if order is None else int(order)
    if k < d_min:
        raise ValueError("order=%d is below the minimum valid order %d" % (k, d_min))

    t = cp.Variable(name="t")
    V_prev = None
    cons, cliques, multipliers, eq_multipliers = [], [], [], []
    offset = 0  # same numbering as SL-chord: x_1, s_1, x_2, s_2, ..., x_n
    for i, st in enumerate(stages):
        L = st.nlocal
        cliques.append(list(range(offset - st.r_in, offset + st.xdim)))
        offset += st.xdim + (len(st.outputs) if i < n - 1 else 0)
        if i < n - 1:
            r = len(st.outputs)
            deg_V = (2 * k) // max(st.degree, 1)
            V, _ = free_poly_variable(r, monomials(r, deg_V))
            lhs = compose(V, st.outputs)
        else:
            V = None
            lhs = st.outputs[0] * sign
        prev = Polynomial.constant(L, t) if i == 0 else V_prev.embed(L, range(st.r_in))
        local = list(st.ineqs)
        if i > 0 and bounds[i - 1] is not None:
            svars = polyvar(L)[: st.r_in]
            local += [(hi - s) * (s - lo) for s, (lo, hi) in zip(svars, bounds[i - 1])]
        if st.shape is not None:
            # matrix-valued last map: F_n(s, x) - V_{n-1}(s) I is an SOS
            # matrix in the quadratic module (Scherer-Hol localizers)
            C = st.matrix(sign)
            rows = _row_cliques(PolyMatrix(C), True)
            for a in range(C.shape[0]):
                C[a, a] = C[a, a] - prev
            mult, eqm = _pmi_localize(
                C, [("scalar", g) for g in local], [g.degree() for g in local],
                st.eqs, k, rows, None, "all",
            )
            multipliers.append(mult)
            eq_multipliers.append(eqm)
            cons.append(SOSMatrix(PolyMatrix(C), cliques=rows))
            continue
        lhs = lhs - prev
        stage_mult = []
        for g in local:
            sg, Q = sos_poly_variable(L, monomials(L, k - _half(g.degree())))
            lhs = lhs - g * sg
            stage_mult.append(Q)
        multipliers.append(stage_mult)
        stage_eq = []
        for h in st.eqs:
            tau, c = free_poly_variable(L, monomials(L, 2 * k - h.degree()))
            lhs = lhs - tau * h
            stage_eq.append(c)
        eq_multipliers.append(stage_eq)
        cons.append(SOS(lhs, sparse=False))
        V_prev = V

    problem = Problem(cp.Maximize(t), cons)
    problem.solve(solver=solver, verbose=verbose, **solver_kwargs)
    return PMIResult(
        value=problem.value,
        status=problem.status,
        order=k,
        cliques=cliques,
        block_sizes=[b for c in cons for b in c.block_sizes],
        problem=problem,
        t=t,
        multipliers=multipliers,
        eq_multipliers=eq_multipliers,
    )


def composition_lower_bound(
    maps,
    box=None,
    order=None,
    method="chord",
    sense="min",
    xdims=1,
    local_ineqs=None,
    local_eqs=None,
    state_bounds="auto",
    scale=True,
    solver="SCS",
    verbose=False,
    **solver_kwargs
):
    """Bound ``p(x)`` given as a chain of low-dimensional polynomial maps.

    The polynomial is ``s_1 = F_1(x_1)``, ``s_i = F_i(s_{i-1}, x_i)``,
    ``p = F_n(s_{n-1}, x_n)`` (arXiv:2604.17563). Each map is a callable
    ``F(s, x)`` receiving the previous state ``s`` (a list of Polynomials,
    empty for the first stage) and the local variable ``x`` (a Polynomial,
    or a list when ``xdims[i] > 1``), and returning the new state as a list
    of numeric Polynomials in those variables. The last map returns a single
    polynomial, or a symmetric polynomial matrix (``PolyMatrix``, nested
    lists or a 2-D object array): the result then bounds
    ``min_x lambda_min(p(x))`` (``sense="max"``: ``max_x lambda_max``), with
    a Scherer–Hol matrix certificate on the last stage (SL-push: the PMI
    ``F_n(s, x) - V_{n-1}(s) I`` in the local quadratic module) or on the
    lifted variable cliques (SL-chord, via :func:`pmi_optimize` with
    ``var_cliques``). The ranks ``r_i`` are the lengths of the returned
    states.

    Parameters
    ----------
    maps : sequence of callables
        The stage maps ``F_1, ..., F_n``.
    box : float, (lo, hi) or list, optional
        Box ``lo <= x_i <= hi`` on every local variable (a number ``a``
        means ``[-a, a]``; a list gives one entry per stage). Also used to
        bound the states by interval arithmetic.
    order : int, optional
        Relaxation order (defaults to the smallest valid order).
    method : {"chord", "push"}
        SL-chord (lifted correlative sparsity; LRPOP for CP inputs) or
        SL-push (push-forward potentials).
    sense : {"min", "max"}
        ``"max"`` bounds ``max p`` from above; the returned ``value`` is then
        an upper bound.
    xdims : int or list of int
        Number of local variables per stage (default 1).
    local_ineqs, local_eqs : callable or list of callables, optional
        ``fn(x) -> list`` of polynomials ``>= 0`` / ``== 0`` in the local
        variables of a stage (one callable for all stages, or one per stage).
    state_bounds : "auto", None or list
        Redundant bounds on the states, one entry per lifted state
        ``s_1..s_{n-1}`` (a number, a ``(lo, hi)`` pair or one pair per
        component). ``"auto"`` derives them from ``box`` by interval
        arithmetic; ``None`` disables them.
    scale : bool
        Rescale every state (and the objective) by its bound magnitude before
        building the SDP (exact change of variables; requires state bounds).
        Strongly recommended for first-order solvers such as SCS.

    Returns a :class:`PMIResult`; ``ranks`` holds ``(r_1, ..., r_{n-1})``.
    For ``method="chord"``, variables are numbered ``x_1, s_1, x_2, s_2, ...``
    in ``cliques``; for ``method="push"`` each entry of ``cliques`` lists the
    variables ``(s_{i-1}, x_i)`` of one stage certificate in the same
    numbering.
    """
    if method not in ("chord", "push"):
        raise ValueError("method must be 'chord' or 'push'")
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    stages = _discover(maps, xdims, box, local_ineqs, local_eqs)
    bounds = _state_bounds(stages, state_bounds)
    obj_scale = _rescale(stages, bounds) if scale else 1.0
    sign = 1.0 if sense == "min" else -1.0
    run = _chord if method == "chord" else _push
    res = run(stages, bounds, sign, order, solver, verbose, solver_kwargs)
    if res.value is not None:
        res.value = sign * obj_scale * res.value
    res.ranks = [len(st.outputs) for st in stages[:-1]]
    return res


# ----------------------------------------------------------------------
# tensor-format front-ends
# ----------------------------------------------------------------------
def _binom(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def _univariate(x, coeffs, basis, interval):
    """``sum_j coeffs[j] * phi_j(x)`` in the monomial or Bernstein basis."""
    coeffs = [float(c) for c in coeffs]
    out = Polynomial.zero(x.nvars)
    if basis == "monomial":
        for j, c in enumerate(coeffs):
            if c != 0:
                out = out + (x ** j) * c
        return out
    if basis != "bernstein":
        raise ValueError("basis must be 'monomial' or 'bernstein'")
    if interval is None:
        raise ValueError("the Bernstein basis needs a finite box")
    lo, hi = interval
    s = (x - lo) * (1.0 / (hi - lo))
    d = len(coeffs) - 1
    for j, c in enumerate(coeffs):
        if c != 0:
            out = out + (s ** j) * ((1 - s) ** (d - j)) * (c * _binom(d, j))
    return out


def cp_lower_bound(factors, weights=None, box=1.0, basis="monomial", matrices=None, **kwargs):
    """Bound a low-rank polynomial ``sum_l w_l prod_i f_{l,i}(x_i)`` (LRPOP).

    ``factors[i]`` is an array of shape ``(deg_i + 1, r)`` whose column
    ``l`` holds the coefficients of ``f_{l,i}`` (TensorLy's CP factor layout
    for the coefficient tensor), in the monomial basis or, with
    ``basis="bernstein"``, in the Bernstein basis of the box interval.
    Lifting the partial products gives the LRPOP hierarchy of
    arXiv:2512.08394 with ``method="chord"`` (blocks on ``r + 2``
    variables); ``method="push"`` is also accepted. Remaining keyword
    arguments are passed to :func:`composition_lower_bound`.

    With ``matrices`` (a list of ``r`` symmetric ``m x m`` arrays ``A_l``),
    the objective is the polynomial matrix
    ``F(x) = sum_l A_l w_l prod_i f_{l,i}(x_i)`` and the result bounds
    ``min_x lambda_min(F(x))`` (``sense="max"``: ``max_x lambda_max``).
    """
    factors = [[list(row) for row in f] for f in factors]
    n = len(factors)
    r = len(factors[0][0])
    if any(len(row) != r for f in factors for row in f):
        raise ValueError("all factors must have the same number of columns (rank)")
    w = [1.0] * r if weights is None else [float(v) for v in weights]
    boxes = _resolve_box(box, n)
    if matrices is not None:
        A = [np.asarray(M, dtype=float) for M in matrices]
        if len(A) != r:
            raise ValueError("matrices needs one matrix per rank-one term (%d)" % r)
        m = A[0].shape[0]

    def column(i, l):
        return [row[l] for row in factors[i]]

    def stage(i):
        def F(s, x):
            f = [_univariate(x, column(i, l), basis, boxes[i]) for l in range(r)]
            prev = [Polynomial.constant(x.nvars, w[l]) for l in range(r)] if i == 0 else s
            new = [prev[l] * f[l] for l in range(r)]
            if i == n - 1:
                if matrices is not None:
                    return [
                        [
                            sum(
                                (new[l] * float(A[l][a][b]) for l in range(r)),
                                Polynomial.zero(x.nvars),
                            )
                            for b in range(m)
                        ]
                        for a in range(m)
                    ]
                total = Polynomial.zero(x.nvars)
                for q in new:
                    total = total + q
                return [total]
            return new

        return F

    return composition_lower_bound([stage(i) for i in range(n)], box=box, **kwargs)


def tt_lower_bound(cores, box=1.0, basis="monomial", **kwargs):
    """Bound a tensor-train polynomial ``P_1(x_1) P_2(x_2) ... P_n(x_n)``.

    ``cores[i]`` is an array of shape ``(r_{i-1}, deg_i + 1, r_i)`` with
    ``r_0 = r_n = 1`` (TensorLy's TT core layout for the coefficient
    tensor), so that ``P_i(x)[a, b] = sum_j cores[i][a, j, b] * phi_j(x)``
    with ``phi_j`` the monomial or Bernstein basis. Zero entries are skipped,
    so sparse cores give sparser lifted graphs. Keyword arguments are passed
    to :func:`composition_lower_bound` (``method="chord"`` or ``"push"``).
    """
    cores = [[[list(c) for c in a] for a in core] for core in cores]
    n = len(cores)
    if len(cores[0]) != 1 or len(cores[-1][0][0]) != 1:
        raise ValueError("boundary TT ranks must be 1")
    boxes = _resolve_box(box, n)

    def entry(i, a, b, x):
        return _univariate(x, [cores[i][a][j][b] for j in range(len(cores[i][a]))], basis, boxes[i])

    def stage(i):
        r_in, r_out = len(cores[i]), len(cores[i][0][0])

        def F(s, x):
            prev = [Polynomial.constant(x.nvars, 1.0)] if i == 0 else s
            if len(prev) != r_in:
                raise ValueError("TT ranks of cores %d and %d do not match" % (i - 1, i))
            out = []
            for b in range(r_out):
                q = Polynomial.zero(x.nvars)
                for a in range(r_in):
                    q = q + prev[a] * entry(i, a, b, x)
                out.append(q)
            return out

        return F

    return composition_lower_bound([stage(i) for i in range(n)], box=box, **kwargs)
