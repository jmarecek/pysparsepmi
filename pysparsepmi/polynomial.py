"""Lightweight multivariate polynomials with numeric or CVXPY-expression coefficients.

This module provides just enough symbolic machinery to formulate sparse
sum-of-squares (SOS) programs: polynomials are dictionaries mapping exponent
tuples to coefficients, and a coefficient may be either a number or an affine
CVXPY expression (the analogue of a YALMIP ``sdpvar`` coefficient).
"""
from __future__ import annotations

import numbers

import numpy as np

__all__ = [
    "Polynomial",
    "polyvar",
    "as_polynomial",
    "poly_matrix",
    "ball_multiplier",
    "compose",
]


def _is_number(c):
    """True if ``c`` is a plain numeric coefficient (not a CVXPY expression)."""
    return isinstance(c, numbers.Number) or isinstance(c, np.generic)


def _is_scalar_expression(obj):
    """Duck-typed check for a scalar CVXPY expression."""
    mod = type(obj).__module__ or ""
    if not mod.startswith("cvxpy"):
        return False
    shape = getattr(obj, "shape", None)
    return shape == () or (shape is not None and int(np.prod(shape)) == 1)


class Polynomial:
    """A multivariate polynomial over a fixed number of variables.

    Terms are stored as ``{exponent_tuple: coefficient}``. Coefficients may be
    numbers or scalar CVXPY expressions; arithmetic keeps everything affine in
    the CVXPY decision variables (products of two parametric coefficients are
    rejected).
    """

    # Make numpy defer to our operators so that e.g. ``A * p`` with a numeric
    # ndarray ``A`` produces an object array of polynomials.
    __array_ufunc__ = None
    __array_priority__ = 1000.0

    __slots__ = ("nvars", "terms")

    def __init__(self, nvars, terms=None):
        self.nvars = int(nvars)
        self.terms = {}
        if terms:
            for expo, coef in terms.items():
                expo = tuple(int(e) for e in expo)
                if len(expo) != self.nvars:
                    raise ValueError(
                        "exponent %r does not match nvars=%d" % (expo, self.nvars)
                    )
                if any(e < 0 for e in expo):
                    raise ValueError("negative exponent in %r" % (expo,))
                if _is_number(coef) and coef == 0:
                    continue
                if expo in self.terms:
                    self.terms[expo] = self.terms[expo] + coef
                else:
                    self.terms[expo] = coef

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls, nvars):
        return cls(nvars)

    @classmethod
    def constant(cls, nvars, value):
        return cls(nvars, {(0,) * nvars: value})

    @classmethod
    def variable(cls, nvars, index):
        expo = [0] * nvars
        expo[index] = 1
        return cls(nvars, {tuple(expo): 1.0})

    # ------------------------------------------------------------------
    # basic queries
    # ------------------------------------------------------------------
    def support(self):
        """Set of exponent tuples with a (possibly parametric) coefficient."""
        return set(self.terms)

    def is_zero(self):
        return not self.terms

    def degree(self):
        """Maximum total degree (0 for the zero polynomial)."""
        return max((sum(e) for e in self.terms), default=0)

    def mindeg(self):
        """Minimum total degree over the support (0 for the zero polynomial)."""
        return min((sum(e) for e in self.terms), default=0)

    def var_degrees(self):
        """Per-variable maximum degree, as a list of length ``nvars``."""
        out = [0] * self.nvars
        for e in self.terms:
            for i, ei in enumerate(e):
                if ei > out[i]:
                    out[i] = ei
        return out

    def variables(self):
        """Indices of variables that actually appear."""
        used = set()
        for e in self.terms:
            for i, ei in enumerate(e):
                if ei > 0:
                    used.add(i)
        return sorted(used)

    def is_numeric(self):
        return all(_is_number(c) for c in self.terms.values())

    def embed(self, nvars, index_map):
        """Copy of ``self`` in a larger space of ``nvars`` variables.

        Variable ``i`` of ``self`` becomes variable ``index_map[i]``.
        """
        index_map = [int(j) for j in index_map]
        if len(index_map) != self.nvars:
            raise ValueError("index_map must have length %d" % self.nvars)
        terms = {}
        for e, c in self.terms.items():
            expo = [0] * nvars
            for i, ei in enumerate(e):
                if ei:
                    expo[index_map[i]] += ei
            terms[tuple(expo)] = c
        return Polynomial(nvars, terms)

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------
    def _coerce(self, other):
        if isinstance(other, Polynomial):
            if other.nvars != self.nvars:
                raise ValueError(
                    "cannot combine polynomials in %d and %d variables"
                    % (self.nvars, other.nvars)
                )
            return other
        if _is_number(other) or _is_scalar_expression(other):
            return Polynomial.constant(self.nvars, other)
        return None

    def _broadcast(self, other, op):
        out = np.empty(other.shape, dtype=object)
        for idx in np.ndindex(other.shape):
            out[idx] = op(other[idx])
        return out

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return self._broadcast(other, lambda b: self + b)
        other = self._coerce(other)
        if other is None:
            return NotImplemented
        terms = dict(self.terms)
        for e, c in other.terms.items():
            if e in terms:
                terms[e] = terms[e] + c
            else:
                terms[e] = c
        return Polynomial(self.nvars, terms)

    def __radd__(self, other):
        if isinstance(other, np.ndarray):
            return self._broadcast(other, lambda b: b + self)
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return self._broadcast(other, lambda b: self - b)
        other = self._coerce(other)
        if other is None:
            return NotImplemented
        return self.__add__(-other)

    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            return self._broadcast(other, lambda b: b - self)
        other = self._coerce(other)
        if other is None:
            return NotImplemented
        return other.__add__(-self)

    def __neg__(self):
        return Polynomial(self.nvars, {e: -1 * c for e, c in self.terms.items()})

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self._broadcast(other, lambda b: self * b)
        other = self._coerce(other)
        if other is None:
            return NotImplemented
        terms = {}
        for e1, c1 in self.terms.items():
            n1 = _is_number(c1)
            for e2, c2 in other.terms.items():
                if not n1 and not _is_number(c2):
                    raise ValueError(
                        "product of two parametric coefficients is not affine"
                    )
                e = tuple(a + b for a, b in zip(e1, e2))
                c = c1 * c2
                if e in terms:
                    terms[e] = terms[e] + c
                else:
                    terms[e] = c
        return Polynomial(self.nvars, terms)

    def __rmul__(self, other):
        if isinstance(other, np.ndarray):
            return self._broadcast(other, lambda b: b * self)
        return self.__mul__(other)

    def __truediv__(self, other):
        if _is_number(other):
            return self * (1.0 / other)
        return NotImplemented

    def __pow__(self, k):
        if not isinstance(k, numbers.Integral) or k < 0:
            raise ValueError("only nonnegative integer powers are supported")
        out = Polynomial.constant(self.nvars, 1.0)
        for _ in range(int(k)):
            out = out * self
        return out

    # ------------------------------------------------------------------
    # evaluation / post-solve helpers
    # ------------------------------------------------------------------
    def value(self):
        """Numeric copy, with CVXPY coefficients replaced by their values."""
        terms = {}
        for e, c in self.terms.items():
            if _is_number(c):
                terms[e] = c
            else:
                v = c.value
                if v is None:
                    raise ValueError("coefficient value not available (unsolved?)")
                terms[e] = float(np.asarray(v).reshape(()))
        return Polynomial(self.nvars, terms)

    def __call__(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (self.nvars,):
            raise ValueError("expected a point of length %d" % self.nvars)
        total = 0.0
        for e, c in self.value().terms.items():
            total += c * float(np.prod(point ** np.asarray(e)))
        return total

    # ------------------------------------------------------------------
    # constraint creation (CVXPY LMI style)
    # ------------------------------------------------------------------
    def __rshift__(self, other):
        """``p >> other``: p - other is a sum of squares (sparse by default)."""
        from .sos import SOS

        other = self._coerce(other)
        if other is None:
            return NotImplemented
        return SOS(self - other)

    def __lshift__(self, other):
        """``p << other``: other - p is a sum of squares (sparse by default)."""
        from .sos import SOS

        other = self._coerce(other)
        if other is None:
            return NotImplemented
        return SOS(other - self)

    def equals(self, other, tol=1e-9):
        """Numeric equality check (both polynomials must be numeric)."""
        other = self._coerce(other)
        diff = self - other
        return all(
            _is_number(c) and abs(c) <= tol for c in diff.terms.values()
        )

    # ------------------------------------------------------------------
    def __repr__(self):
        if not self.terms:
            return "0"
        parts = []
        for e in sorted(self.terms, key=lambda t: (sum(t), t)):
            c = self.terms[e]
            mono = "*".join(
                "x%d" % i if p == 1 else "x%d^%d" % (i, p)
                for i, p in enumerate(e)
                if p > 0
            )
            cs = repr(c) if not _is_number(c) else ("%g" % c)
            parts.append(cs if not mono else (cs + "*" + mono))
        return " + ".join(parts)


def polyvar(nvars):
    """Return the list ``[x0, ..., x_{nvars-1}]`` of coordinate polynomials."""
    return [Polynomial.variable(nvars, i) for i in range(nvars)]


def as_polynomial(obj, nvars=None):
    """Coerce ``obj`` (Polynomial, number, or scalar CVXPY expr) to Polynomial."""
    if isinstance(obj, Polynomial):
        return obj
    if nvars is None:
        raise ValueError("nvars is required to coerce %r" % (obj,))
    if _is_number(obj) or _is_scalar_expression(obj):
        return Polynomial.constant(nvars, obj)
    raise TypeError("cannot interpret %r as a polynomial" % (obj,))


def poly_matrix(M, nvars=None):
    """Coerce a matrix-like of polynomials/numbers to an object ndarray.

    ``nvars`` is inferred from the first Polynomial entry when omitted.
    """
    if isinstance(M, np.ndarray) and M.dtype == object:
        arr = M.copy()
    else:
        arr = np.empty(np.shape(M), dtype=object)
        for idx in np.ndindex(arr.shape):
            entry = M[idx[0]]
            for k in idx[1:]:
                entry = entry[k]
            arr[idx] = entry
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("expected a square matrix, got shape %r" % (arr.shape,))
    if nvars is None:
        for idx in np.ndindex(arr.shape):
            if isinstance(arr[idx], Polynomial):
                nvars = arr[idx].nvars
                break
        else:
            raise ValueError("cannot infer nvars: no Polynomial entries")
    for idx in np.ndindex(arr.shape):
        arr[idx] = as_polynomial(arr[idx], nvars)
    return arr


def compose(p, subs):
    """Substitute ``subs[i]`` for variable ``i`` of ``p``: ``p(subs[0], ...)``.

    ``subs`` are Polynomials sharing one variable space (numbers are
    accepted as constants). ``p`` may have CVXPY coefficients as long as the
    substituted polynomials are numeric, so ``compose(V, F)`` of a
    parametric potential ``V`` with a numeric map ``F`` stays affine.
    """
    subs = list(subs)
    if len(subs) != p.nvars:
        raise ValueError("expected %d substitutions, got %d" % (p.nvars, len(subs)))
    target = None
    for q in subs:
        if isinstance(q, Polynomial):
            target = q.nvars
            break
    if target is None:
        raise ValueError("cannot infer the target space: no Polynomial in subs")
    subs = [as_polynomial(q, target) for q in subs]
    powers = [[Polynomial.constant(target, 1.0)] for _ in subs]

    def power(i, k):
        while len(powers[i]) <= k:
            powers[i].append(powers[i][-1] * subs[i])
        return powers[i][k]

    out = Polynomial.zero(target)
    for e, c in p.terms.items():
        mono = Polynomial.constant(target, 1.0)
        for i, ei in enumerate(e):
            if ei:
                mono = mono * power(i, ei)
        out = out + mono * c
    return out


def ball_multiplier(nvars, nu, vars=None):
    """The polynomial ``(sum_i x_i^2)**nu`` used as a positive multiplier.

    This is the weight from Theorem 3.4 of Zheng & Fantuzzi (2020): a sparse
    polynomial matrix ``P`` that is strictly positive definite admits, for some
    ``nu``, a chordal decomposition of ``(x'x)^nu * P`` into clique SOS blocks.
    """
    if vars is None:
        vars = range(nvars)
    terms = {}
    for i in vars:
        e = [0] * nvars
        e[i] = 2
        terms[tuple(e)] = 1.0
    ball = Polynomial(nvars, terms)
    return ball ** nu
