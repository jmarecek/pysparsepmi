"""Monomial basis generation and reduction for Gram (SOS) representations."""
from __future__ import annotations

__all__ = ["monomials", "gram_candidates", "reduce_bases"]


def monomials(nvars, maxdeg, mindeg=0, vars=None, var_maxdeg=None):
    """All exponent tuples over ``vars`` with total degree in [mindeg, maxdeg].

    ``var_maxdeg`` optionally caps the degree of each individual variable
    (a dict or sequence indexed by variable). Variables not in ``vars`` have
    exponent zero. The result is sorted (by total degree, then lexicographic).
    """
    if vars is None:
        vars = list(range(nvars))
    else:
        vars = sorted(vars)

    def cap(i):
        if var_maxdeg is None:
            return maxdeg
        try:
            return var_maxdeg[i]
        except (KeyError, IndexError):
            return maxdeg

    out = []
    expo = [0] * nvars

    def rec(pos, remaining):
        if pos == len(vars):
            if sum(expo) >= mindeg:
                out.append(tuple(expo))
            return
        v = vars[pos]
        top = min(remaining, cap(v))
        for e in range(top + 1):
            expo[v] = e
            rec(pos + 1, remaining - e)
        expo[v] = 0

    rec(0, maxdeg)
    out.sort(key=lambda t: (sum(t), t))
    return out


def gram_candidates(support, nvars, vars=None, degree_support=None):
    """Candidate Gram-basis monomials for an SOS representation.

    ``support`` is an iterable of exponent tuples of the polynomial (or, for
    matrix SOS, the union of entry supports used for degree bounds — pass the
    diagonal supports via ``degree_support`` to get sharper bounds).

    Uses the standard safe degree bounds: a monomial ``a`` may appear in some
    SOS decomposition only if ``mindeg/2 <= |a| <= maxdeg/2`` and
    ``a_i <= floor(deg_i/2)`` for every variable, because the extremal
    homogeneous parts of a sum of squares cannot cancel.
    """
    bound_support = list(degree_support if degree_support is not None else support)
    if not bound_support:
        return []
    degs = [sum(e) for e in bound_support]
    hi = max(degs) // 2
    lo = (min(degs) + 1) // 2
    var_max = [0] * nvars
    for e in bound_support:
        for i, ei in enumerate(e):
            if ei > var_max[i]:
                var_max[i] = ei
    var_max = [d // 2 for d in var_max]
    return monomials(nvars, hi, mindeg=lo, vars=vars, var_maxdeg=var_max)


def reduce_bases(bases, support):
    """Iteratively prune Gram-basis candidates (diagonal-consistency check).

    A candidate ``a`` in some basis can be removed when ``2a`` is not in the
    target ``support`` and no *distinct* pair ``b + c = 2a`` exists within a
    single basis: the Gram-matrix equation for the monomial ``2a`` then forces
    the corresponding nonnegative diagonal entries to vanish, so the whole
    row/column is zero. Works jointly across several bases (cliques): a
    diagonal contribution may come from any block, so pairs are searched in
    every basis. This mirrors the monomial-reduction step YALMIP performs when
    compiling SOS programs.
    """
    sets = [set(B) for B in bases]
    target = set(tuple(e) for e in support)
    changed = True
    while changed:
        changed = False
        for B in sets:
            for alpha in list(B):
                dbl = tuple(2 * a for a in alpha)
                if dbl in target:
                    continue
                found = False
                for B2 in sets:
                    for beta in B2:
                        gamma = tuple(d - b for d, b in zip(dbl, beta))
                        if gamma != beta and all(g >= 0 for g in gamma) and gamma in B2:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    B.discard(alpha)
                    changed = True
    return [sorted(B, key=lambda t: (sum(t), t)) for B in sets]
