"""Chordal extensions, maximal cliques and correlative sparsity.

Python port of two MATLAB routines used by the ``sos.csp`` option in YALMIP
(as adapted in https://github.com/aeroimperial-optimization/sos-chordal-decomposition-pmi):

* ``cliquesFromSpMatD.m`` (from SparseCoLO): computes the maximal cliques of a
  chordal extension of a sparsity pattern. The MATLAB code uses a symmetric
  approximate-minimum-degree ordering followed by a sparse Cholesky
  factorization; here we perform the equivalent symbolic elimination with a
  greedy minimum-degree ordering, which yields the same class of chordal
  extensions.

* ``corrsparsity.m``: builds the correlative sparsity pattern of a set of
  monomials (two variables are adjacent iff they appear in the same monomial)
  and returns its maximal cliques.
"""
from __future__ import annotations

import numpy as np

from .polynomial import Polynomial

__all__ = ["chordal_cliques", "correlative_sparsity"]


def _adjacency(pattern, n=None):
    """Build an adjacency dict {vertex: set(neighbours)} from a pattern.

    ``pattern`` may be a square array-like, a scipy sparse matrix, or an
    iterable of ``(i, j)`` edges (in which case ``n`` is required).
    """
    if hasattr(pattern, "tocoo"):  # scipy sparse
        coo = pattern.tocoo()
        n = pattern.shape[0]
        adj = {i: set() for i in range(n)}
        for i, j in zip(coo.row, coo.col):
            if i != j:
                adj[int(i)].add(int(j))
                adj[int(j)].add(int(i))
        return adj
    arr = np.asarray(pattern)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0 and (
        arr.dtype != object
    ):
        n = arr.shape[0]
        adj = {i: set() for i in range(n)}
        rows, cols = np.nonzero(arr)
        for i, j in zip(rows, cols):
            if i != j:
                adj[int(i)].add(int(j))
                adj[int(j)].add(int(i))
        return adj
    # iterable of edges
    if n is None:
        raise ValueError("n is required when the pattern is given as an edge list")
    adj = {i: set() for i in range(n)}
    for i, j in pattern:
        if i != j:
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))
    return adj


def chordal_cliques(pattern, n=None, order=None):
    """Maximal cliques of a chordal extension of a sparsity pattern.

    The extension is produced by symbolic Gaussian elimination with a greedy
    minimum-degree ordering (ties broken by vertex index, so the result is
    deterministic). Returns a sorted list of cliques, each a sorted list of
    0-based vertex indices. Isolated vertices yield singleton cliques.

    ``order`` optionally fixes the elimination order: its vertices are
    eliminated first, in the given sequence, and any remaining vertices by
    minimum degree. Structured problems often have a known order that beats
    the greedy heuristic, e.g. the column-by-column order of Theorem 3.3 of
    arXiv:2512.08394, which attains the treewidth ``r + 1`` of lifted
    low-rank (CP) graphs where minimum degree does not.
    """
    adj = _adjacency(pattern, n)
    remaining = set(adj)
    forced = [] if order is None else [int(v) for v in order]
    if len(set(forced)) != len(forced) or not set(forced) <= remaining:
        raise ValueError("order must list distinct vertices of the pattern")
    forced.reverse()
    candidates = []
    while remaining:
        if forced:
            v = forced.pop()
        else:
            v = min(remaining, key=lambda u: (len(adj[u]), u))
        nb = adj[v]
        candidates.append(frozenset(nb | {v}))
        # eliminate v: connect its neighbourhood into a clique (fill-in)
        for a in nb:
            adj[a] |= nb
            adj[a].discard(a)
            adj[a].discard(v)
        remaining.remove(v)
    # keep only maximal cliques
    candidates.sort(key=len, reverse=True)
    kept = []
    for c in candidates:
        if not any(c <= k for k in kept):
            kept.append(c)
    cliques = sorted(sorted(c) for c in kept)
    return cliques


def correlative_sparsity(supports, nvars=None):
    """Correlative sparsity pattern and its cliques (port of ``corrsparsity.m``).

    ``supports`` may be a :class:`Polynomial`, an iterable of Polynomials, or
    an iterable of exponent tuples. Two variables are adjacent in the pattern
    iff they appear together in some monomial. Returns ``(C, cliques)`` where
    ``C`` is the 0/1 pattern matrix (with ones on the diagonal for variables
    that appear) and ``cliques`` are the maximal cliques of its chordal
    extension.
    """
    if isinstance(supports, Polynomial):
        nvars = supports.nvars
        exponents = list(supports.terms)
    else:
        supports = list(supports)
        if supports and isinstance(supports[0], Polynomial):
            nvars = supports[0].nvars
            exponents = []
            for p in supports:
                if p.nvars != nvars:
                    raise ValueError("polynomials have inconsistent nvars")
                exponents.extend(p.terms)
        else:
            if nvars is None:
                raise ValueError("nvars is required for raw exponent tuples")
            exponents = [tuple(e) for e in supports]

    C = np.zeros((nvars, nvars), dtype=int)
    for expo in exponents:
        idx = [i for i, e in enumerate(expo) if e > 0]
        for a in idx:
            C[a, a] = 1
            for b in idx:
                C[a, b] = 1
    cliques = chordal_cliques(C)
    return C, cliques
