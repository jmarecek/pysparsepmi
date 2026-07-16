"""Term-sparsity (TSSOS-style) block structures for SOS certificates.

Implements the iterative support-extension / chordal-extension procedure of
Miller, Wang & Guo, *Sparse Polynomial Matrix Optimization*
(arXiv:2411.15479): Definition 3.2 and eqs. (26)-(29) for a single PMI
``P(x) >> 0``. The constrained variant (Section 4 of the paper) lives in
:mod:`pysparsepmi.pmi`.

Nodes of the term sparsity pattern (TSP) graph are pairs ``(row, alpha)``:
a monomial ``alpha`` of the Gram basis attached to matrix row ``row``. Two
nodes are adjacent when their monomial product lies in the current support
of the corresponding matrix entry. The graph is grown by alternating
*support extension* and *chordal extension* — either the block closure
(``"block"``, connected components become complete blocks) or a greedy
minimum-degree chordal closure (``"MD"``, the "MinimumDegree" choice in
TSSOS) — and each maximal clique of the resulting graph indexes one PSD
Gram block.
"""
from __future__ import annotations

from .chordal import chordal_cliques

__all__ = ["graph_blocks", "ts_matrix_blocks"]


def _validate_ts(method):
    if method is True:
        return "block"
    if method in ("block", "MD"):
        return method
    raise ValueError(
        "ts must be 'block' or 'MD' (TSSOS's TS= choices), got %r" % (method,)
    )


def graph_blocks(n_nodes, edges, method):
    """Blocks of a chordal extension of the graph ``([n_nodes], edges)``.

    ``method="block"`` returns the connected components (the block closure:
    each component becomes one complete block); ``method="MD"`` returns the
    maximal cliques of a greedy minimum-degree chordal closure. Blocks are
    sorted lists of node indices; isolated nodes give singleton blocks.
    """
    method = _validate_ts(method)
    if method == "MD":
        return chordal_cliques(list(edges), n=n_nodes)
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comps = {}
    for v in range(n_nodes):
        comps.setdefault(find(v), []).append(v)
    return sorted(sorted(c) for c in comps.values())


def ts_matrix_blocks(entry_supports, bases, method="block", max_steps=None):
    """Term-sparsity Gram blocks for an SOS-matrix constraint.

    Runs the support-extension / chordal-extension iteration (eqs. (26)-(29)
    of arXiv:2411.15479) on the TSP graph of a symmetric polynomial matrix.

    Parameters
    ----------
    entry_supports : dict
        Maps ``(i, j)`` with ``i <= j`` to an iterable of exponent tuples,
        the support of entry ``P[i, j]``. Missing entries are treated as
        zero.
    bases : list of list of exponent tuples
        One Gram basis per matrix row (``v^j(x)`` in the paper).
    method : {"block", "MD"}
        Chordal extension used in each step: block closure or greedy
        minimum-degree chordal closure.
    max_steps : int, optional
        Number of extension steps (the sparse order ``s``); ``None``
        iterates until the edge set stabilizes.

    Returns
    -------
    blocks : list of list of ``(row, alpha)`` nodes
    steps : int
        Number of extension steps actually performed.
    """
    _validate_ts(method)
    m = len(bases)
    nodes = [(i, tuple(a)) for i in range(m) for a in bases[i]]
    nnode = len(nodes)

    # initial supports, eq. (26): supp(P_ij); the diagonal also gets the
    # squares of its basis monomials (always admissible Gram diagonal terms)
    C0 = {}
    for i in range(m):
        for j in range(i, m):
            supp = set(tuple(e) for e in entry_supports.get((i, j), ()))
            if i == j:
                supp |= {tuple(2 * a for a in alpha) for alpha in map(tuple, bases[i])}
            C0[(i, j)] = supp
    C = {k: set(v) for k, v in C0.items()}

    blocks = [[v] for v in range(nnode)]
    prev_edges = None
    steps = 0
    while True:
        # support extension, eq. (27)
        edges = set()
        for a in range(nnode):
            ia, alpha = nodes[a]
            for b in range(a + 1, nnode):
                ib, beta = nodes[b]
                key = (ia, ib) if ia <= ib else (ib, ia)
                gamma = tuple(x + y for x, y in zip(alpha, beta))
                if gamma in C[key]:
                    edges.add((a, b))
        if edges == prev_edges:
            break
        # chordal extension, eq. (28)
        blocks = graph_blocks(nnode, edges, method)
        prev_edges = edges
        steps += 1
        # support update, eq. (29) (kept monotone by including C0)
        C = {k: set(v) for k, v in C0.items()}
        for blk in blocks:
            for p in range(len(blk)):
                ia, alpha = nodes[blk[p]]
                for q in range(p, len(blk)):
                    ib, beta = nodes[blk[q]]
                    key = (ia, ib) if ia <= ib else (ib, ia)
                    C[key].add(tuple(x + y for x, y in zip(alpha, beta)))
        if max_steps is not None and steps >= max_steps:
            break
    return [[nodes[v] for v in blk] for blk in blocks], steps
