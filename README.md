# pysparsepmi

Sparse sum-of-squares (SOS) relaxations for **polynomial matrix inequalities
(PMIs)** in Python, exploiting chordal (correlative/matrix) sparsity, term
sparsity, and composition / low-rank structure via state lifting.

The PMI methods implemented here are those of Y. Zheng and G. Fantuzzi and,
independently, of J. Miller, J. Wang and F. Guo:

- Y. Zheng, G. Fantuzzi, *Sum-of-squares chordal decomposition of polynomial
  matrix inequalities* ([arXiv:2007.11410](https://arxiv.org/pdf/2007.11410)),
- J. Miller, J. Wang, F. Guo, *Sparse Polynomial Matrix Optimization*
  ([arXiv:2411.15479](https://arxiv.org/abs/2411.15479)).

In addition, we implement the state-lifting methods that L. Balada Gaggioli,
D. Henrion and M. Korda proposed for polynomial optimization problems (POPs),
and extend them to matrix-valued objectives:

- L. Balada Gaggioli, D. Henrion, M. Korda, *Global optimization of
  low-rank polynomials* ([arXiv:2512.08394](https://arxiv.org/abs/2512.08394)),
- L. Balada Gaggioli, D. Henrion, M. Korda, *Composition and tensor train
  structure in polynomial optimization*
  ([arXiv:2604.17563](https://arxiv.org/abs/2604.17563)),

with implementation reference to the accompanying
[sos-chordal-decomposition-pmi](https://github.com/aeroimperial-optimization/sos-chordal-decomposition-pmi)
code, [TSSOS](https://github.com/wangjie212/TSSOS), and the Julia packages
[LRPOP](https://github.com/llorebaga/LRPOP) and
[SLPOP](https://github.com/llorebaga/SLPOP).

The modelling layer is deliberately close to
[CVXPY's LMI API](https://www.cvxpy.org/api_reference/cvxpy.constraints.html#psd):
you build polynomial matrices, write `P >> 0`, and solve with a `Problem`
object that wraps `cvxpy.Problem`. Objectives and free parameters are plain
CVXPY expressions/variables.

## What it does

A symmetric polynomial matrix `P(x)` with a **chordal sparsity pattern** whose
maximal cliques are `C_1, ..., C_t` is certified positive semidefinite through
the decomposition

```
(x'x)^nu * P(x) = sum_k  E_k' S_k(x) E_k,        S_k SOS matrices on the cliques,
```

so instead of one Gram matrix of size `m * |basis|` you get one small PSD
block per clique (Theorem 3.4 of Zheng & Fantuzzi; `nu = 0` by default). For
scalar polynomials, `p >> 0` splits the Gram basis over the maximal cliques of
the **correlative sparsity pattern** of the variables (Waki et al.).

Everything compiles to a standard SDP solved by any CVXPY-supported conic
solver (SCS by default; MOSEK works if installed).

## Installation

```bash
pip install numpy cvxpy      # dependencies
pip install -e .             # from this directory
```

## Quick start

### LMI-style SOS matrix constraints

```python
import cvxpy as cp
import pysparsepmi as psp

x, = psp.polyvar(1)                       # polynomial variables x0
t = cp.Variable()                         # ordinary CVXPY decision variable

F = psp.PolyMatrix([[1 + x**2, x],
                    [x,        1]])

# largest t such that F(x) - t*I admits a (chordally decomposed) SOS certificate
prob = psp.Problem(cp.Maximize(t), [F - psp.eye(2, nvars=1) * t >> 0])
prob.solve(solver="SCS")
print(prob.value)
```

`P >> 0` / `P << Q` return `SOSMatrix` constraint objects; use the class
directly for options:

```python
con = psp.SOSMatrix(P, sparse=True, nu=1)   # (x'x)^nu weight, clique blocks
psp.Problem(None, [con]).solve()
con.cliques, con.block_sizes                # inspect the decomposition
```

> **Note** — keep polynomials on the *left* of products with CVXPY scalars
> (`x * t`, `psp.eye(2, 1) * t`): CVXPY does not know how to multiply its
> expressions by `Polynomial` objects from the left.

### Scalar SOS with correlative sparsity

```python
x = psp.polyvar(4)
p = (x[0]*x[1] - 1)**2 + (x[1]*x[2] - 1)**2 + (x[2]*x[3] - 1)**2
con = p >> 0                                # SOS constraint, csp-split basis
psp.Problem(None, [con]).solve()
print(con.cliques)                          # [[0, 1], [1, 2], [2, 3]]
```

### PMI optimization

Lower-bound `inf_x lambda_min(F(x))` over a semialgebraic set defined by
scalar and/or matrix inequalities, via a sparse Scherer–Hol certificate:

```python
x1, x2 = psp.polyvar(2)
F = psp.PolyMatrix([[1 + x1**2, x1], [x1, 1]])
G = psp.PolyMatrix([[x1*x2*(-4.0) + 1, x1], [x1, 4 - x1**2 - x2**2]])

res = psp.pmi_optimize(F, ineqs=[G, 1 - x2**2], order=2)
print(res.value, res.cliques, res.block_sizes)
```

Equality constraints enter through free polynomial-matrix multipliers, and
`cs=True` additionally exploits correlative sparsity in the variables: every
SOS matrix and multiplier is split over the cliques of the chordally extended
variable graph, on top of the row-clique decomposition of `F`:

```python
res = psp.pmi_optimize(F, ineqs=[...], eqs=[h], order=2, cs=True)
res.cliques, res.var_cliques      # row cliques of F, variable cliques
```

For scalar polynomial optimization with constraints (sparse Lasserre / Waki
et al.), use `psp.sos_lower_bound(f, ineqs=[...], order=d)`.

### Term sparsity (TSSOS-style)

`ts="block"` or `ts="MD"` (mirroring TSSOS's `TS=` option) additionally
splits every Gram matrix into blocks along the cliques of the *term
sparsity pattern* graph, grown by iterating support extension and chordal
extension (block closure or minimum-degree chordal closure; Miller, Wang &
Guo, arXiv:2411.15479):

```python
# first step of the term-sparsity hierarchy, as in
# tssos(F, G, x, 3, TS="MD") followed by tssos(data, TS="MD"):
res1 = psp.pmi_optimize(F, ineqs=[G1, G2], order=3, ts="MD", ts_order=1)
res2 = psp.pmi_optimize(F, ineqs=[G1, G2], order=3, ts="MD", ts_order=2)
res  = psp.pmi_optimize(F, ineqs=[G1, G2], order=3, ts="MD")  # stabilized
res.block_sizes, res.ts_block_sizes    # PSD blocks per constraint
```

The bounds are monotone in the sparse order `ts_order` and converge to the
dense bound of the same relaxation order at stabilization
(`ts_order=None`). The same keywords work on constraint objects
(`SOS(p, ts=...)`, `SOSMatrix(P, ts=...)`) and on `sos_lower_bound`.

### Composition, tensor-train and low-rank structure (state lifting)

Many dense polynomials are evaluated through a chain of low-dimensional
maps, `s_1 = F_1(x_1)`, `s_i = F_i(s_{i-1}, x_i)`, `p = F_n(s_{n-1}, x_n)`:
dynamical systems, Markov chains, neural networks, tensor trains, and
low-rank (CP) polynomials. Lifting the states `s_i` to variables tied to `x`
by equalities makes the problem chain-sparse, so PSD blocks depend on the
state dimensions (ranks) `r_i` and not on `n`:

```python
# controlled 2-state Markov chain (arXiv:2604.17563, Sec. 7.1)
a = lambda x: 0.95 - 0.2 * x**2
b = lambda x: 0.05 - 0.05 * x**2
maps = ([lambda s, x: [a(x), 1 - a(x)]]
        + [lambda s, x: [s[0]*a(x) + s[1]*b(x), s[0]*(1 - a(x)) + s[1]*(1 - b(x))]] * 8
        + [lambda s, x: s[0]*a(x) + s[1]*b(x)])
res = psp.composition_lower_bound(maps, box=1.0, order=2, sense="max")
res.value, res.ranks, res.block_sizes   # ~0.67434; exact max is 1/2 + 0.9^10/2
```

Each map receives the previous state (a list of polynomials) and the local
variable, and returns the new state; ranks are inferred. Two hierarchies are
available:

- `method="chord"` (SL-chord): lifted correlative sparsity, with the
  stage-by-stage elimination order of the papers, so cliques have
  `r_i + r_{i+1} + 1` variables (`r + 2` for rank-`r` CP polynomials —
  the LRPOP hierarchy).
- `method="push"` (SL-push): push-forward potentials `V_i(s_i)` with one
  certificate per stage on `r_{i-1} + 1` variables, at the price of higher
  degrees (`deg V_i = floor(2*order / deg F_i)`).

Front-ends take polynomials in tensor formats (TensorLy layouts, so the output
of `tensorly.decomposition.parafac` / `tensor_train` on a coefficient tensor
plugs in):

```python
psp.cp_lower_bound(factors, box=1.0, order=2)                 # factors[i]: (deg+1, r)
psp.cp_lower_bound(factors, box=1.0, basis="bernstein")       # Bernstein coefficients
psp.tt_lower_bound(cores, box=1.0, order=2, method="push")    # cores[i]: (r_{i-1}, deg+1, r_i)
```

**Matrix-valued objectives.** The last map may return a symmetric
polynomial matrix; the result then bounds `min_x lambda_min(F(x))`
(`sense="max"`: `max_x lambda_max(F(x))`). SL-push certifies the PMI
`F_n(s, x) - V_{n-1}(s) I` on the last stage only; SL-chord uses
`pmi_optimize` with the lifted variable cliques. This extension of the papers
(which treat scalar objectives) gives valid bounds, but convergence of the
matrix versions is not claimed. `cp_lower_bound(..., matrices=[A_1, ...])`
handles `F(x) = sum_l A_l prod_i f_{l,i}(x_i)`.

```python
# worst-case gain lambda_max(P' P) of P = M(x_1) ... M(x_n), with the
# symmetric state S_i = M(x_i)' S_{i-1} M(x_i) lifted (3 scalars per stage)
res = psp.composition_lower_bound([first] + [step] * (n - 2) + [last],
                                  box=1.0, order=2, sense="max", method="push")
```

With a `box`, redundant bounds on the states are derived by interval
arithmetic (as the papers add for convergence) and every state is rescaled by
its bound (`scale=True`), an exact change of variables that keeps the SDP
well conditioned. Vector-valued local variables (`xdims`) and extra local
constraints (`local_ineqs`, `local_eqs`) are supported.

> **Solver note** — on long chains SCS needs many iterations
> (`max_iters=200000` or more). A result with status
> `optimal_inaccurate` is not a certified bound, and can even exceed the
> true optimum; MOSEK is much faster and more accurate at scale.

The underlying building blocks are also exposed: `sos_lower_bound(...,
eqs=[h, ...], cliques=..., multiplier_hosts="one")` for equality
constraints and user-given cliques, `chordal_cliques(pattern, order=...)`
for a prescribed elimination order, and `compose` / `Polynomial.embed`
for substitution and variable-space changes.

### Recovering minimizers

The dual variables of the coefficient-matching constraints are the
pseudo-moments of the relaxation, so every result can propose minimizers
and certify them:

```python
res = psp.sos_lower_bound(x + y, eqs=[x**2 + y**2 - 1])
xs = res.minimizer()                 # first-order moments: [-0.7071, -0.7071]
res.gap(), res.max_violation(xs)     # ~0, ~0: bound tight, xs globally optimal
res.moment_matrix(order=1)           # rank 1 <=> a single atom

res = psp.pmi_optimize(F, ineqs=[G, 1 - x2**2], order=2)   # two minimizers
res.atoms()                          # Henrion-Lasserre extraction: both points
```

- `minimizer()` returns the first-order moments. It is exact when the
  optimal pseudo-moments come from a single point, and averages the
  minimizers otherwise.
- `atoms(vars=None, order=None)` extracts several minimizers from a flat
  moment matrix (Henrion & Lasserre). With correlative sparsity, cross-clique
  moments do not exist, so pass the variables of one clique.
- For PMIs, the matrix-valued moments are trace-normalized.
- For lifted problems, `minimizer()` returns the original variables
  `x_1..x_n`: from the lifted moments (SL-chord) or from the first moments
  of each stage measure (SL-push).
- `gap(x)` and `max_violation(x)` evaluate the original objective
  (`lambda_min`/`lambda_max` for PMIs; the original chain for lifted
  problems) and constraints. A feasible point with zero gap certifies both
  the bound and global optimality, and a positive gap at a good point shows
  that the bound is loose.

## API summary

| Function / class | Purpose |
|---|---|
| `polyvar(n)` | create `n` polynomial variables |
| `PolyMatrix([[...]])`, `eye(m, nvars)` | build symmetric polynomial matrices |
| `p >> 0`, `SOS(p, sparse=True, ts=...)` | scalar SOS constraint with correlative-sparsity basis splitting (optionally term sparsity) |
| `P >> 0`, `SOSMatrix(P, sparse=True, nu=..., ts=..., var_cliques=...)` | matrix SOS constraint with chordal clique decomposition (optionally term sparsity, or variable cliques) |
| `Problem(objective, constraints).solve()` | CVXPY-style problem wrapper |
| `pmi_optimize(F, ineqs, order=d, ts=..., ts_order=s)` | lower-bound `lambda_min(F)` on a semialgebraic set |
| `sos_lower_bound(f, ineqs, order=d, ts=..., ts_order=s)` | sparse Lasserre lower bound for scalar polynomials |
| `sos_lower_bound(f, ineqs, eqs=[...], cliques=...)` | ... with equality constraints and user-given cliques |
| `pmi_optimize(F, ineqs, eqs=[...], cs=True)` | PMI bound with equalities and correlative sparsity in the variables |
| `composition_lower_bound(maps, box, method="chord"/"push")` | state-lifting bounds for chained polynomial maps (SL-chord / SL-push) |
| `cp_lower_bound(factors, ...)`, `tt_lower_bound(cores, ...)` | LRPOP for CP polynomials, state lifting for tensor trains |
| `res.minimizer()`, `res.atoms()`, `res.gap(x)`, `res.max_violation(x)` | recover and certify minimizers from the pseudo-moments |
| `chordal_cliques(pattern, order=...)` | maximal cliques of a chordal extension (optional elimination order) |
| `correlative_sparsity(polys)` | correlative sparsity pattern of a set of polynomials |

Low-level building blocks: `sos_poly_variable` / `sos_matrix_variable` create
Gram-parameterized SOS multipliers, `monomials`, `gram_candidates`,
`reduce_bases` handle basis generation and Newton-style reduction,
`free_poly_variable` creates polynomials with free coefficients, and
`compose` substitutes polynomials into polynomials.

**Scope notes.** This package implements *correlative/matrix* (clique)
sparsity, *term* sparsity (block closure and minimum-degree chordal
closure) and *state lifting* for composition, tensor-train and CP structure
(scalar and matrix-valued objectives). Minimizers are recovered from the
pseudo-moments (first moments or Henrion–Lasserre atom extraction; sparse
gluing of atoms across cliques is not implemented).

## Examples

- `examples/tssos_pmi.py` — two small PMI optimization problems (the first
  bound, −4.0, is exact).
- `examples/term_sparsity_pmi.py` — the term-sparsity hierarchy
  (`ts="MD"`, sparse orders 1, 2, stabilized) on a 5-variable, 5x5 PMI
  problem with two PMI constraints, mirroring TSSOS's
  `tssos(F, G, x, 3, TS="MD")`.
- `examples/quantum_identification.py` — certified GKSL (Lindblad)
  parameter reconstruction for a qubit (arXiv:2501.05270): bounding the
  reconstruction residual subject to a PSD Kossakowski matrix, including a
  certificate that no physical model fits non-physical data.
- `examples/banded_pmi.py` — banded parametric PMI in the style of Example
  5.1 of Zheng & Fantuzzi, comparing sparse clique blocks against the dense
  certificate.
- `examples/low_rank_pop.py` — LRPOP on the well-conditioned rank-2
  Bernstein instances of arXiv:2512.08394 (Sec. 4.2): the exact minimum
  `r` for `n` up to 20 (total degree 40) with a constant 15x15 PSD block.
- `examples/markov_chain_composition.py` — certified upper bounds for a
  controlled two-state Markov chain (arXiv:2604.17563, Sec. 7.1), SL-chord
  versus SL-push, against the closed form `1/2 + 0.9^n/2`, with the
  optimal controls recovered from the pseudo-moments.
- `examples/matrix_product_gain.py` — worst-case gain
  `max lambda_max(P'P)` of a product of parameter-dependent 2x2 matrices,
  lifting the symmetric states `S_i = M(x_i)' S_{i-1} M(x_i)` with a
  matrix-valued last stage (PMI extension of SL-chord / SL-push).
- `examples/sparse_pop.py` — chained scalar polynomial optimization with
  correlative sparsity (20 variables, cliques of size 2).

Run the tests with `python tests/test_pysparsepmi.py` (or `pytest`).

## References

1. Y. Zheng, G. Fantuzzi (2020). Sum-of-squares chordal decomposition of
   polynomial matrix inequalities. arXiv:2007.11410.
2. J. Miller, J. Wang, F. Guo (2024). Sparse polynomial matrix optimization.
   arXiv:2411.15479.
3. H. Waki, S. Kim, M. Kojima, M. Muramatsu (2006). Sums of squares and
   semidefinite program relaxations for polynomial optimization problems with
   structured sparsity. SIAM J. Optim. 17(1).
4. C. W. Scherer, C. W. J. Hol (2006). Matrix sum-of-squares relaxations for
   robust semi-definite programs. Math. Program. 107.
5. W. Parvaiz, J. Aspman, A. Wodecki, G. Korpas, J. Marecek (2025).
   Identifiability of autonomous and controlled open quantum systems.
   arXiv:2501.05270.
6. L. Balada Gaggioli, D. Henrion, M. Korda (2025). Global optimization of
   low-rank polynomials. arXiv:2512.08394.
7. L. Balada Gaggioli, D. Henrion, M. Korda (2026). Composition and tensor
   train structure in polynomial optimization. arXiv:2604.17563.

## Appendix: correspondence with YALMIP and TSSOS

| pysparsepmi | YALMIP / MATLAB | TSSOS (Julia) |
|---|---|---|
| `polyvar(n)` | `sdpvar x y ...` | `@polyvar x[1:n]` |
| `PolyMatrix([[...]])`, `eye(m, nvars)` | matrix of `sdpvar` polys | `Matrix{Poly}` |
| `p >> 0`, `SOS(p, sparse=True)` | `sos(p)` + `sdpsettings('sos.csp',1)` | — |
| `P >> 0`, `SOSMatrix(P, sparse=True, nu=...)` | scalarized `u'*P*u` + `sos.csp` | — |
| `Problem(cp.Maximize(t), cons).solve()` | `solvesos(CNSTR, -t, opts, params)` | — |
| `pmi_optimize(F, ineqs, order=d)` | — | `tssos(F, G, x, d, TS=false)` |
| `pmi_optimize(..., ts="MD", ts_order=s)` | — | `tssos(F, G, x, d, TS="MD")`, then `tssos(data, TS="MD")` |
| `sos_lower_bound(f, ineqs, order=d)` | `solvesos` + `sos.csp` | `cs_tssos(f, g, x, d)` (CS only) |
| `sos_lower_bound(..., ts="block")` | — | `tssos(f, g, x, d, TS="block")` |
| `cp_lower_bound(factors, method="chord")` | — | LRPOP.jl |
| `composition_lower_bound(maps, method="chord"/"push")` | — | SLPOP.jl (SL-chord / SL-push) |
| `chordal_cliques(pattern)` | `cliquesFromSpMatD.m` | `clique_decomp` |
| `correlative_sparsity(polys)` | `corrsparsity.m` | — |
