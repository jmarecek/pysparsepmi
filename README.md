# pysparsepmi

Sparse sum-of-squares (SOS) relaxations for **polynomial matrix inequalities
(PMIs)** in Python, exploiting chordal sparsity.

The methods implemented here are based on:

- Y. Zheng, G. Fantuzzi, *Sum-of-squares chordal decomposition of polynomial
  matrix inequalities* ([arXiv:2007.11410](https://arxiv.org/pdf/2007.11410)),
- J. Miller, J. Wang, F. Guo, *Sparse Polynomial Matrix Optimization*
  ([arXiv:2411.15479](https://arxiv.org/abs/2411.15479)),

with implementation reference to the accompanying
[sos-chordal-decomposition-pmi](https://github.com/aeroimperial-optimization/sos-chordal-decomposition-pmi)
code and [TSSOS](https://github.com/wangjie212/TSSOS).

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

For scalar polynomial optimization with constraints (sparse Lasserre / Waki
et al.), use `psp.sos_lower_bound(f, ineqs=[...], order=d)`.

## API summary

| Function / class | Purpose |
|---|---|
| `polyvar(n)` | create `n` polynomial variables |
| `PolyMatrix([[...]])`, `eye(m, nvars)` | build symmetric polynomial matrices |
| `p >> 0`, `SOS(p, sparse=True)` | scalar SOS constraint with correlative-sparsity basis splitting |
| `P >> 0`, `SOSMatrix(P, sparse=True, nu=...)` | matrix SOS constraint with chordal clique decomposition |
| `Problem(objective, constraints).solve()` | CVXPY-style problem wrapper |
| `pmi_optimize(F, ineqs, order=d)` | lower-bound `lambda_min(F)` on a semialgebraic set |
| `sos_lower_bound(f, ineqs, order=d)` | sparse Lasserre lower bound for scalar polynomials |
| `chordal_cliques(pattern)` | maximal cliques of a chordal extension |
| `correlative_sparsity(polys)` | correlative sparsity pattern of a set of polynomials |

Low-level building blocks: `sos_poly_variable` / `sos_matrix_variable` create
Gram-parameterized SOS multipliers, `monomials`, `gram_candidates`,
`reduce_bases` handle basis generation and Newton-style reduction.

**Scope notes.** This package implements *correlative/matrix* (clique)
sparsity — not *term* sparsity — and provides bounds only (no moment-side
solution extraction).

## Examples

- `examples/tssos_pmi.py` — two small PMI optimization problems (the first
  bound, −4.0, is exact).
- `examples/banded_pmi.py` — banded parametric PMI in the style of Example
  5.1 of Zheng & Fantuzzi, comparing sparse clique blocks against the dense
  certificate.
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
