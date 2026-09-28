"""pysparsepmi: sparse sum-of-squares relaxations for polynomial matrix inequalities.

A Python port of the sparse-PMI machinery from

* Zheng & Fantuzzi (2020), "Sum-of-squares chordal decomposition of
  polynomial matrix inequalities" (the ``sos.csp`` option in YALMIP, as
  adapted in aeroimperial-optimization/sos-chordal-decomposition-pmi),
* Miller, Wang & Guo (2024), "Sparse polynomial matrix optimization"
  (arXiv:2411.15479): term sparsity for PMIs and constrained polynomial
  matrix optimization (the ``ts=``/``ts_order=`` options, mirroring
  TSSOS's ``TS=`` keyword), and
* the polynomial-matrix examples of TSSOS (wangjie212/TSSOS,
  ``example/pmi.jl``), and
* Balada Gaggioli, Henrion & Korda: state lifting for low-rank (CP)
  polynomials (LRPOP, arXiv:2512.08394) and for composition / tensor-train
  structure (SL-chord and SL-push, arXiv:2604.17563).

The modelling API mirrors CVXPY's LMI interface: build polynomial matrices,
write ``P >> 0`` to obtain a (chordally decomposed) SOS-matrix constraint,
and solve with :class:`Problem`, which wraps ``cvxpy.Problem``.
"""

from .polynomial import (
    Polynomial,
    polyvar,
    as_polynomial,
    poly_matrix,
    ball_multiplier,
    compose,
)
from .chordal import chordal_cliques, correlative_sparsity
from .basis import monomials, gram_candidates, reduce_bases
from .sos import (
    PolyMatrix,
    eye,
    SOS,
    SOSMatrix,
    Problem,
    SOSInfeasibleError,
    sos_poly_variable,
    sos_matrix_variable,
    free_poly_variable,
)
from .pmi import pmi_optimize, sos_lower_bound, PMIResult
from .lifting import composition_lower_bound, cp_lower_bound, tt_lower_bound

__version__ = "0.1.0"

__all__ = [
    "Polynomial",
    "polyvar",
    "as_polynomial",
    "poly_matrix",
    "ball_multiplier",
    "compose",
    "chordal_cliques",
    "correlative_sparsity",
    "monomials",
    "gram_candidates",
    "reduce_bases",
    "PolyMatrix",
    "eye",
    "SOS",
    "SOSMatrix",
    "Problem",
    "SOSInfeasibleError",
    "sos_poly_variable",
    "sos_matrix_variable",
    "free_poly_variable",
    "pmi_optimize",
    "sos_lower_bound",
    "PMIResult",
    "composition_lower_bound",
    "cp_lower_bound",
    "tt_lower_bound",
    "__version__",
]
