"""pysparsepmi: sparse sum-of-squares relaxations for polynomial matrix inequalities.

A Python port of the sparse-PMI machinery from

* Zheng & Fantuzzi (2020), "Sum-of-squares chordal decomposition of
  polynomial matrix inequalities" (the ``sos.csp`` option in YALMIP, as
  adapted in aeroimperial-optimization/sos-chordal-decomposition-pmi), and
* the polynomial-matrix examples of TSSOS (wangjie212/TSSOS,
  ``example/pmi.jl``).

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
)
from .pmi import pmi_optimize, sos_lower_bound, PMIResult

__version__ = "0.1.0"

__all__ = [
    "Polynomial",
    "polyvar",
    "as_polynomial",
    "poly_matrix",
    "ball_multiplier",
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
    "pmi_optimize",
    "sos_lower_bound",
    "PMIResult",
    "__version__",
]
