"""This module focuses on an exhaustive implementation of polynomials.

(c) Yalishanda <yalishanda@abv.bg>
"""

from .core import (
    Constant,
    PolynomialError,
    DegreeError,
    TermError,
    Monomial,
    Polynomial,
)
from .frozen import FrozenPolynomial, ZeroPolynomial
from .binomial import Binomial, LinearBinomial
from .trinomial import Trinomial, QuadraticTrinomial
