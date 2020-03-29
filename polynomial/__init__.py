"""This module focuses on an exhaustive implementation of polynomials.

(c) Yalishanda <yalishanda@abv.bg>
"""

from .frozen import Freezable
from .core import (
    Constant,
    FrozenPolynomial,
    Monomial,
    Polynomial,
    ZeroPolynomial
)
from .binomial import Binomial, LinearBinomial
from .trinomial import Trinomial, QuadraticTrinomial
