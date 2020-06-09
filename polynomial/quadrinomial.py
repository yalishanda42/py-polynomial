"""This module defines different types of quadrinomials and their methods."""

from cmath import sqrt as csqrt
from polynomial.core import (
    Polynomial,
    Monomial,
    Constant,
    FixedDegreePolynomial,
    FixedTermPolynomial
)


class Quadrinomial(FixedTermPolynomial, valid_term_counts=(0, 1, 2, 3, 4)):
    """Implements single-variable mathematical trinomials."""

    def __init__(self,
                 monomial1=None,
                 monomial2=None,
                 monomial3=None,
                 monomial4=None):
        """Initialize the quadrinomial with 4 monomials.

        The arguments can also be 2-tuples in the form:
            (coefficient, degree)
        """
        if not monomial1:
            monomial1 = Monomial(1, 1)
        if not monomial2:
            monomial2 = Monomial(1, 2)
        if not monomial3:
            monomial3 = Monomial(1, 3)
        if not monomial4:
            monomial4 = Monomial(1, 4)
        args = [monomial1, monomial2, monomial3, monomial4]
        Polynomial.__init__(self, args, from_monomials=True)

    def __repr__(self):
        """Return repr(self)."""
        terms = self.terms
        assert len(terms) == 4
        t1, t2, t3, t4 = terms
        return (
            "Quadrinomial(Monomial({0}, {1}), Monomial({2}, {3}), "
            "Monomial({4}, {5}), Monomial({6}, {7}))"
            .format(*t1, *t2, *t3, *t4)
        )


class CubicQuadrinomial(FixedDegreePolynomial, Quadrinomial, valid_degrees=3):
    """Implements cubic polynomials and their related methods."""

    def __init__(self, a=1, b=1, c=1, d=1):
        """Initialize the quadrinomial as ax^3 + bx^2 + cx + d."""
        if a == 0:
            raise ValueError("Object not a cubic quadrinomial since a==0!")
        Polynomial.__init__(self, a, b, c, d)

    @property
    def complex_roots(self):
        """Return a 3-tuple with the complex roots of ax^3 + bx^2 + cx + d = 0.

        This method uses the general cubic roots formula.
        """
        a, b, c, d = self.a, self.b, self.c, self.d

        delta0 = b*b - 3*a*c
        delta1 = 2*b*b*b - 9*a*b*c + 27*a*a*d
        cardano1 = ((delta1 + csqrt(delta1**2 - 4*(delta0**3))) / 2) ** (1/3)
        cardano2 = ((delta1 - csqrt(delta1**2 - 4*(delta0**3))) / 2) ** (1/3)

        cardano = cardano2 if not cardano1 else cardano1

        xi = (-1 + csqrt(-3)) / 2

        roots = [(b + (xi**k)*cardano + (delta0 / (xi**k)*cardano)) / (-3 * a)
                 for k in range(3)]

        return tuple(roots)

    @property
    def complex_factors(self):
        """Return (a, (x-x_0), (x-x_1), (x-x_2)), where x_i are the roots."""
        roots = self.complex_roots
        return (Constant(self.a),
                Polynomial(1, -roots[0]),
                Polynomial(1, -roots[1]),
                Polynomial(1, -roots[2]))

    def __repr__(self):
        """Return repr(self)."""
        return (
            "CubicQuadrinomial({0!r}, {1!r}, {2!r}, {3!r})"
            .format(self.a, self.b, self.c, self.d)
        )
