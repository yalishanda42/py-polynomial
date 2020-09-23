"""This module defines different types of trinomials and their methods."""

from polynomial.core import (
    Polynomial,
    Monomial,
    Constant,
    FixedDegreePolynomial,
    FixedTermPolynomial
)
from math import sqrt


class Trinomial(FixedTermPolynomial, valid_term_counts=(0, 1, 2, 3)):
    """Implements single-variable mathematical trinomials."""

    def __init__(self,
                 monomial1=None,
                 monomial2=None,
                 monomial3=None):
        """Initialize the trinomial with 3 monomials.

        The arguments can also be 2-tuples in the form:
            (coefficient, degree)
        """
        if not monomial1:
            monomial1 = Monomial(1, 1)
        if not monomial2:
            monomial2 = Monomial(1, 2)
        if not monomial3:
            monomial3 = Monomial(1, 3)
        args = [monomial1, monomial2, monomial3]
        Polynomial.__init__(self, args, from_monomials=True)

    def __repr__(self):
        """Return repr(self)."""
        terms = self.terms
        assert len(terms) == 3
        t1, t2, t3 = terms
        return (
            "Trinomial(Monomial({0}, {1}), Monomial({2}, {3}), "
            "Monomial({4}, {5}))"
            .format(*t1, *t2, *t3)
        )


class QuadraticTrinomial(FixedDegreePolynomial, Trinomial, valid_degrees=2):
    """Implements quadratic trinomials and their related methods."""

    def __init__(self, a=1, b=1, c=1):
        """Initialize the trinomial as ax^2 + bx + c."""
        if a == 0:
            raise ValueError("Object not a quadratic trinomial since a==0!")
        Polynomial.__init__(self, a, b, c)

    @property
    def discriminant(self):
        """Return the discriminant of ax^2 + bx + c = 0."""
        c, b, a = self._vector
        return b * b - 4 * a * c

    @property
    def complex_roots(self):
        """Return a 2-tuple with the 2 complex roots of ax^2 + bx + c = 0.

        + root is first, - root is second.
        """
        c, b, a = self._vector
        D = b * b - 4 * a * c
        sqrtD = sqrt(D) if D >= 0 else sqrt(-D) * 1j
        a = a * 2
        return (-b + sqrtD) / a, (-b - sqrtD) / a

    @property
    def real_roots(self):
        """Return a 2-tuple with the real roots if self.discriminant>=0.

        Return an empty tuple otherwise.
        """
        if self.discriminant < 0:
            return tuple()
        return self.complex_roots

    @property
    def complex_factors(self):
        """Return (a, (x-x_0), (x-x_1)), where x_0 and x_1 are the roots."""
        roots = self.complex_roots
        return (Constant(self.a),
                Polynomial([1, -roots[0]]),
                Polynomial([1, -roots[1]]))

    @property
    def real_factors(self):
        """Return (self,) if D < 0. Return the factors otherwise."""
        if self.discriminant < 0:
            return (self,)
        return self.complex_factors

    def __repr__(self):
        """Return repr(self)."""
        return (
            "QuadraticTrinomial({0!r}, {1!r}, {2!r})"
            .format(self.a, self.b, self.c)
        )
