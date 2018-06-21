"""This module focuses on trinomial methods and problem solving."""


from Polynomial import Polynomial, Monomial as M
from math import sqrt


class Trinomial(Polynomial):
    """Implements single-variable mathematical trinomials."""

    def __init__(self, monomial1=M(), monomial2=M(), monomial3=M()):
        """Initialize the trinomial with 3 monomials.

        The arguments can also be 2-tuples in the form:
            (coefficient, degree)
        """
        args = [monomial1, monomial2, monomial3]
        Polynomial.__init__(self, args, from_monomials=True)


class QuadraticTrinomial(Trinomial):
    """Implements quadratic trinomials and their related methods."""

    def __init__(self, a=1, b=0, c=0):
        """Initialize the trinomial as ax^2 + bx + c."""
        if a == 0:
            raise ValueError("object not a quadratic trinomial since a=0!")
        Polynomial.__init__(self, a, b, c)
        self.a = a
        self.b = b
        self.c = c

    @property
    def discriminant(self):
        """Return the discriminant of ax^2 + bx + c = 0."""
        return self.b**2 - 4*self.a*self.c

    def get_roots(self):
        """Return a 2-tuple with the 2 roots of ax^2 + bx + c = 0.

        If self.discriminant < 0, it automatically converts them to complex.
        + root is first, - root is second.
        """
        D = self.discriminant
        sqrtD = sqrt(D) if D >= 0 else sqrt(-D)*1j
        return (-self.b + sqrtD)/(2*self.a), (-self.b - sqrtD)/(2*self.a)
