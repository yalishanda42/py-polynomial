"""This module defines different types of binomials and their methods."""

from polynomial import Polynomial, Monomial


class Binomial(Polynomial):
    """Implements single-variable mathematical binomials."""

    def __init__(self, monomial1=None, monomial2=None):
        """Initialize the binomial with 2 monomials.

        The arguments can also be 2-tuples in the form:
            (coefficient, degree)
        """
        if not monomial1:
            monomial1 = Monomial(1, 1)
        if not monomial2:
            monomial2 = Monomial(2, 2)
        Polynomial.__init__(self, [monomial1, monomial2], from_monomials=True)


class LinearBinomial(Binomial):
    """Implements linear binomials and their methods."""

    def __init__(self, a=1, b=0):
        """Initialize the binomial as ax + b."""
        if a == 0:
            raise ValueError("object not a linear binomial since a = 0!")
        Polynomial.__init__(self, [a, b])

    @property
    def root(self):
        """Solve for ax + b = 0."""
        return -self.b / self.a

    def __repr__(self):
        """Return repr(self)."""
        return "LinearBinomial({0!r}, {1!r})".format(self.a, self.b)
