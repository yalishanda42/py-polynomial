"""Focuses on binomial methods and problem solving."""


from Polynomial import Polynomial, Monomial as M


class Binomial(Polynomial):
    """Implements single-variable mathematical binomials."""

    def __init__(self, monomial1=M(), monomial2=M()):
        """Initialize the binomial with 2 monomials.

        The arguments can also be 2-tuples in the form:
            (coefficient, degree)
        """
        args = [monomial1, monomial2]
        Polynomial.__init__(self, args, from_monomials=True)


class LinearBinomial(Binomial):
    """Implements linear binomials and their methods."""

    def __init__(self, a=1, b=0):
        """Initialize the binomial as ax + b."""
        if a == 0:
            raise ValueError("object not a linear binomial since a = 0!")
        Polynomial.__init__(self, a, b)
        self.a = a
        self.b = b

    def get_root(self):
        """Solve for ax + b = 0."""
        return -self.b / self.a
