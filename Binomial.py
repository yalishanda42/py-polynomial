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

    def __getattr__(self, name):
        """Implement getattr(self, name).

        Allows access to properties a,b with their capitalized letters.
        (e.g. self.A <==> self.a, etc.)
        """
        if name in ("A", "B"):
            return getattr(self, name.lower())
        return object.__getattr__(self, name)

    def __setattr__(self, name, value):
        """Implement setattr(self, name, value).

        Makes sure that when setting a,b the polynomial is changed
        accordingly.
        """
        if name in ("a", "b"):
            # set the corresponding value in the polynomial vector also
            self._vector[1 - ord(name) + ord("a")] = value
        elif name in ("A", "B", "C"):
            return setattr(self, name.lower(), value)
        return object.__setattr__(self, name, value)

    def get_root(self):
        """Solve for ax + b = 0."""
        return -self.b / self.a
