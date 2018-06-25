"""This module focuses on trinomial methods and problem solving."""


from Polynomial import Polynomial, Monomial as M, Constant
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

    def __getattr__(self, name):
        """Implement getattr(self, name).

        Allows access to properties a,b,c with their capitalized letters.
        (e.g. self.A <==> self.a, etc.)
        """
        if name in ("A", "B", "C"):
            return getattr(self, name.lower())
        return object.__getattr__(self, name)

    def __setattr__(self, name, value):
        """Implement setattr(self, name, value).

        Makes sure that when setting a,b,c the polynomial is changed
        accordingly.
        """
        if name in ("a", "b", "c"):
            # set the corresponding value in the polynomial vector also
            self._vector[2 - ord(name) + ord("a")] = value
        elif name in ("A", "B", "C"):
            return setattr(self, name.lower(), value)
        return object.__setattr__(self, name, value)

    @property
    def discriminant(self):
        """Return the discriminant of ax^2 + bx + c = 0."""
        return self.b**2 - 4*self.a*self.c

    def get_complex_roots(self):
        """Return a 2-tuple with the 2 complex roots of ax^2 + bx + c = 0.

        + root is first, - root is second.
        """
        D = self.discriminant
        sqrtD = sqrt(D) if D >= 0 else sqrt(-D)*1j
        return (-self.b + sqrtD)/(2*self.a), (-self.b - sqrtD)/(2*self.a)

    def get_real_roots(self):
        """Return a 2-tuple with the real roots if self.discriminant>=0.

        Return an empty tuple otherwise.
        """
        if self.discriminant < 0: return tuple()
        return self.get_complex_roots()

    def get_complex_factors(self):
        """Return (a, (x-x_0), (x+x_1)), where x_0 and x_1 are the roots."""
        roots = self.get_complex_roots()
        return (Constant(self.a),
                Polynomial(1, -roots[0]),
                Polynomial(1, -roots[1]))

    def get_real_factors(self):
        """Return (self,) if D < 0. Return the factors otherwise."""
        if self.discriminant < 0: return self,
        return self.get_complex_factors()
