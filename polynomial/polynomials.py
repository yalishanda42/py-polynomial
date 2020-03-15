"""This module focuses on an exhaustive implementation of polynomials.

(c) Yalishanda <yalishanda@abv.bg>
"""

from itertools import accumulate
from math import sqrt, inf
import string


def accepts_many_arguments(function):
    """Make a function that accepts an iterable handle many *args."""
    def decorated(self, *args, **kwargs):
        if len(args) == 1 and type(args[0]) not in [int, float, complex]:
            function(self, args[0], kwargs)
        else:
            function(self, args, kwargs)
    return decorated


class Polynomial:
    """Implements a single-variable mathematical polynom."""

    @accepts_many_arguments
    def __init__(self, iterable=[], from_monomials=False):
        """Initialize the polynomial.

        iterable ::= the coefficients from the highest degree term
        to the lowest.
        The method is decorated so that it can accept many *args which
        it automatically transofrms into a single iterable.
        If the from_monomials flag is True then it can accept many
        monomials or a single iterable with monomials which altogether
        add up to form this polynomial.

        Example usage:
        Polynomial([1,2,3,4,5])
        Polynomial(1,2,3,4,5)
        Polynomial(range(1, 6))
        Polynomial([(1,4), (2,3), (3,2), (4,1), (5,0)], from_monomials=True)
        Polynomial(((i + 1, 4 - i) for i in range(5)), from_monomials=True)
        """
        if from_monomials:
            iterable = list(iterable)
            for i, monomial in enumerate(iterable):
                if not isinstance(monomial, Monomial) and len(monomial) == 2:
                    iterable[i] = Monomial(monomial[0], monomial[1])
                elif not isinstance(monomial, Monomial):
                    raise TypeError("{} cannot be a monomial.".
                                    format(monomial))
            iterable.sort(reverse=True, key=lambda m: m.degree)
            self._vector = [0 for _ in range(iterable[0].degree + 1)]
            if self._vector:
                self._vector[-1] = iterable[0].a
                for mon in iterable[1:]:
                    self._vector[mon.degree] += mon.a
        else:
            iterable = list(reversed(list(iterable)))  # a more convenient way
            while iterable != []:
                if not iterable[-1] or iterable[-1] == "0":
                    iterable.pop()
                else:
                    break
            self._vector = iterable

    @property
    def degree(self):
        """Return the degree of the polynomial."""
        if not self:
            return -inf  # the degree of the zero polynomial is -infinity

        return len(self._vector) - 1

    @property
    def derivative(self):
        """Return a polynomial object which is the derivative of self."""
        return Polynomial(reversed([i * self[i]
                                    for i in range(1, self.degree + 1)]))

    @property
    def monomials(self, reverse=True):
        """Return a list with all terms in the form of monomials.

        List is sorted from the highest degree term to the lowest
        by default.
        """
        return sorted([Monomial(k, deg) for deg, k in enumerate(self._vector)],
                      reverse=reverse)

    def calculate(self, x):
        """Calculate the value of the polynomial at a given point."""
        if self.degree < 0:
            return 0

        return sum(ak * (x ** k) for k, ak in enumerate(self._vector))

    def __getattr__(self, name):
        """Get coefficient by letter name: ax^n + bx^{n-1} + ... + yx + z."""
        if len(name) == 1 and name in string.ascii_uppercase:
            return self.__getattr__(name.lower())
        if len(name) == 1 and name in string.ascii_lowercase:
            return self[self.degree - (ord(name) - ord('a'))]

        return object.__getattr__(self, name)

    def __setattr__(self, name, new_value):
        """Set coefficient by letter name: ax^n + bx^{n-1} + ... + yx + z."""
        if len(name) == 1 and name in string.ascii_uppercase:
            self.__setattr__(name.lower(), new_value)
        elif len(name) == 1 and name in string.ascii_lowercase:
            self[self.degree - (ord(name) - ord('a'))] = new_value
        else:
            object.__setattr__(self, name, new_value)

    def __getitem__(self, degree):
        """Get the coefficient of the term with the given degree."""
        if degree > self.degree or degree < 0:
            raise IndexError("Attempt to get coefficient of term with \
degree {0} of a {1}-degree polynomial".format(degree, self.degree))
        return self._vector[degree]

    def __setitem__(self, degree, new_value):
        """Set the coefficient of the term with the given degree."""
        if degree > self.degree:
            raise IndexError("Attempt to set coefficient of term with \
degree {0} of a {1}-degree polynomial".format(degree, self.degree))
        self._vector[degree] = new_value

    def __iter__(self):
        """Return the coefficients from the highest degree to the lowest."""
        return iter(reversed(self._vector))

    def __repr__(self):
        """Return repr(self) in human-friendly form."""
        if self.degree < 0:
            return "0"

        def ones_removed(ak, k):
            #  the  coefficients before the non-zero-degree terms
            #  should not be explicitly displayed if they are
            #  1 or -1
            if ak == 1 and k != 0:
                return ""
            elif ak == -1 and k != 0:
                return "-"
            else:
                return ak

        terms = ["{0}x^{1}".
                 format(ones_removed(ak, k), k)
                 for k, ak in enumerate(self._vector)
                 if ak != 0]
        joined_terms = " + ".join(reversed(terms))
        replace_terms = {"x^1": "x",
                         "x^0": "",
                         " + -": " - "}
        for k, v in replace_terms.items():
            joined_terms = joined_terms.replace(k, v)
        return joined_terms

    def __eq__(self, other):
        """Return self == other.

        self == 0 <==> self == Polynomial()
        """
        if other == 0:
            return self._vector == []

        return self.degree == other.degree and self._vector == other._vector

    def __ne__(self, other):
        """Return self != other.

        self != 0 <==> self != Polynomial()
        """
        if other == 0:
            return self._vector != []

        return self._vector != other._vector

    def __bool__(self):
        """Return not self == 0."""
        return not self == 0

    def __add__(self, other):
        """Return self + other."""
        if not self:
            return other
        elif not other:
            return self
        elif isinstance(other, Polynomial):
            new_vector = []
            max_iterations = max(self.degree, other.degree) + 1
            for i in range(max_iterations):
                a, b = 0, 0
                try:
                    a = self[i]
                except IndexError:
                    pass
                try:
                    b = other[i]
                except IndexError:
                    pass
                new_vector.append(a+b)
            return Polynomial(list(reversed(new_vector)))
        else:
            return self + Constant(other)

    def __radd__(self, other):
        """Return other + self."""
        return self + other

    def __mul__(self, other):
        """Return self * other."""
        if not self or not other:
            return ZeroPolynomial()
        elif isinstance(other, Polynomial):
            self_m = self.get_monomials()
            other_m = other.get_monomials()
            return list(accumulate([x*y for x in self_m for y in other_m]))[-1]
        else:
            return self * Constant(other)

    def __rmul__(self, other):
        """Return other * self."""
        return self * other


class Monomial(Polynomial):
    """Implements a single-variable monomial. A single-term polynomial."""

    def __init__(self, coefficient=0, degree=0):
        """Initialize the following monomial: coefficient * x^(degree)."""
        if type(degree) is not int:
            raise ValueError("Monomial's degree should be a natural number.")
        if degree < 0:
            raise ValueError("Polynomials cannot have negative-degree terms.")
        coeffs = [0 for i in range(degree+1)]
        if coeffs:
            coeffs[0] = coefficient
        Polynomial.__init__(self, coeffs)

    @property
    def coefficient(self):
        """Return the coefficient of the monomial."""
        return self.a

    @coefficient.setter
    def coefficient(self, new_value):
        """Set the coefficient of the monomial."""
        self.a = new_value

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, Monomial):
            return Monomial(self.a * other.a, self.degree + other.degree)
        elif isinstance(other, Polynomial):
            return Polynomial(self) * other  # avoiding stack overflow
        else:
            return self * Constant(other)

    def __rmul__(self, other):
        """Return other * self."""
        return self * other

    def __lt__(self, other):
        """Return self < other.

        Compares the degrees of the monomials and then, if
        they are equal, compares their coefficients.
        """
        if self.degree == other.degree:
            return self.a < other.a
        else:
            return self.degree < other.degree

    def __gt__(self, other):
        """Return self > other.

        Compares the degrees of the monomials and then, if
        they are equal, compares their coefficients.
        """
        if self.degree == other.degree:
            return self.a > other.a
        else:
            return self.degree > other.degree


class Trinomial(Polynomial):
    """Implements single-variable mathematical trinomials."""

    def __init__(self,
                 monomial1=Monomial(),
                 monomial2=Monomial(),
                 monomial3=Monomial()):
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
            raise ValueError("object not a quadratic trinomial since a==0!")
        Polynomial.__init__(self, a, b, c)
        self.a = a
        self.b = b
        self.c = c

    @property
    def discriminant(self):
        """Return the discriminant of ax^2 + bx + c = 0."""
        return self.b**2 - 4*self.a*self.c

    @property
    def complex_roots(self):
        """Return a 2-tuple with the 2 complex roots of ax^2 + bx + c = 0.

        + root is first, - root is second.
        """
        D = self.discriminant
        sqrtD = sqrt(D) if D >= 0 else sqrt(-D)*1j
        return (-self.b + sqrtD)/(2*self.a), (-self.b - sqrtD)/(2*self.a)

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
        """Return (a, (x-x_0), (x+x_1)), where x_0 and x_1 are the roots."""
        roots = self.complex_roots
        return (Constant(self.a),
                Polynomial(1, -roots[0]),
                Polynomial(1, -roots[1]))

    @property
    def real_factors(self):
        """Return (self,) if D < 0. Return the factors otherwise."""
        if self.discriminant < 0:
            return self,
        return self.complex_factors


class Binomial(Polynomial):
    """Implements single-variable mathematical binomials."""

    def __init__(self, monomial1=Monomial(), monomial2=Monomial()):
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

    @property
    def root(self):
        """Solve for ax + b = 0."""
        return -self.b / self.a


class Constant(Monomial):
    """Implements constants as monomials of degree 0."""

    def __init__(self, const=1):
        """Initialize the constant with value const."""
        Monomial.__init__(self, const)

    def __int__(self):
        """Return int(self)."""
        return int(self._vector[0])

    def __float__(self):
        """Return float(self)."""
        return float(self._vector[0])

    def __complex__(self):
        """Return complex(self)."""
        return complex(self._vector[0])


class ZeroPolynomial(Polynomial):
    """The zero polynomial."""

    def __init__(self):
        """Equivalent to Polynomial()."""
        Polynomial.__init__(self)

    def __int__(self):
        """Return 0."""
        return 0

    def __float__(self):
        """Return 0.0."""
        return 0.0

    def __complex__(self):
        """Return 0j."""
        return 0j
