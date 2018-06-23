"""This module focuses on an exhaustive implementation of polynomials.

2018 Yalishanda
"""

from itertools import accumulate


def accepts_many_arguments(function):
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
        add up to form this polynom.
        """
        if from_monomials:
            iterable = list(iterable)
            for i, monomial in enumerate(iterable):
                if type(monomial) != Monomial and len(monomial) == 2:
                    iterable[i] = Monomial(monomial[0], monomial[1])
                elif type(monomial) != Monomial:
                    raise TypeError("{} cannot be a monomial.".
                                    format(type(monomial)))
            iterable.sort(reverse=True, key=lambda m: m.degree)
            self._vector = [iterable[0].a]
            for i in range(iterable[0].degree):
                self._vector.append(0)
            self._vector = list(reversed(self._vector))
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
        return len(self._vector)-1  # thus the degree of the 0-polynomial is -1

    def __getitem__(self, degree):
        """Get the coefficient of the term with the given degree."""
        if degree > self.degree:
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
        if self.degree == -1: return "0"

        def remove_ones(ak, k):
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
                 format(remove_ones(ak, k),
                        k)
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
        if self == 0:
            return other
        elif other == 0:
            return self
        else:
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

    def __mul__(self, other):
        """Return self * other."""
        self_mon = self.get_monomials()
        other_mon = other.get_monomials()
        return list(accumulate([x*y for x in self_mon for y in other_mon]))[-1]

    def calculate(self, x):
        """Calculate the value of the polynomial for a given x."""
        if self.degree == -1:
            return 0
        else:
            sum = 0
            for k, ak in enumerate(self._vector):
                sum += ak * (x ** k)
            return sum

    def get_monomials(self, reverse=True):
        """Return a list with all terms in the form of monomials.

        List is sorted from the highest degree term to the lowest
        by default.
        """
        return sorted(
                      [Monomial(k, deg) for deg, k in enumerate(self._vector)],
                      reverse=reverse)


class Monomial(Polynomial):
    """Implements a single-variable monomial. A single-term polynomial."""

    def __init__(self, coefficient=0, degree=0):
        """Initialize the following monomial: coefficient * x^(degree)."""
        coeffs = [coefficient]
        for i in range(degree):
            coeffs.append(0)
        Polynomial.__init__(self, coeffs)
        self.a = coefficient
        self.coefficient = coefficient  # other name for self.a

    def __mul__(self, other):
        """Multiply two monomials and return the product."""
        return Monomial(self.a * other.a, self.degree + other.degree)

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


class Constant(Monomial):
    """Implements constants as monomials of degree 0."""

    def __init__(self, const=1):
        """Initialize the constant with value const."""
        Monomial.__init__(self, const)


class ZeroPolynomial(Polynomial):
    """The zero polynomial."""

    def __init__(self):
        """Equivalent to Polynomial()."""
        Polynomial.__init__(self)
