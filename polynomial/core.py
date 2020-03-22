"""This module defines mutable polynomials, monomials and constants."""

from copy import deepcopy
from math import inf
import string


def accepts_many_arguments(function):
    """Make a function that accepts an iterable handle many *args."""

    def decorated(self, *args, **kwargs):
        if len(args) == 1 and not isinstance(args[0], (int, float, complex)):
            function(self, args[0], kwargs)
        else:
            function(self, args, kwargs)

    return decorated


def extract_polynomial(method):
    """Call method with the second argument as a Polynomial.

    If casting is not possible or not appropriate, raise a ValueError.
    """

    def decorated(self, other):
        if isinstance(other, Polynomial):
            return method(self, other)
        if isinstance(other, (int, float, complex)):
            return method(self, Constant(other))

        raise ValueError(
            "{0}.{1} requires a Polynomial or number, got {2}."
            .format(
                self.__class__.__name__,
                method.__name__,
                type(other).__name__
            )
        )

    return decorated


class Polynomial:
    """Implements a single-variable mathematical polynomial."""

    @accepts_many_arguments
    def __init__(self, iterable=None, from_monomials=False):
        """Initialize the polynomial.

        iterable ::= the coefficients from the highest degree term
        to the lowest.
        The method is decorated so that it can accept many *args which
        it automatically transforms into a single iterable.
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
        if iterable is None:
            iterable = []

        iterable = list(iterable)

        if from_monomials:
            for i, monomial in enumerate(iterable):
                if isinstance(monomial, Monomial):
                    iterable[i] = (monomial.a, monomial.degree)
                elif len(monomial) == 2:
                    continue
                else:
                    raise TypeError("{} cannot be a monomial.".
                                    format(monomial))
            self.terms = iterable
        else:
            self._vector = iterable[::-1]
            self._trim()

    def _trim(self):
        """Trims self._vector to length. Keeps constant terms."""
        if not self._vector or len(self._vector) == 1:
            return

        ind = len(self._vector)
        while self._vector[ind-1] == 0 and ind > 0:
            ind -= 1

        self._vector = self._vector[:ind]

    @property
    def degree(self):
        """Return the degree of the polynomial."""
        if not self:
            return -inf  # the degree of the zero polynomial is -infinity

        return len(self._vector) - 1

    @property
    def derivative(self):
        """Return a polynomial object which is the derivative of self."""
        if not self:
            return ZeroPolynomial()

        return Polynomial([i * self[i] for i in range(self.degree, 0, -1)])

    @property
    def terms(self):
        """Get the terms of self as a list of tuples in coeff, deg form.

        Terms are returned from largest degree to smallest degree, excluding
        any terms with a zero coefficient.
        """
        s_d = self.degree
        return [(coeff, s_d - deg) for deg, coeff
                in enumerate(self) if coeff != 0]

    @terms.setter
    def terms(self, terms):
        """Set the terms of self as a list of tuples in coeff, deg form."""
        if not terms:
            self._vector = [0]
            return

        max_deg = max(terms, key=lambda x: x[1])[1] + 1
        self._vector = [0] * max_deg
        for coeff, deg in terms:
            self._vector[deg] += coeff
        self._trim()

    @property
    def monomials(self, reverse=True):
        """Return a list with all terms in the form of monomials.

        List is sorted from the highest degree term to the lowest
        by default.
        """
        if reverse:
            return [Monomial(k, deg) for k, deg in self.terms]

        return [Monomial(k, deg) for k, deg in reversed(self.terms)]

    def calculate(self, x):
        """Calculate the value of the polynomial at a given point."""
        if self.degree < 0:
            return 0

        return sum(ak * (x ** k) for ak, k in self.terms)

    def __getattr__(self, name):
        """Get coefficient by letter name: ax^n + bx^{n-1} + ... + yx + z."""
        if len(name) != 1:
            return object.__getattribute__(self, name)
        if name in string.ascii_letters:
            return self[self.degree - ord(name.lower()) + ord('a')]
        raise AttributeError("attribute {0} is not defined for Polynomial."
                             .format(name))

    def __setattr__(self, name, new_value):
        """Set coefficient by letter name: ax^n + bx^{n-1} + ... + yx + z."""
        if len(name) != 1:
            object.__setattr__(self, name, new_value)
        elif name in string.ascii_letters:
            self[self.degree - ord(name.lower()) + ord('a')] = new_value
        else:
            raise AttributeError("attribute {0} is not defined for Polynomial."
                                 .format(name))

    def __getitem__(self, degree):
        """Get the coefficient of the term with the given degree."""
        if isinstance(degree, slice):
            indices = degree.indices(self.degree + 1)
            return [self._vector[degree] for degree in range(*indices)]

        if degree > self.degree or degree < 0:
            raise IndexError("Attempt to get coefficient of term with \
degree {0} of a {1}-degree polynomial".format(degree, self.degree))
        return self._vector[degree]

    def __setitem__(self, degree, new_value):
        """Set the coefficient of the term with the given degree."""
        if isinstance(degree, slice):
            indices = degree.indices(self.degree + 1)
            for deg, value in zip(range(*indices), new_value):
                self._vector[deg] = value
        elif degree > self.degree:
            raise IndexError("Attempt to set coefficient of term with \
degree {0} of a {1}-degree polynomial".format(degree, self.degree))

        self._vector[degree] = new_value
        self._trim()

    def __iter__(self):
        """Return the coefficients from the highest degree to the lowest."""
        return reversed(self._vector)

    def __repr__(self):
        """Return repr(self)."""
        if not self:
            return "Polynomial()"
        terms = ', '.join([repr(ak) for ak in self])
        return "Polynomial({0})".format(terms)

    def __str__(self):
        """Return str(self)."""
        if not self:
            return "0"

        def components(ak, k, is_leading):
            ak = str(ak)

            if ak[0] == "-":
                # Strip - from ak
                ak = ak[1:]
                sign = "-" if is_leading else "- "
            else:
                sign = "" if is_leading else "+ "

            # if ak is 1, the 1 is implicit when raising x to non-zero k,
            # so strip it.
            ak = "" if ak == "1" and k != 0 else ak

            # set x^k portion.
            if k == 0:
                p, k = "", ""
            elif k == 1:
                p, k = "x", ""
            else:
                p = "x^"

            return sign, ak, p, k

        # 0: sign, 1: coeff, 2: x^, 3: a
        # eg. -         5       x^     2
        s_d = self.degree
        terms = ["{0}{1}{2}{3}".
                 format(*components(ak, k, k == s_d))
                 for ak, k in self.terms]

        return " ".join(terms)

    def __eq__(self, other):
        """Return self == other.

        self == 0 <==> self == Polynomial()
        """
        if other == 0:
            return bool(self)

        return self.degree == other.degree and self.terms == other.terms

    def __ne__(self, other):
        """Return self != other.

        self != 0 <==> self != Polynomial()
        """
        if other == 0:
            return not self

        return self.degree != other.degree and self.terms != other.terms

    def __bool__(self):
        """Return True if self is not a zero polynomial, otherwise False."""
        self._trim()

        if not self._vector:
            return False
        if len(self._vector) > 1:
            return True
        if self._vector[0] == 0:
            return False
        return True

    @extract_polynomial
    def __add__(self, other):
        """Return self + other."""
        if not self:
            return deepcopy(other)

        if not other:
            return deepcopy(self)

        max_iterations = max(self.degree, other.degree) + 1
        new_vector = [None] * max_iterations

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
            new_vector[-i - 1] = a + b
        return Polynomial(new_vector)

    @extract_polynomial
    def __radd__(self, other):
        """Return other + self."""
        return self + other

    @extract_polynomial
    def __iadd__(self, other):
        """Implement self += other."""
        result = self + other
        self.terms = result.terms
        return self

    @extract_polynomial
    def __mul__(self, other):
        """Return self * other."""
        if not self or not other:
            return ZeroPolynomial()

        result = Polynomial()
        for s_m in self.monomials:
            for o_m in other.monomials:
                result += s_m * o_m
        return result

    @extract_polynomial
    def __rmul__(self, other):
        """Return other * self."""
        return self * other

    @extract_polynomial
    def __imul__(self, other):
        """Implement self *= other."""
        result = self * other
        self.terms = result.terms
        return self

    def __pos__(self):
        """Return +self."""
        self._trim()
        return self

    def __neg__(self):
        """Return -self."""
        self._trim()
        result_vector = [-k for k in self]
        return Polynomial(result_vector)

    @extract_polynomial
    def __sub__(self, other):
        """Return self - other."""
        return self + (-other)

    @extract_polynomial
    def __rsub__(self, other):
        """Return other - self."""
        return other + (-self)

    @extract_polynomial
    def __isub__(self, other):
        """Implement self -= other."""
        result = self - other
        self.terms = result.terms
        return self

    def __copy__(self):
        """Create a shallow copy of self. _vector is not copied."""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Create a deep copy of self."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @extract_polynomial
    def __ifloordiv__(self, other):
        """Return self //= other."""
        self.terms = divmod(self, other)[0].terms
        return self

    @extract_polynomial
    def __floordiv__(self, other):
        """Return self // other."""
        return divmod(self, other)[0]

    @extract_polynomial
    def __divmod__(self, other):
        """Return divmod(self, other).

        The remainder is any term that would have degree < 0.
        """
        if other.degree == -inf:
            raise ZeroDivisionError("Can't divide a Polynomial by 0")

        if isinstance(other, Monomial):
            vec = self._vector[other.degree:]
            remainder = self._vector[:other.degree]
            for i, v in enumerate(vec):
                vec[i] = v / other.a
            return Polynomial(vec[::-1]), Polynomial(remainder[::-1])

        working = deepcopy(self)
        vec = []

        while working.degree >= other.degree:
            val = working.a / other.a
            vec.append(val)
            working -= other * Monomial(val, working.degree - other.degree)

        return Polynomial(vec), working


class Monomial(Polynomial):
    """Implements a single-variable monomial. A single-term polynomial."""

    def __init__(self, coefficient=1, degree=1):
        """Initialize the following monomial: coefficient * x^(degree)."""
        if not isinstance(degree, int):
            raise ValueError("Monomial's degree should be a natural number.")
        if degree < 0:
            raise ValueError("Polynomials cannot have negative-degree terms.")
        coeffs = [coefficient] + [0] * degree
        Polynomial.__init__(self, coeffs)

    @property
    def coefficient(self):
        """Return the coefficient of the monomial."""
        return self.a

    @coefficient.setter
    def coefficient(self, new_value):
        """Set the coefficient of the monomial."""
        self.a = new_value

    @extract_polynomial
    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, Monomial):
            return Monomial(self.a * other.a, self.degree + other.degree)
        if isinstance(other, Polynomial):
            return Polynomial(self) * other  # avoiding stack overflow
        return self * other

    @extract_polynomial
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
        return self.degree < other.degree

    def __gt__(self, other):
        """Return self > other.

        Compares the degrees of the monomials and then, if
        they are equal, compares their coefficients.
        """
        if self.degree == other.degree:
            return self.a > other.a
        return self.degree > other.degree

    def __repr__(self):
        """Return repr(self)."""
        return "Monomial({0!r}, {1!r})".format(self.a, self.degree)


class Constant(Monomial):
    """Implements constants as monomials of degree 0."""

    def __init__(self, const=1):
        """Initialize the constant with value const."""
        Monomial.__init__(self, const, 0)

    def __eq__(self, other):
        """Return self == other."""
        if other == self.const:
            return True
        return super().__eq__(other)

    @property
    def const(self):
        """Return the constant term."""
        return self._vector[0]

    @const.setter
    def const(self, val):
        """Set the constant term."""
        self._vector[0] = val

    def __int__(self):
        """Return int(self)."""
        return int(self.const)

    def __float__(self):
        """Return float(self)."""
        return float(self.const)

    def __complex__(self):
        """Return complex(self)."""
        return complex(self.const)

    def __repr__(self):
        """Return repr(self)."""
        return "Constant({0!r})".format(self.const)


class ZeroPolynomial(Constant):
    """The zero polynomial."""

    def __init__(self):
        """Equivalent to Polynomial()."""
        Constant.__init__(self, 0)

    def __int__(self):
        """Return 0."""
        return 0

    def __float__(self):
        """Return 0.0."""
        return 0.0

    def __complex__(self):
        """Return 0j."""
        return 0j

    def __repr__(self):
        """Return repr(self)."""
        return "ZeroPolynomial()"
