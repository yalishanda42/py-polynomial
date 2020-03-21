"""This module focuses on an exhaustive implementation of polynomials.

(c) Yalishanda <yalishanda@abv.bg>
"""

from copy import deepcopy
from math import sqrt, inf
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
    """Implements a single-variable mathematical polynom."""

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

        if from_monomials:
            iterable = list(iterable)
            for i, monomial in enumerate(iterable):
                if isinstance(monomial, Monomial):
                    continue
                if len(monomial) == 2:
                    iterable[i] = Monomial(monomial[0], monomial[1])
                else:
                    raise TypeError("{} cannot be a monomial.".
                                    format(monomial))
            leading_monomial = max(iterable, key=lambda m: m.degree)
            self._vector = [0 for _ in range(leading_monomial.degree + 1)]
            for mon in iterable:
                self._vector[mon.degree] += mon.a
            self._trim()
        else:
            first_index = 0
            iterable = list(iterable)
            for i, val in enumerate(iterable):
                if val in (0, "0"):
                    first_index = i + 1
                else:
                    break

            self._vector = iterable[first_index:][::-1]

    def _trim(self):
        """Trims self._vector to length"""
        if not self._vector:
            return

        ind = len(self._vector)
        while self._vector[ind-1] == 0 and ind >= 0:
            ind -= 1

        self._vector = self._vector[:ind]

    @property
    def degree(self):
        """Return the degree of the polynomial."""
        if not self:
            return -inf  # the degree of the zero polynomial is -infinity

        # Trim vector down in case the leading degrees no longer exist.
        self._trim()

        return len(self._vector) - 1 if self._vector else -inf

    @property
    def derivative(self):
        """Return a polynomial object which is the derivative of self."""
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
            self._vector = []
            return

        max_deg = max(terms, key=lambda x: x[1])[1] + 1
        self._vector = [0] * max_deg
        for coeff, deg in terms:
            self._vector[deg] += coeff

    @property
    def monomials(self, reverse=True):
        """Return a list with all terms in the form of monomials.

        List is sorted from the highest degree term to the lowest
        by default.
        """
        if reverse:
            return [Monomial(k, deg) for k, deg in self.terms]
        else:
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
        terms = ', '.join([repr(ak) for ak in self])
        return "Polynomial({0})".format(terms)

    def __str__(self):
        """Return str(self)."""
        if self.degree < 0:
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
            self._trim()
            return self._vector == []

        return self.degree == other.degree and self.terms == other.terms

    def __ne__(self, other):
        """Return self != other.

        self != 0 <==> self != Polynomial()
        """
        if other == 0:
            self._trim()
            return self._vector != []

        return self.terms != other.terms

    def __bool__(self):
        """Return not self == 0."""
        return self != 0

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
        self._vector = result._vector
        self._trim()
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
        result._trim()
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


class Monomial(Polynomial):
    """Implements a single-variable monomial. A single-term polynomial."""

    def __init__(self, coefficient=0, degree=0):
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
            raise ValueError("Object not a quadratic trinomial since a==0!")
        Polynomial.__init__(self, a, b, c)

    @property
    def discriminant(self):
        """Return the discriminant of ax^2 + bx + c = 0."""
        return self.b ** 2 - 4 * self.a * self.c

    @property
    def complex_roots(self):
        """Return a 2-tuple with the 2 complex roots of ax^2 + bx + c = 0.

        + root is first, - root is second.
        """
        D = self.discriminant
        sqrtD = sqrt(D) if D >= 0 else sqrt(-D) * 1j
        return ((-self.b + sqrtD) / (2 * self.a),
                (-self.b - sqrtD) / (2 * self.a))

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


class Binomial(Polynomial):
    """Implements single-variable mathematical binomials."""

    def __init__(self, monomial1=Monomial(), monomial2=Monomial()):
        """Initialize the binomial with 2 monomials.

        The arguments can also be 2-tuples in the form:
            (coefficient, degree)
        """
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


class Constant(Monomial):
    """Implements constants as monomials of degree 0."""

    def __init__(self, const=1):
        """Initialize the constant with value const."""
        Monomial.__init__(self, const)

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

    def __repr__(self):
        """Return repr(self)."""
        return "ZeroPolynomial()"
