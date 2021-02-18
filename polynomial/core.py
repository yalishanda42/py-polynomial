"""This module defines mutable polynomials, monomials and constants."""
from copy import deepcopy
from itertools import chain
from math import inf
import string


class PolynomialError(Exception):
    """Raised when a Polynomial encounters an error."""


class DegreeError(PolynomialError):
    """Raised when a Polynomial's degree changes."""


class TermError(PolynomialError):
    """Raised when a Polynomial's term count changes."""


def _accepts_many_arguments(function):
    """Make a function that accepts an iterable handle many *args."""

    def decorated(self, *args, **kwargs):
        if len(args) == 1 and not isinstance(args[0], (int, float, complex)):
            function(self, args[0], kwargs)
        else:
            function(self, args, kwargs)

    return decorated


def _extract_polynomial(method):
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


def _get_more_permissive_class(a, b):
    """Return the most permissive class of a, b."""
    a_cls = a.__class__
    b_cls = b.__class__
    return b_cls if issubclass(a_cls, b_cls) else a_cls


def _trim(_vector):
    """Return _vector with all trailing zeros removed."""
    if not _vector or len(_vector) == 1:
        return _vector

    ind = len(_vector)
    while _vector[ind - 1] == 0 and ind > 0:
        ind -= 1

    return _vector[:ind]


def _to_terms(vec):
    """Take a list of numbers and return the tuple form."""
    s_d = _degree(vec, tuples=False)
    return [(coeff, s_d - deg) for deg, coeff
            in enumerate(reversed(vec)) if coeff != 0]


def _degree(vec, tuples=True):
    """Return the degree of vec."""
    if not vec:
        return -inf
    if tuples:
        return max(vec, key=lambda term: term[1] if term[0] else -inf)[1]
    return len(vec) - 1


def _mul(lhs, rhs):
    """Return lhs * rhs."""
    if not lhs or not rhs:
        return [(0, 0)]
    deg = _degree(lhs) + _degree(rhs) + 1

    res = [0] * deg
    for lcoeff, ldeg in lhs:
        for rcoeff, rdeg in rhs:
            res[ldeg + rdeg] += lcoeff * rcoeff

    return _to_terms(res)


def _add(lhs, rhs):
    """Return lhs + rhs."""
    if not lhs:
        return rhs
    if not rhs:
        return lhs
    deg = max(_degree(lhs), _degree(rhs)) + 1

    res = [0] * deg
    for coeff, deg in chain(lhs, rhs):
        res[deg] += coeff

    return _to_terms(res)


def _neg(vec):
    """Return -vec."""
    return [(-coeff, deg) for coeff, deg in vec]


def _sub(lhs, rhs):
    """Return lhs - rhs."""
    if not lhs:
        return _neg(rhs)
    if not rhs:
        return lhs
    deg = max(_degree(lhs), _degree(rhs)) + 1

    res = [0] * deg
    for coeff, deg in lhs:
        res[deg] += coeff
    for coeff, deg in rhs:
        res[deg] -= coeff

    return _to_terms(res)


class Polynomial:
    """Implements a single-variable mathematical polynomial."""

    @_accepts_many_arguments
    def __init__(self, iterable, from_monomials=False):
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
        if from_monomials:
            def monomial_to_tuple(monomial):
                if isinstance(monomial, Monomial):
                    return monomial.a, monomial.degree
                if len(monomial) == 2:
                    return monomial

                raise TypeError("{} cannot be a monomial.".
                                format(monomial))

            self.terms = [monomial_to_tuple(monomial) for monomial in iterable]
        else:
            self._vector = _trim(list(iterable)[::-1])

    @classmethod
    def zero_instance(cls):
        """Return the Polynomial which is 0."""
        return Polynomial()

    def _trim(self):
        """Trims self._vector to length. Keeps constant terms."""
        self._vector = _trim(self._vector)

    @property
    def degree(self):
        """Return the degree of the polynomial."""
        if not self:
            return -inf  # the degree of the zero polynomial is -infinity

        return len(self._vector) - 1

    @property
    def derivative(self):
        """Return a polynomial object which is the derivative of self."""
        return self.nth_derivative()

    def nth_derivative(self, n=1):
        """Return the polynomial object which is the nth derivative of self."""
        if not isinstance(n, int) or n < 0:
            raise ValueError(
                "n must be a non-negative integer (got {0})".format(n)
            )

        if not self or n > self.degree:
            # Short circuit since the result would be zero.
            return self.zero_instance()

        if n == 0:
            return deepcopy(self)
        if n == 1:
            factors = range(1, self.degree + 1)
        else:
            d = self.degree - n + 1
            factorial_term = n + 1
            factors = [1] * d

            # Calculate n! for base term.
            for i in range(1, factorial_term):
                factors[0] *= i

            for i in range(1, d):
                # The last number is n * (n-1) * (n-2) * ... * i
                # The next number is (n+1) * n * (n-1) * ... * i + 1
                # To get the next number, we multiply the last number by
                # n + 1 and divide by i.
                factors[i] = (factors[i - 1] // i) * factorial_term
                factorial_term += 1

        return Polynomial(
            [c * x for c, x
             in zip(self, reversed(factors))]
        )

    def integral(self, a, b):
        """Return the integral of self from a to b."""
        res = self._indefinite_integral
        return res.calculate(b) - res.calculate(a)

    @property
    def _indefinite_integral(self):
        """Return the polynomial object which is the integral of self."""
        if not self:
            return Polynomial()

        return Polynomial(
            [c/x for c, x in
                zip(self, range(self.degree + 1, 0, -1))] + [0]
            )

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
            _vector = [0]
        else:
            list_len = max(terms, key=lambda x: x[1])[1] + 1
            _vector = [0] * list_len

            for coeff, deg in terms:
                _vector[deg] += coeff

            _vector = _trim(_vector)
        self._vector = _vector

    @property
    def monomials(self):
        """Return a list with all terms in the form of monomials.

        List is sorted from the highest degree term to the lowest.
        """
        return [Monomial(k, deg) for k, deg in self.terms]

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
            return self._vector[degree]
        if degree == -inf and self.degree == -inf:
            return 0
        if degree > self.degree or degree < 0:
            raise IndexError("Attempt to get coefficient of term with \
degree {0} of a {1}-degree polynomial".format(degree, self.degree))
        return self._vector[degree]

    def __setitem__(self, degree, new_value):
        """Set the coefficient of the term with the given degree."""
        if isinstance(degree, slice):
            self._vector[degree] = new_value
        elif degree == -inf:
            if self.degree == -inf:
                self._vector = [new_value]
            else:
                raise IndexError(
                    "Can not set term with degree -inf on a"
                    " non-zero polynomial."
                )
        elif degree > self.degree:
            raise IndexError("Attempt to set coefficient of term with \
degree {0} of a {1}-degree polynomial".format(degree, self.degree))
        else:
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

    @_extract_polynomial
    def __eq__(self, other):
        """Return self == other.

        self == 0 <==> self == Polynomial()
        """
        if other == 0:
            return not self

        return self.degree == other.degree and self.terms == other.terms

    @_extract_polynomial
    def __ne__(self, other):
        """Return self != other.

        self != 0 <==> self != Polynomial()
        """
        if other == 0:
            return bool(self)

        return self.degree != other.degree or self.terms != other.terms

    def __bool__(self):
        """Return True if self is not a zero polynomial, otherwise False."""
        self._trim()

        if not self._vector:
            return False
        if len(self._vector) > 1:
            return True

        return self._vector[0] != 0

    @_extract_polynomial
    def __add__(self, other):
        """Return self + other."""
        if not self:
            return deepcopy(other)

        if not other:
            return deepcopy(self)

        return self.__class__().try_set_self(
            _add(self.terms, other.terms)
        )

    @_extract_polynomial
    def __radd__(self, other):
        """Return other + self."""
        return self + other

    @_extract_polynomial
    def __iadd__(self, other):
        """Implement self += other."""
        return self.try_set_self(_add(self.terms, other.terms))

    @_extract_polynomial
    def __mul__(self, other):
        """Return self * other."""
        if not self or not other:
            return _get_more_permissive_class(self, other).zero_instance()

        ret_val = deepcopy(self)
        ret_val *= other
        return ret_val

    @_extract_polynomial
    def __rmul__(self, other):
        """Return other * self."""
        return self * other

    @_extract_polynomial
    def __imul__(self, other):
        """Implement self *= other."""
        return self.try_set_self(_mul(self.terms, other.terms))

    def __pos__(self):
        """Return +self."""
        self._trim()
        return deepcopy(self)

    def __neg__(self):
        """Return -self."""
        ret_val = deepcopy(self)
        ret_val._vector = [-x for x in _trim(self._vector)]
        return ret_val

    @_extract_polynomial
    def __sub__(self, other):
        """Return self - other."""
        return self + (-other)

    @_extract_polynomial
    def __rsub__(self, other):
        """Return other - self."""
        return other + (-self)

    @_extract_polynomial
    def __isub__(self, other):
        """Implement self -= other."""
        return self.try_set_self(_sub(self.terms, other.terms))

    @_extract_polynomial
    def __ifloordiv__(self, other):
        """Return self //= other."""
        return self.try_set_self(divmod(self, other)[0].terms)

    @_extract_polynomial
    def __floordiv__(self, other):
        """Return self // other."""
        return divmod(self, other)[0]

    @_extract_polynomial
    def __imod__(self, other):
        """Return self %= other."""
        return self.try_set_self(divmod(self, other)[1].terms)

    @_extract_polynomial
    def __mod__(self, other):
        """Return self % other."""
        return divmod(self, other)[1]

    @_extract_polynomial
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

        working = self.terms
        wd0 = _degree(working)
        other_terms = other.terms
        other_deg = other.degree
        vec = []

        while wd0 >= other_deg:
            val = working[0][0] / other.a
            wd = wd0
            working = _sub(working, _mul(other_terms, [(val, wd - other_deg)]))
            wd0 = _degree(working)
            vec.append((val, wd - other_deg if wd0 != -inf else 0))

        return (
            Polynomial(vec, from_monomials=True),
            Polynomial(working, from_monomials=True)
        )

    def __pow__(self, power, modulo=None):
        """Return self ** power or pow(self, other, modulo)."""
        if not isinstance(power, int):
            raise ValueError(
                "Can't call Polynomial() ** x with a non-integer type."
            )

        if power < 0:
            raise ValueError(
                "Polynomial can only be raised to a non-negative power."
            )

        if power == 0:
            result = Constant(1)
        elif power % 2 == 1:
            result = Polynomial(self)
            if power > 1:
                result *= (self ** (power // 2)) ** 2
        else:
            if power == 2:
                result = Polynomial(self)
            else:
                result = self ** (power // 2)
            result *= result

        return result % modulo if modulo is not None else result

    def __ipow__(self, other):
        """Return self **= power."""
        terms = (self ** other).terms
        return self.try_set_self(terms)

    def __lshift__(self, other):
        """Return self << other.

        Increases the degree of each term by other.
        """
        ret = deepcopy(self)
        ret <<= other
        return ret

    def __ilshift__(self, other):
        """Return self <<= other.

        Increases the degree of each term by other.
        """
        if other < 0:
            self >>= -other
        else:
            self._vector = [0] * other + self._vector

        return self

    def __rshift__(self, other):
        """Return self >> other.

        Decreases the degree of each term by other.
        """
        ret = deepcopy(self)
        ret >>= other
        return ret

    def __irshift__(self, other):
        """Return self >>= other.

        Decreases the degree of each term by other.
        """
        if other < 0:
            self <<= -other
        else:
            self._vector = _trim(self._vector[other:])

        return self

    def __contains__(self, item):
        """Return item in self.

        Requires item to be a tuple, list of tuples, a set of tuples,
        or a Polynomial. Each tuple should contain two values, the
        first being the coefficient and the second being the degree.
        """
        if isinstance(item, tuple):
            return item in self.terms
        if isinstance(item, list):
            return set(item).issubset(self.terms)
        if isinstance(item, set):
            return item.issubset(self.terms)
        if isinstance(item, Polynomial):
            return set(item.terms).issubset(self.terms)
        raise ValueError(
            "Can not check {0} for membership. A two-tuple, list of "
            "two-tuples, a set, or a Polynomial are required."
                .format(type(item).__name__)
        )

    def terms_are_valid(self, terms):
        """Return true if the terms are valid."""
        return True

    def try_set_self(self, terms):
        """Try applying terms to self if possible.

        If not possible, returns a Polynomial with the terms.
        """
        if self.terms_are_valid(terms):
            self.terms = terms
            return self
        return Polynomial(terms, from_monomials=True)


def _setvalue_decorator(error, _terms_are_valid, _fn):
    """Decorate __setattr__, checking if self._vector is still valid."""

    def method(self, name, new_value):
        _fn(self, name, new_value)
        if (name == '_vector' and
            not _terms_are_valid(self, _to_terms(self._vector))):
            raise error

    return method


class FixedDegreePolynomial(Polynomial):
    """This Polynomial must maintain its degree."""

    def __init_subclass__(cls, **kwargs):
        """Init a subclass of self."""
        deg = kwargs["valid_degrees"]
        if not isinstance(deg, tuple):
            deg = (deg,)

        cls.valid_degrees = deg

        orig_terms_are_valid = cls.terms_are_valid

        def _terms_are_valid(self, terms):
            return _degree(terms) in self.valid_degrees

        def terms_are_valid(self, terms):
            return (
                _terms_are_valid(self, terms)
                and orig_terms_are_valid(self, terms)
            )

        cls.terms_are_valid = terms_are_valid
        cls.__setattr__ = _setvalue_decorator(
            DegreeError,
            _terms_are_valid,
            cls.__setattr__
        )


class FixedTermPolynomial(Polynomial):
    """This Polynomial must maintain the number of terms."""

    def __init_subclass__(cls, **kwargs):
        """Init a subclass of self.

        Expects valid_term_counts to be provided as a tuple.
        """
        cls.valid_term_counts = kwargs["valid_term_counts"]

        orig_terms_are_valid = cls.terms_are_valid

        # Check that number of terms is correct.
        def _terms_are_valid(self, terms):
            return len(terms) in self.valid_term_counts

        def terms_are_valid(self, terms):
            return (
                _terms_are_valid(self, terms)
                and orig_terms_are_valid(self, terms)
            )

        cls.terms_are_valid = terms_are_valid
        cls.__setattr__ = _setvalue_decorator(
            TermError,
            _terms_are_valid,
            cls.__setattr__
        )


class Monomial(FixedTermPolynomial, valid_term_counts=(0, 1)):
    """Implements a single-variable monomial. A single-term polynomial."""

    def __init__(self, coefficient=1, degree=1):
        """Initialize the following monomial: coefficient * x^(degree)."""
        if not isinstance(degree, int):
            raise ValueError("Monomial's degree should be a natural number.")
        if degree < 0:
            raise ValueError("Polynomials cannot have negative-degree terms.")
        self._degree = degree
        self._coeff = coefficient

    def _trim(self):
        """Trims self._vector to length. Keeps constant terms."""

    @property
    def terms(self):
        """Get the terms of self as a list of tuples in coeff, deg form.

        Terms are returned from largest degree to smallest degree, excluding
        any terms with a zero coefficient.
        """
        if self._coeff == 0:
            return [(0, 0)]
        return [(self._coeff, self._degree)]

    @terms.setter
    def terms(self, terms):
        """Set the terms of self as a list of tuples in coeff, deg form."""
        if not terms:
            self._coeff = 0
        elif len(terms) == 1:
            self._coeff, self._degree = terms[0]
        else:
            terms = sorted([term for term in terms if term[0] != 0],
                           key=lambda x: x[1])
            if terms[0][1] == terms[-1][1]:
                self._coeff = sum(term[0] for term in terms)
                self._degree = terms[0][1]
            else:
                err_msg = "terms has more than one non-zero term."
                curr_coeff, curr_deg = terms[0]
                termx = []

                for coeff, deg in terms[1:]:
                    if curr_deg == deg:
                        curr_coeff += coeff
                    else:
                        if curr_coeff != 0:
                            if termx:
                                raise TermError(err_msg)
                            termx.append((curr_coeff, curr_deg))
                        curr_coeff = coeff
                        curr_deg = deg
                if termx:
                    if curr_coeff:
                        raise TermError(err_msg)
                    self._coeff, self._degree = termx[0]
                else:
                    self._coeff = curr_coeff
                    self._degree = curr_deg

    @property
    def _vector(self):
        """Get _vector."""
        if self.degree == -inf:
            return [0]
        return [0] * self._degree + [self._coeff]

    @_vector.setter
    def _vector(self, _vector):
        """Set _vector."""
        max_deg = len(_vector) - 1
        is_set = False
        for index, coeff in enumerate(reversed(_vector)):
            if coeff != 0:
                if is_set:
                    raise TermError("_vector has > 1 non-zero term.")
                self._coeff = coeff
                self._degree = max_deg - index
                is_set = True
        if not is_set:
            self._coeff = 0

    @classmethod
    def zero_instance(cls):
        """Return the Monomial which is 0."""
        return Monomial(0, 0)

    @property
    def coefficient(self):
        """Return the coefficient of the monomial."""
        return self._coeff

    @coefficient.setter
    def coefficient(self, coeff):
        """Set the coefficient of the monomial."""
        self._coeff = coeff

    @property
    def degree(self):
        """Return the degree of the monomial."""
        if self._coeff == 0:
            self._degree = -inf
        elif self._degree == -inf:
            self._degree = 0

        return self._degree

    @degree.setter
    def degree(self, degree):
        """Set the degree of the monomial."""
        self._degree = degree

    @_extract_polynomial
    def __mul__(self, other):
        """Return self * other.

        The class which is more permissive will be returned.
        """
        if isinstance(other, Monomial) and self and other:
            return Monomial(self.coefficient * other.coefficient,
                            self.degree + other.degree)
        return super().__mul__(other)

    @_extract_polynomial
    def __rmul__(self, other):
        """Return other * self.

        The class which is more permissive will be returned.
        """
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

    def __pow__(self, power, modulo=None):
        """Return self ** power or pow(self, other, modulo)."""
        result = deepcopy(self)
        result **= power

        return result % modulo if modulo is not None else result

    def __ipow__(self, other):
        """Return self **= power.

        Assumes self is mutable.
        Does not mutate in the case that self == 0 and other != 1.
        """
        if not isinstance(other, int):
            raise ValueError(
                "Can't call Monomial() **= x with a non-integer type."
            )

        if other < 0:
            raise ValueError(
                "Monomial can only be raised to a non-negative power."
            )

        if not self:
            if other != 0:
                return self
            terms = [(1, 0)]
        else:
            terms = [(self.coefficient ** other, self.degree * other)]

        # No native option exists to modify Monomial degree.
        return self.try_set_self(terms)

    def __lshift__(self, other):
        """Return self << other.

        Returns a Monomial that is self * x^other.
        """
        if other < 0:
            return self >> -other

        if not self:
            return self.zero_instance()

        return Monomial(self.coefficient, self.degree + other)

    def __ilshift__(self, other):
        """Return self <<= other.

        Returns a Monomial that is self * x^other. Does not
        guarantee the same type is returned.
        """
        if other < 0:
            self >>= -other
            return self

        if not self:
            return self

        return self.try_set_self([(self.coefficient, self.degree + other)])

    def __irshift__(self, other):
        """Return self >>= other."""
        if other < 0:
            self <<= -other
            return self

        if not self:
            return self

        if other > self.degree:
            return self.try_set_self([(0, 0)])
        return self.try_set_self([(self.coefficient, self.degree - other)])

    def __repr__(self):
        """Return repr(self)."""
        deg = max(0, self.degree)
        return "Monomial({0!r}, {1!r})".format(self.coefficient, deg)

    def __getitem__(self, degree):
        """Get the coefficient of the term with the given degree."""
        if isinstance(degree, slice):
            return self._vector[degree]
        if degree == self.degree:
            return self._coeff
        if degree > self.degree or degree < 0:
            raise IndexError("Attempt to get coefficient of term with \
degree {0} of a {1}-degree monomial".format(degree, self.degree))
        return 0

    def __setitem__(self, degree, new_value):
        """Set the coefficient of the term with the given degree."""
        if isinstance(degree, slice):
            _vector = self._vector
            _vector[degree] = new_value
            self._vector = _vector
        elif degree == self.degree:
            self.coefficient = new_value
        else:
            raise TermError("Can not set more than 1 term on Monomial.")


class Constant(FixedDegreePolynomial, Monomial, valid_degrees=(0, -inf)):
    """Implements constants as monomials of degree 0."""

    def __init__(self, const=1):
        """Initialize the constant with value const."""
        Monomial.__init__(self, const, 0)

    @classmethod
    def zero_instance(cls):
        """Return the constant which is 0."""
        return Constant(0)

    def __eq__(self, other):
        """Return self == other."""
        if other == self.const:
            return True
        return super().__eq__(other)

    @property
    def degree(self):
        """Return self.degree."""
        return 0 if self._coeff else -inf

    @degree.setter
    def degree(self, degree):
        """Set self.degree."""
        raise DegreeError("Can't change the degree of Constant")

    @property
    def const(self):
        """Return the constant term."""
        return self.coefficient

    @const.setter
    def const(self, val):
        """Set the constant term."""
        self.coefficient = val

    @_extract_polynomial
    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, Constant):
            return Constant(self.const * other.const)

        return super().__mul__(other)

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
