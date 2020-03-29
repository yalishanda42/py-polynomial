"""Includes all classes which can be frozen."""
from polynomial.frozen import Freezable
from polynomial.core import Constant, Polynomial


class FrozenPolynomial(Freezable, Polynomial):
    """A polynomial which can not be directly modified."""

    def __init__(self, *args, **kwargs):
        """Create a polynomial from the args, and then freeze it."""
        Polynomial.__init__(self, *args, **kwargs)
        self._trim = self._no_op
        self._freeze()

    @classmethod
    def zero_instance(cls):
        return FrozenPolynomial()

    @classmethod
    def from_polynomial(cls, polynomial):
        """Create a frozen copy of the polynomial."""
        return cls(polynomial)

    def _no_op(self):
        """Do nothing. Used as a dummy method."""


class ZeroPolynomial(Freezable, Constant):
    """The zero polynomial."""

    def __init__(self):
        """Equivalent to Polynomial()."""
        Constant.__init__(self, 0)
        self.ajsdakj = "HELLO"
        self._freeze()

    @classmethod
    def zero_instance(cls):
        return ZeroPolynomial()

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