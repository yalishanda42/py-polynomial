"""Unit-testing module defining polynomials __repr__ test cases."""

import inspect
import unittest

import polynomial
from polynomial import (
    Constant,
    LinearBinomial,
    Monomial,
    Polynomial,
    QuadraticTrinomial,
    ZeroPolynomial,
    FrozenPolynomial,
    Trinomial,
    Binomial
)


class TestPolynomialsRepr(unittest.TestCase):
    """Defines polynomials __repr__ test cases."""

    def test_smoke(self):
        """Perform a test case containing all possible complicated inputs."""
        coeffs = [0, 0, -2, 0, 4.6666666666, -(69 + 420j), -1, 1337]
        expect = "Polynomial(-2, 0, 4.6666666666, (-69-420j), -1, 1337)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_canonical_string(self):
        """Test with string coefficients."""
        coeffs = "abcdef"
        expect = """Polynomial('a', 'b', 'c', 'd', 'e', 'f')"""

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_zero_polynomial_repr_is_constructor(self):
        """Test that the 0-polynomial is ZeroPolynomial()'."""
        r = repr(ZeroPolynomial())

        self.assertEqual("ZeroPolynomial()", r)

    def test_monomial_repr(self):
        """Test that repr() output of a Monomial is valid."""
        expect = "Monomial(1, 2)"

        r = repr(Monomial(1, 2))

        self.assertEqual(expect, r)

    def test_constant_repr(self):
        """Test that repr() output of a Constant is valid."""
        expect = "Constant(5)"

        r = repr(Constant(5))

        self.assertEqual(expect, r)

    def test_binomial(self):
        """Test that repr() output of a Binomial is valid."""
        expect = "Binomial(Monomial(4, 3), Monomial(2, 1))"

        r = repr(Binomial((4, 3), (2, 1)))

        self.assertEqual(expect, r)

    def test_linear_binomial(self):
        """Test that repr() output of a LinearBinomial is valid."""
        expect = "LinearBinomial(5, 2)"

        r = repr(LinearBinomial(5, 2))

        self.assertEqual(expect, r)

    def test_trinomial(self):
        """Test that repr() output of a Trinomial is valid."""
        expect = "Trinomial(Monomial(5, 5), Monomial(2, 3), Monomial(1, 1))"

        r = repr(Trinomial((5, 5), (2, 3), (1, 1)))

        self.assertEqual(expect, r)

    def test_quadratic_trinomial(self):
        """Test that repr() output of a QuadraticTrinomial is valid."""
        expect = "QuadraticTrinomial(1, -4, 4)"

        r = repr(QuadraticTrinomial(1, -4, 4))

        self.assertEqual(expect, r)

    def test_frozen(self):
        """Test that repr() output of a FrozenPolynomial is valid."""
        expect = "FrozenPolynomial(1, 2, 3)"
        r = repr(FrozenPolynomial(1, 2, 3))
        self.assertEqual(expect, r)

    def test_all_reprs_start_correctly(self):
        """Test that the repr of all classes start correctly."""
        for cls_str in dir(polynomial):
            cls = getattr(polynomial, cls_str)
            if not inspect.isclass(cls):
                continue
            if not issubclass(cls, Polynomial):
                continue
            self.assertTrue(
                repr(cls()).startswith(cls_str),
                "{0} should start with {1}".format(repr(cls()), cls_str)
            )


if __name__ == '__main__':
    unittest.main()
