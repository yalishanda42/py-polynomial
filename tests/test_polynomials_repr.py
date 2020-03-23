"""Unit-testing module defining polynomials __repr__ test cases."""

import unittest
from polynomial import (
    Constant,
    LinearBinomial,
    Monomial,
    Polynomial,
    QuadraticTrinomial,
    ZeroPolynomial,
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

    def test_linear_binomial(self):
        """Test that repr() output of a LinearBinomial is valid."""
        expect = "LinearBinomial(5, 2)"

        r = repr(LinearBinomial(5, 2))

        self.assertEqual(expect, r)

    def test_quadratic_trinomial(self):
        """Test that repr() output of a QuadraticTrinomial is valid."""
        expect = "QuadraticTrinomial(1, -4, 4)"

        r = repr(QuadraticTrinomial(1, -4, 4))

        self.assertEqual(expect, r)


if __name__ == '__main__':
    unittest.main()
