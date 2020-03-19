"""Unit-testing module defining polynomials __str__ test cases."""

import unittest
from polynomial import Polynomial, ZeroPolynomial, Monomial, Constant, LinearBinomial, QuadraticTrinomial


class TestPolynomialsRepr(unittest.TestCase):
    """Defines polynomials __str__ test cases."""

    def test_smoke(self):
        """Perform a test case containing all possible complicated inputs."""
        coeffs = [0, 0, -2, 0, 4.6666666666, -(69 + 420j), -1, 1337]
        expect = "Polynomial(-2, 0, 4.6666666666, (-69-420j), -1, 1337)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_negative_coefficient_replaces_plus_with_minus(self):
        """Test that negative coefficients replace the + between the terms."""
        coeffs = [3, -2, 0, 0]
        expect = "Polynomial(3, -2, 0, 0)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_first_degree_term_does_not_display_variable_power(self):
        """Test that the first-degree-term ends with 'x' and not 'x^1'."""
        coeffs = [2, 0]
        expect = "Polynomial(2, 0)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_constant_does_not_display_variable_power(self):
        """Test that the constant term does not end with 'x' or 'x^0'."""
        coeffs = [2]
        expect = "Polynomial(2)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_unit_coefficient_is_not_displayed_if_not_constant(self):
        """Test that coefficients with value 1 are not displayed."""
        coeffs = [1, 2]
        expect = "Polynomial(1, 2)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_unit_coefficient_is_displayed_if_constant_term(self):
        """Test that constants with value 1 are displayed properly."""
        coeffs = [1, 0, 1]
        expect = "Polynomial(1, 0, 1)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_negative_coefficient_at_the_beggining_puts_short_minus(self):
        """Test that there is a short minus sign if the first coeff is < 0."""
        coeffs = [-1, 0, 2]
        expect = "Polynomial(-1, 0, 2)"

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_canonical_string(self):
        """Test with string coefficients."""
        coeffs = "abcdef"
        expect = """Polynomial('a', 'b', 'c', 'd', 'e', 'f')"""

        r = repr(Polynomial(coeffs))

        self.assertEqual(expect, r)

    def test_zero_polynomial_repr_is_constructor(self):
        """Test that the 0-polynomial is just '0'."""
        r = repr(ZeroPolynomial())

        self.assertEqual("ZeroPolynomial()", r)

    def test_monomial_repr(self):
        expect = "Monomial(1, 2)"

        r = repr(Monomial(1, 2))

        self.assertEqual(expect, r)

    def test_constant_repr(self):
        expect = "Constant(5)"

        r = repr(Constant(5))

        self.assertEqual(expect, r)

    def test_linear_binomial(self):
        expect = "LinearBinomial(5, 2)"

        r = repr(LinearBinomial(5, 2))

        self.assertEqual(expect, r)

    def test_quadratic_trinomial(self):
        expect = "QuadraticBinomial(1, -4, 4)"

        r = repr(QuadraticTrinomial(1, -4, 4))

        self.assertEqual(expect, r)


if __name__ == '__main__':
    unittest.main()
