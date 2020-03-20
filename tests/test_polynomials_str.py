"""Unit-testing module defining polynomials __str__ test cases."""

import unittest
from polynomial import Monomial, Polynomial, ZeroPolynomial


class TestPolynomialsStr(unittest.TestCase):
    """Defines polynomials __str__ test cases."""

    def test_smoke(self):
        """Perform a test case containing all possible complicated inputs."""
        coeffs = [0, 0, -2, 0, 4.6666666666, -(69+420j), -1, 1337]
        expect = "-2x^5 + 4.6666666666x^3 + (-69-420j)x^2 - x + 1337"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_negative_coefficient_replaces_plus_with_minus(self):
        """Test that negative coefficients replace the + between the terms."""
        coeffs = [3, -2, 0, 0]
        expect = "3x^3 - 2x^2"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_first_degree_term_does_not_display_variable_power(self):
        """Test that the first-degree-term ends with 'x' and not 'x^1'."""
        coeffs = [2, 0]
        expect = "2x"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_constant_does_not_display_variable_power(self):
        """Test that the constant term does not end with 'x' or 'x^0'."""
        coeffs = [2]
        expect = "2"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_unit_coefficient_is_not_displayed_if_not_constant(self):
        """Test that coefficients with value 1 are not displayed."""
        coeffs = [1, 2]
        expect = "x + 2"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_unit_coefficient_is_displayed_if_constant_term(self):
        """Test that constants with value 1 are displayed properly."""
        coeffs = [1, 0, 1]
        expect = "x^2 + 1"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_negative_coefficient_at_the_beggining_puts_short_minus(self):
        """Test that there is a short minus sign if the first coeff is < 0."""
        coeffs = [-1, 0, 2]
        expect = "-x^2 + 2"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_power_starting_in_one_kept_if_power_is_not_one(self):
        """Test x^10 is not converted to x and properly appears as x^10."""
        expect = "5x^10"

        s = str(Monomial(5, 10))

        self.assertEqual(expect, s)

    def test_canonical_string(self):
        """Test with string coefficients."""
        coeffs = "abcdef"
        expect = "ax^5 + bx^4 + cx^3 + dx^2 + ex + f"

        s = str(Polynomial(coeffs))

        self.assertEqual(expect, s)

    def test_zero_polynomial_str_is_zero(self):
        """Test that the 0-polynomial is just '0'."""
        s = str(ZeroPolynomial())

        self.assertEqual("0", s)


if __name__ == '__main__':
    unittest.main()
