"""Unit-testing module defining polynomials repr test cases."""

import unittest
from polynomial.polynomials import *


class TestPolynomialsRepr(unittest.TestCase):
    """Defines polynomials repr test cases."""

    def test_repr_equals_str(self):
        """Test that the output of repr is equal to that of str."""
        coeffs = [0, 0, -2, 0, 3, 4.6, -234j, (69+420j), -1, 1337]
        p = Polynomial(coeffs)

        r = repr(p)
        s = str(p)

        self.assertEqual(r, s)

    def test_smoke(self):
        """Perform a test case containing all possible complicated inputs."""
        coeffs = [0, 0, -2, 0, 4.6666666666, -(69+420j), -1, 1337]
        expect = "-2x^5 + 4.6666666666x^3 + (-69-420j)x^2 - x + 1337"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_negative_coefficient_replaces_plus_with_minus(self):
        """Test that negative coefficients replace the + between the terms."""
        coeffs = [3, -2, 0, 0]
        expect = "3x^3 - 2x^2"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_first_degree_term_does_not_display_variable_power(self):
        """Test that the first-degree-term ends with 'x' and not 'x^1'."""
        coeffs = [2, 0]
        expect = "2x"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_constant_does_not_display_variable_power(self):
        """Test that the constant term does not end with 'x' or 'x^0'."""
        coeffs = [2]
        expect = "2"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_unit_coefficient_is_not_displayed_if_not_constant(self):
        """Test that coefficients with value 1 are not displayed."""
        coeffs = [1, 2]
        expect = "x + 2"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_unit_coefficient_is_displayed_if_constant_term(self):
        """Test that constants with value 1 are displayed properly."""
        coeffs = [1, 0, 1]
        expect = "x^2 + 1"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_negative_coefficient_at_the_beggining_puts_short_minus(self):
        """Test that there is a short minus sign if the first coeff is < 0."""
        coeffs = [-1, 0, 2]
        expect = "-x^2 + 2"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_canonical_string_repr(self):
        """Test a string coefficients repr."""
        coeffs = "abcdef"
        expect = "ax^5 + bx^4 + cx^3 + dx^2 + ex + f"

        r = repr(Polynomial(coeffs))

        self.assertEqual(r, expect)

    def test_zero_polynomial_repr_is_zero(self):
        """Test that the 0-polynomial repr is just '0'."""
        r = repr(ZeroPolynomial())

        self.assertEqual(r, "0")


if __name__ == '__main__':
    unittest.main()
