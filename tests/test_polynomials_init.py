"""Unit-testing module defining polynomials initialization test cases."""

import unittest
from polynomial import Polynomial, Constant, Monomial


class TestPolynomialsInit(unittest.TestCase):
    """Defines polynomials initialization test cases."""

    def test_polynomial_from_int_list_same_as_from_int_args(self):
        """Test that int coefficients list is the same as int *args."""
        coeffs = list(range(10))

        p1 = Polynomial(coeffs)
        p2 = Polynomial(*coeffs)

        self.assertEqual(p1, p2)

    def test_polynomial_from_float_list_same_as_from_float_args(self):
        """Test that float coefficients list is the same as float *args."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        p1 = Polynomial(coeffs)
        p2 = Polynomial(*coeffs)

        self.assertEqual(p1, p2)

    def test_polynomial_from_complex_list_same_as_from_complex_args(self):
        """Test that complex coefficients list is the same as complex *args."""
        coeffs = [1j, 2j, 3j, 4j, 5j]

        p1 = Polynomial(coeffs)
        p2 = Polynomial(*coeffs)

        self.assertEqual(p1, p2)

    def test_polynomial_from_string_the_same_as_string_args(self):
        """Test that char coefficients list is the same as string *args."""
        coeffs = "abcdefghijklmnopqrstuvwxyz"

        p1 = Polynomial(coeffs)
        p2 = Polynomial(*coeffs)

        self.assertEqual(p1, p2)

    def test_leading_zeroes_are_removed(self):
        """Test leading terms with coefficients equal to zero are removed."""
        p1 = Polynomial(1, 2, 3, 0)
        p2 = Polynomial(0, 0, 0, 1, 2, 3, 0)

        self.assertEqual(repr(p1), repr(p2))
        self.assertEqual(str(p1), str(p2))
        self.assertEqual(p1, p2)

    def test_default_monomial_is_x(self):
        """Test that the default Monomial is 'x'."""
        m = Monomial()
        expect = Monomial(1, 1)

        self.assertEqual(repr(expect), repr(m))
        self.assertEqual(str(expect), str(m))
        self.assertEqual(expect, m)

    def test_default_constant_is_one(self):
        """Test that the default Constant is '1'."""
        c = Constant()
        expect = Constant(1)

        self.assertEqual(expect, c)


if __name__ == '__main__':
    unittest.main()
