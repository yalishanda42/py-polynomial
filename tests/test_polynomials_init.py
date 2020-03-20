"""Unit-testing module defining polynomials initialization test cases."""

import unittest
from polynomial import Polynomial


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

    def test_polynomial_degree_goes_down_when_zeroed(self):
        """Test that polynomial degree changes when subtraction sets leading term to zero."""
        coeffs = [1, 2]
        p1 = Polynomial(coeffs)
        p2 = Polynomial(coeffs)

        a = p1 - p2

        self.assertEquals(a.degree, 0)

    def test_polynomial_degree_goes_down_on_setitem(self):
        """Test that polynomial degree changes when the leading term is set to zero."""
        coeffs = [1, 2, 3]
        p = Polynomial(coeffs)

        p[2] = 0

        self.assertEquals(a.degree, 1)

if __name__ == '__main__':
    unittest.main()
