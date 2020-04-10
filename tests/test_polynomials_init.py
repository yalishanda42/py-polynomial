"""Unit-testing module defining polynomials initialization test cases."""

import unittest
from polynomial import (
    Polynomial,
    Constant,
    Monomial,
    Binomial,
    LinearBinomial,
    Trinomial,
    QuadraticTrinomial,
)


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

    def test_binomial_init_from_monomials(self):
        """Test that a binomial is successfully initialized from monomials."""
        m1 = Monomial(3, 3)
        m2 = Monomial(4, 4)
        t1 = (3, 3)
        t2 = (4, 4)
        expected = Polynomial([m1, m2], from_monomials=True)

        b1 = Binomial(m1, m2)
        b2 = Binomial(t1, t2)

        self.assertEqual(expected, b1)
        self.assertEqual(expected, b2)
        self.assertEqual(b1, b2)

    def test_binomial_default_init(self):
        """Test that the default binomial is 'x^2 + x'."""
        expected = Polynomial(1, 1, 0)

        b = Binomial()

        self.assertEqual(expected, b)

    def test_linear_binomial_init(self):
        """Test that a linear binomial is successfully initialized."""
        a, b = 6, 9
        expected = Polynomial(a, b)

        lb = LinearBinomial(a, b)

        self.assertEqual(expected, lb)

    def test_linear_binomial_default_init(self):
        """Test that the default linear binomial is 'x + 1'."""
        expected = Polynomial(1, 1)

        b = LinearBinomial()

        self.assertEqual(expected, b)

    def test_trinomial_init_from_monomials(self):
        """Test that a trinomial is successfully initialized from monomials."""
        m1 = Monomial(3, 3)
        m2 = Monomial(4, 4)
        m3 = Monomial(5, 5)
        expected = Polynomial([m1, m2, m3], from_monomials=True)

        t = Trinomial(m1, m2, m3)

        self.assertEqual(expected, t)

    def test_trinomial_default_init(self):
        """Test that the default trinomial is 'x^3 + x^2 + x'."""
        expected = Polynomial(1, 1, 1, 0)

        t = Trinomial()

        self.assertEqual(expected, t)

    def test_quadratic_trinomial_init(self):
        """Test that a quadratic trinomial is successfully initialized."""
        a, b, c = 2, 3, 4
        expected = Polynomial(a, b, c)

        qt = QuadraticTrinomial(a, b, c)

        self.assertEqual(expected, qt)

    def test_quadratic_trinomial_default_init(self):
        """Test that the default quadratic trinomial is 'x^2 + x + 1'."""
        expected = Polynomial(1, 1, 1)

        qt = QuadraticTrinomial()

        self.assertEqual(expected, qt)

    def test_linear_binomial_fails_leading_zero(self):
        """Test that LinearBinomial(0, ?) raises a ValueError."""
        self.assertRaises(ValueError, LinearBinomial, 0, 1)

    def test_quadratic_trinomial_fails_leading_zero(self):
        """Test that QuadraticTrinomial(0, ?) raises a ValueError."""
        self.assertRaises(ValueError, QuadraticTrinomial, 0, 1)

    def test_monomial_degree_positive_int(self):
        """Test that monomial only accepts a positive int."""
        self.assertRaises(ValueError, Monomial, 1, -1)
        self.assertRaises(ValueError, Monomial, 1, 1.2)

    def test_polynomial_with_non_monomial_terms(self):
        """Test that Polynomial from monomials with > 2 tuples fails."""
        self.assertRaises(
            TypeError,
            Polynomial,
            [(1, 2, 3)],
            from_monomials=True
        )


if __name__ == '__main__':
    unittest.main()
