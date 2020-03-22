"""Unit-testing module for testing various polynomial operations."""

import unittest
from polynomial import (
    Constant,
    Monomial,
    Polynomial,
    ZeroPolynomial,
)
from math import inf


class TestPolynomialsOperations(unittest.TestCase):
    """Defines polynomial operations test cases."""

    def test_derivative_general_case_correct(self):
        """Test that the derived polynomial is generally correct."""
        p = Polynomial(1, 2, 3)
        expect = Polynomial(2, 2)

        result = p.derivative

        self._assert_polynomials_are_the_same(expect, result)

    def test_derivative_of_zero_is_zero(self):
        """Test that the derivative of the zero polynomial is the zero poly."""
        p = ZeroPolynomial()
        expect = ZeroPolynomial()

        result = p.derivative

        self._assert_polynomials_are_the_same(expect, result)

    def test_derivative_of_constant_equals_zero(self):
        """Test that the derivative of a Constant is equal to the zero poly."""
        c = Constant(1)
        expect = ZeroPolynomial()

        result = c.derivative

        self.assertEqual(expect, result)

    def test_sub_isub_polynomials_result_same(self):
        """Test that subtracting two polynomials works correctly."""
        p = Polynomial(3, 3, 3)
        sub = Polynomial(2, 2, 2)
        expect = Polynomial(1, 1, 1)

        result = p - sub
        p -= sub

        self._assert_polynomials_are_the_same(expect, result)
        self._assert_polynomials_are_the_same(expect, p)

    def test_sub_rsub_isub_constant_result_same(self):
        """Test that subtracting polynomials and constants works correctly."""
        p = Polynomial(3, 3, 3)
        c = 2
        expect1 = Polynomial(3, 3, 1)
        expect2 = Polynomial(-3, -3, -1)

        result1 = p - c
        result2 = c - p
        p -= c

        self._assert_polynomials_are_the_same(expect1, result1)
        self._assert_polynomials_are_the_same(expect2, result2)
        self._assert_polynomials_are_the_same(expect1, p)

    def test_subtraction_to_zero_properties_change_accordingly(self):
        """Test that the properties change when subtraction sets self to 0."""
        coeffs = [1, 2]
        p1 = Polynomial(coeffs)
        p2 = Polynomial(coeffs)
        expect = Polynomial()

        a = p1 - p2

        self._assert_polynomials_are_the_same(expect, a)
        self.assertEqual(ZeroPolynomial(), a)

    def test_setitem_lowers_degree_correctly(self):
        """Test that the properties change when the leading term is zeroed."""
        p = Polynomial(1, 2, 3)
        expect = Polynomial(2, 3)

        p[p.degree] = 0

        self._assert_polynomials_are_the_same(expect, p)

    def test_sub_lowers_degree_correctly(self):
        """Test that the properties change when the degree is reduced."""
        p = Polynomial(1, 2, 3)
        sub = Polynomial(1, 1, 1)
        expect = Polynomial(1, 2)

        p -= sub

        self._assert_polynomials_are_the_same(expect, p)

    def test_constant_can_be_cast_to_int(self):
        """Test Constant(number) == number where type(number) is int."""
        number = 69
        c = Constant(number)

        self.assertEqual(number, int(c))
        self.assertEqual(number, c)
        self.assertEqual(number, c.const)

    def test_constant_can_be_cast_to_float(self):
        """Test Constant(number) == number where type(number) is float."""
        number = 4.20
        c = Constant(number)

        self.assertEqual(number, float(c))
        self.assertEqual(number, c)
        self.assertEqual(number, c.const)

    def test_constant_can_be_cast_to_complex(self):
        """Test Constant(number) == number where type(number) is complex."""
        number = 4.20 - 69j
        c = Constant(number)

        self.assertEqual(number, complex(c))
        self.assertEqual(number, c)
        self.assertEqual(number, c.const)

    def test_zero_polynomial_equals_zero(self):
        """Test that ZeroPolynomial() == Polynomial() == Constant(0) == 0."""
        z = ZeroPolynomial()
        p = Polynomial()
        c = Constant(0)

        self.assertTrue(c == p == z == 0)
        self.assertTrue(c == p == z == 0.0)
        self.assertTrue(c == p == z == 0.0j)

    def test_degree_of_zero_is_minus_infinity(self):
        """Test that the degree of the zero poly is minus infinity."""
        z = ZeroPolynomial()
        p = Polynomial()
        c = Constant(0)
        expect = -inf

        self.assertEqual(expect, z.degree)
        self.assertEqual(expect, p.degree)
        self.assertEqual(expect, c.degree)

    def _assert_polynomials_are_the_same(self, p1, p2):
        self.assertEqual(repr(p1), repr(p2))
        self.assertEqual(str(p1), str(p2))
        self.assertEqual(p1, p2)
        self.assertEqual(p1.terms, p2.terms)
        self.assertEqual(p1.monomials, p2.monomials)
        self.assertEqual(p1.degree, p2.degree)

    def test_add_iadd_polynomials_result_same(self):
        """Test that adding two polynomials works correctly."""
        p1 = Polynomial(1, 1, 1)
        p2 = Polynomial(2, 2)
        expect = Polynomial(1, 3, 3)

        result1 = p1 + p2
        result2 = p2 + p1
        p1 += p2

        self._assert_polynomials_are_the_same(expect, result1)
        self._assert_polynomials_are_the_same(expect, result2)
        self._assert_polynomials_are_the_same(expect, p1)

    def test_add_radd_iadd_constant_result_same(self):
        """Test that adding polynomials and constants works correctly."""
        p = Polynomial(1, 1, 1)
        c = 2
        expect = Polynomial(1, 1, 3)

        result1 = p + c
        result2 = c + p
        p += c

        self._assert_polynomials_are_the_same(expect, result1)
        self._assert_polynomials_are_the_same(expect, result2)
        self._assert_polynomials_are_the_same(expect, p)

    def test_add_zero_result_the_same(self):
        """Test that adding zero does not change the polynomial."""
        coeffs = [1, 1, 1]
        p1 = Polynomial(coeffs)
        p2 = Polynomial(coeffs)
        p3 = Polynomial(coeffs)
        p4 = Polynomial(coeffs)
        z = ZeroPolynomial()
        expect = Polynomial(coeffs)

        result1 = p1 + z
        result2 = z + p1
        p1 += z
        p2 += 0
        p3 += 0.0
        p4 += 0j

        self._assert_polynomials_are_the_same(p1, result1)
        self._assert_polynomials_are_the_same(p1, result2)
        self._assert_polynomials_are_the_same(expect, p1)
        self._assert_polynomials_are_the_same(expect, p2)
        self._assert_polynomials_are_the_same(expect, p3)
        self._assert_polynomials_are_the_same(expect, p4)

    def test_mul_imul_polynomials_result_same(self):
        """Test that multiplying two polynomials works correctly."""
        p1 = Polynomial(1, 2, 3)
        p2 = Polynomial(1, 2, 3)
        expect = Polynomial(1, 4, 10, 12, 9)

        result1 = p1 * p2
        result2 = p2 * p1
        p1 *= p2

        self._assert_polynomials_are_the_same(expect, result1)
        self._assert_polynomials_are_the_same(expect, result2)
        self._assert_polynomials_are_the_same(expect, p1)

    def test_mul_rmul_imul_constant_result_same(self):
        """Test that multiplying polynomials and constants works correctly."""
        p = Polynomial(1, 1, 1)
        c = 10
        expect = Polynomial(10, 10, 10)

        result1 = p * c
        result2 = c * p
        p *= c

        self._assert_polynomials_are_the_same(expect, result1)
        self._assert_polynomials_are_the_same(expect, result2)
        self._assert_polynomials_are_the_same(expect, p)

    def test_mul_zero_result_zero(self):
        """Test that multiplying by zero equals zero."""
        coeffs = [1, 1, 1]
        p1 = Polynomial(coeffs)
        p2 = Polynomial(coeffs)
        p3 = Polynomial(coeffs)
        p4 = Polynomial(coeffs)
        z0 = ZeroPolynomial()
        z1 = Polynomial()

        # Multiplication like this can downcast a Polynomial
        # to a ZeroPolynomial.
        result1 = p1 * z0
        result2 = z0 * p2
        # Inplace multiplication will not downcast a Polynomial.
        # It may however upcast to a Polynomial if the operands
        # are not compatible.
        p1 *= z0
        p2 *= 0
        p3 *= 0.0
        p4 *= 0j

        self._assert_polynomials_are_the_same(z0, result1)
        self._assert_polynomials_are_the_same(z0, result2)
        self._assert_polynomials_are_the_same(z1, p1)
        self._assert_polynomials_are_the_same(z1, p2)
        self._assert_polynomials_are_the_same(z1, p3)
        self._assert_polynomials_are_the_same(z1, p4)

    # Note that for the division tests, we don't use
    # _assert_polynomials_are_the_same because an integer divided by an
    # integer results in a float.

    def test_divmod_same_polynomial(self):
        """Test that divmodding two identical polynomials works correctly."""
        p1 = Polynomial(1, 4, 4)
        p2 = Polynomial(1, 4, 4)

        p3, remainder = divmod(p1, p2)

        self.assertEqual(p3, Polynomial(1))
        self.assertEqual(remainder, Polynomial())

    def test_divmod_no_remainder(self):
        """Test that divmodding a polynomial with a factor works correctly."""
        p1 = Polynomial(1, 4, 4)
        p2 = Polynomial(1, 2)

        p3, remainder = divmod(p1, p2)

        self.assertEqual(p3, Polynomial(1, 2))
        self.assertEqual(remainder, Polynomial())

    def test_divmod_remainder_exists(self):
        """Test that divmodding with a non-zero remainder works correctly."""
        p1 = Polynomial(1, 2, 3)
        p2 = Polynomial(1, 2)

        p3, remainder = divmod(p1, p2)

        self.assertEqual(p3, Polynomial(1, 0))
        self.assertEqual(remainder, Polynomial(3))

    def test_divmod_against_constant(self):
        """Test that Polynomial(*) divmod a Constant leaves no remainder."""
        p1 = Polynomial(1, 2, 3)
        p2 = Constant(5)

        p3, remainder = divmod(p1, p2)

        self.assertEqual(p3, Polynomial(1/5, 2/5, 3/5))
        self.assertEqual(remainder, Polynomial())

    def test_divmod_against_monomial(self):
        """Test that divmodding by a larger monomial leaves original val."""
        p1 = Polynomial(1, 2, 3)
        p2 = Monomial(1, 10)

        p3, remainder = divmod(p1, p2)

        self.assertEqual(p3, Polynomial())
        self.assertEqual(p1, remainder)

    def test_inplace_floor_div(self):
        """Test that a //= x behaves as expected."""
        p1 = Polynomial(1, 4, 4)
        p2 = Polynomial(1, 2)

        p1 //= p2

        self.assertEqual(p1, Polynomial(1, 2))

    def test_floor_div(self):
        """Test that a = b // x behaves as expected."""
        p1 = Polynomial(1, 4, 4)
        p2 = Polynomial(1, 2)

        p3 = p1 // p2

        self.assertEqual(p3, Polynomial(1, 2))

if __name__ == '__main__':
    unittest.main()
