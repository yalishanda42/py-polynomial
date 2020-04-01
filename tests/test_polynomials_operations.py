"""Unit-testing module for testing various polynomial operations."""

import unittest
from math import inf

from polynomial import (
    Constant,
    FrozenPolynomial,
    Monomial,
    Polynomial,
    ZeroPolynomial,
    LinearBinomial,
    QuadraticTrinomial,
)
from polynomial.frozen import Freezable


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
        z1 = Polynomial.zero_instance()

        # Multiplication will return the most permissive
        # class (eg. which allows the most mutability).
        result1 = p1 * z0
        result2 = z0 * p2
        p1 *= z0
        p2 *= 0
        p3 *= 0.0
        p4 *= 0j

        self._assert_polynomials_are_the_same(z1, result1)
        self._assert_polynomials_are_the_same(z1, result2)
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

    def test_inplace_mod(self):
        """Test that a %= x behaves as expected."""
        p1 = Polynomial(1, 2, 3)
        p2 = Polynomial(1, 2)
        expect = Polynomial(3)

        p1 %= p2

        self.assertEqual(expect, p1)

    def test_mod(self):
        """Test that a = b % x behaves as expected."""
        p1 = Polynomial(1, 2, 3)
        p2 = Polynomial(1, 2)
        expect = Polynomial(3)

        p3 = p1 % p2

        self.assertEqual(expect, p3)

    def test_eq_neq_opposite_when_equals(self):
        """Tests that equal polynomials are truly equal."""
        self.assertEqual(Polynomial(1, 2, 3), Polynomial(1, 2, 3))
        self.assertFalse(Polynomial(1, 2, 3) != Polynomial(1, 2, 3))

    def test_eq_neq_opposite_when_one_is_zero(self):
        """Tests that nonzero polynomial != 0."""
        self.assertNotEqual(Polynomial(1, 2), 0)
        self.assertFalse(Polynomial(1, 2) == 0)

    def test_eq_neq_opposite_when_both_are_zero(self):
        """Tests that zero polynomial == 0."""
        self.assertEqual(Polynomial(), 0)
        self.assertFalse(Polynomial() != 0)

    def test_in_different_polynomials(self):
        """Tests that a polynomial is in another polynomial."""
        p1 = Polynomial(1, 2, 3)
        p2 = Polynomial(6, 5, 4, 1, 2, 3)
        self.assertIn(p1, p2)
        self.assertNotIn(p2, p1)

    def test_membership_with_all_legal_types(self):
        """Test that all valid types are handled in membership check."""
        terms = [(1, 2), (2, 1), (3, 0)]
        p = Polynomial(1, 2, 3)

        # Test that single tuples work.
        self.assertIn(terms, p)
        self.assertIn(terms[0], p)
        self.assertIn(terms[1], p)
        self.assertIn(terms[2], p)

        # Test that partial matching works as well.
        self.assertIn(terms[:2], p)
        self.assertIn(terms[1:], p)
        self.assertIn([terms[0], terms[2]], p)

        # Test that sets and polynomials are correctly handled.
        self.assertIn(set(terms), p)
        self.assertIn(Polynomial(terms, from_monomials=True), p)

    def test_membership_false_on_partial_match(self):
        """Tests that membership is only true if all elements match."""
        p1 = Polynomial(1, 2, 3)
        p2 = Polynomial(1, 2, 4)

        self.assertNotIn(p1, p2)
        self.assertNotIn(p2, p1)

    def test_membership_matches_degrees(self):
        """Test that degrees don't change matching behaviour."""
        p1 = Polynomial(1, 2, 3)
        p2 = Polynomial(1, 2, 3, 0)

        self.assertNotIn(p1, p2)
        self.assertNotIn(p2, p1)

    def test_nth_derivative(self):
        """Test that the nth derivative is correct for various n."""
        p = Polynomial(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        pd = p
        for i in range(10):
            result = p.nth_derivative(i)
            self._assert_polynomials_are_the_same(pd, result)
            pd = pd.derivative

        result = p.nth_derivative(10)
        self._assert_polynomials_are_the_same(pd, result)

    def test_zero_raises_err(self):
        """Test that the ZeroPolynomial raises errors when setting values."""
        z = ZeroPolynomial()

        self.assertRaises(AttributeError, z.__setattr__, "x", 5)
        self.assertRaises(AttributeError, z.__setitem__, 0, 5)

    def test_frozen_polynomial_raises_err(self):
        f = FrozenPolynomial(1, 2, 3)

        self.assertRaises(AttributeError, f.__setattr__, "x", 5)
        self.assertRaises(AttributeError, f.__setitem__, 0, 5)
        self.assertRaises(AttributeError, f.__imul__, 5)

    def test_pow_monomial(self):
        """Test power against various Monomial subclasses."""
        c = Constant(5)
        ec = Constant(25)
        m = Monomial(5, 10)
        em = Monomial(25, 20)
        z = ZeroPolynomial()
        ez = ZeroPolynomial()

        self._assert_polynomials_are_the_same(ec, c ** 2)
        self._assert_polynomials_are_the_same(em, m ** 2)
        self._assert_polynomials_are_the_same(ez, z ** 2)

    def test_pow_zero_case(self):
        """Test pow ** 0 returns 1."""
        one = Constant(1)
        m_one = Monomial(1, 0)
        c = Constant(5)
        m = Monomial(5, 10)
        z = ZeroPolynomial()
        p = Polynomial(1, 2, 3)

        self._assert_polynomials_are_the_same(one, c ** 0)
        self._assert_polynomials_are_the_same(m_one, m ** 0)
        self._assert_polynomials_are_the_same(one, z ** 0)
        self._assert_polynomials_are_the_same(one, p ** 0)

    def test_pow_one_case(self):
        """Tests pow ** 1 returns a copy of the polynomial."""
        c = Constant(5)
        m = Monomial(5, 10)
        z = ZeroPolynomial()
        p = Polynomial(1, 2, 3)

        self._assert_polynomials_are_the_same(c, c ** 1)
        self._assert_polynomials_are_the_same(m, m ** 1)
        self._assert_polynomials_are_the_same(z, z ** 1)
        self._assert_polynomials_are_the_same(p, p ** 1)

    def test_pow_two_case(self):
        """Test pow ** 2."""
        c = Constant(5)
        m = Monomial(5, 10)
        z = ZeroPolynomial()
        p = Polynomial(1, 2, 3)

        self._assert_polynomials_are_the_same(c * c, c ** 2)
        self._assert_polynomials_are_the_same(m * m, m ** 2)
        self._assert_polynomials_are_the_same(z * z, z ** 2)
        self._assert_polynomials_are_the_same(p * p, p ** 2)

    def test_general_pow(self):
        """Check that pow returns the expected value."""
        p = Polynomial(1, 2)
        expected = Polynomial(p)

        for i in range(1, 10):
            res = p ** i
            self._assert_polynomials_are_the_same(expected, res)
            self.assertIsNot(p, res)
            expected *= p

    def test_setting_zero_const_raises(self):
        """Test that doing ZeroPolynomial.const = x raises an error."""
        z = ZeroPolynomial()
        self.assertRaises(AttributeError, setattr, z, "const", None)

    def test_zero_const_is_zero(self):
        """Test that ZeroPolynomial.const is always 0."""
        self.assertEqual(0, ZeroPolynomial().const)

    def test_shift_zero(self):
        """Test that any shift by 0 does nothing."""
        coeffs = [1, 1, 1]
        p = Polynomial(coeffs)
        p1 = Polynomial(coeffs)
        p2 = Polynomial(coeffs)
        p3 = Polynomial(coeffs)
        p4 = Polynomial(coeffs)

        p1 <<= 0
        p2 >>= 0
        p3 = p3 << 0
        p4 = p4 >> 0

        self._assert_polynomials_are_the_same(p, p1)
        self._assert_polynomials_are_the_same(p, p2)
        self._assert_polynomials_are_the_same(p, p3)
        self._assert_polynomials_are_the_same(p, p4)

    def test_lshift_general(self):
        """Test that lshift behaves correctly for various inputs."""
        coeffs = [1, 3, 5]
        p = Polynomial(coeffs)
        p1 = p << 3
        p2 = p << 1
        p3 = p << -1

        self._assert_polynomials_are_the_same(Polynomial(1, 3, 5, 0, 0, 0), p1)
        self._assert_polynomials_are_the_same(Polynomial(1, 3, 5, 0), p2)
        self._assert_polynomials_are_the_same(Polynomial(1, 3), p3)

    def test_rshift_general(self):
        """Test that rshift behaves correctly for various inputs."""
        coeffs = [1, 3, 5]
        p = Polynomial(coeffs)
        p1 = p >> 1
        p2 = p >> 3
        p3 = p >> -1

        self._assert_polynomials_are_the_same(Polynomial(1, 3), p1)
        self._assert_polynomials_are_the_same(Polynomial(), p2)
        self._assert_polynomials_are_the_same(Polynomial(1, 3, 5, 0), p3)

    def test_lshift_monomial(self):
        """Test that lshift on a monomial behaves correctly."""
        m1 = Monomial(1, 5) << 10
        m2 = Monomial(1, 15) << -10
        m3 = Constant(5) << 10

        self._assert_polynomials_are_the_same(Monomial(1, 15), m1)
        self._assert_polynomials_are_the_same(Monomial(1, 5), m2)
        self._assert_polynomials_are_the_same(Monomial(5, 10), m3)

    def test_rshift_monomial(self):
        """Test that rshift on a monomial behaves correctly."""
        m1 = Monomial(1, 5) >> -10
        m2 = Monomial(1, 15) >> 10
        self._assert_polynomials_are_the_same(Monomial(1, 15), m1)
        self._assert_polynomials_are_the_same(Monomial(1, 5), m2)

    def test_shift_polynomial_past_end(self):
        """Test that shifting a polynomial beyond 0 yields 0."""
        p1 = Polynomial(*range(1, 11)) >> 15
        p2 = Polynomial(*range(1, 11)) << -15

        self._assert_polynomials_are_the_same(Polynomial(), p1)
        self._assert_polynomials_are_the_same(Polynomial(), p2)

    def test_shift_monomial_past_end(self):
        """Test that shifting a Monomial beyond 0 yields 0."""
        m1 = Monomial(1, 10) >> 15
        m2 = Monomial(1, 10) << -15
        m0 = Monomial.zero_instance()

        self._assert_polynomials_are_the_same(m0, m1)
        self._assert_polynomials_are_the_same(m0, m2)

    def test_shifting_constant_not_inplace(self):
        """Test that constant/zero objects are not modified in place."""
        c = Constant(5)
        c1 = c
        c1 <<= 5
        z = ZeroPolynomial()
        z1 = z
        z1 <<= 5

        self.assertIsNot(c, c1)
        self.assertIsNot(z, z1)

    def test_constant_constant_mul_yields_constant(self):
        """Test that Constant * Constant yields Constant."""
        c = Constant(5)
        expected = Constant(25)
        self._assert_polynomials_are_the_same(expected, c * c)

    def test_positive_discriminant_roots(self):
        """Test the real and complex roots and factors for D > 0."""
        qt = QuadraticTrinomial(1, -5, 6)
        exp_discri = 1
        exp_real_roots = (3, 2)
        exp_complex_roots = (3, 2)
        exp_real_factors = (1, LinearBinomial(1, -3), LinearBinomial(1, -2))
        exp_complex_factors = exp_real_factors

        res_discri = qt.discriminant
        res_real_roots = qt.real_roots
        res_complex_roots = qt.complex_roots
        res_real_factors = qt.real_factors
        res_complex_factors = qt.complex_factors

        self.assertEqual(exp_discri, res_discri)
        self.assertEqual(exp_real_roots, res_real_roots)
        self.assertEqual(exp_complex_roots, res_complex_roots)
        self.assertEqual(exp_real_factors, res_real_factors)
        self.assertEqual(exp_complex_factors, res_complex_factors)

    def test_zero_discriminant_roots(self):
        """Test the real and complex roots and factors for D == 0."""
        qt = QuadraticTrinomial(1, 4, 4)
        exp_discri = 0
        exp_real_roots = (-2, -2)
        exp_complex_roots = (-2, -2)
        exp_real_factors = (1, LinearBinomial(1, 2), LinearBinomial(1, 2))
        exp_complex_factors = exp_real_factors

        res_discri = qt.discriminant
        res_real_roots = qt.real_roots
        res_complex_roots = qt.complex_roots
        res_real_factors = qt.real_factors
        res_complex_factors = qt.complex_factors

        self.assertEqual(exp_discri, res_discri)
        self.assertEqual(exp_real_roots, res_real_roots)
        self.assertEqual(exp_complex_roots, res_complex_roots)
        self.assertEqual(exp_real_factors, res_real_factors)
        self.assertEqual(exp_complex_factors, res_complex_factors)

    def test_negative_discriminant_roots(self):
        """Test the real and complex roots and factors for D < 0."""
        qt = QuadraticTrinomial(1, 2, 5)
        exp_discri = -16
        exp_real_roots = ()
        exp_complex_roots = (-1 + 2j, -1 - 2j)
        exp_real_factors = (qt,)
        exp_complex_factors = (
            1,
            LinearBinomial(1, 1 - 2j),
            LinearBinomial(1, 1 + 2j)
        )

        res_discri = qt.discriminant
        res_real_roots = qt.real_roots
        res_complex_roots = qt.complex_roots
        res_real_factors = qt.real_factors
        res_complex_factors = qt.complex_factors

        self.assertEqual(exp_discri, res_discri)
        self.assertEqual(exp_real_roots, res_real_roots)
        self.assertEqual(exp_complex_roots, res_complex_roots)
        self.assertEqual(exp_real_factors, res_real_factors)
        self.assertEqual(exp_complex_factors, res_complex_factors)

    def test_zero_instance_mutable(self):
        """Test that zero instances are mutable"""
        zp = Polynomial.zero_instance()
        zm = Monomial.zero_instance()
        zc = Constant.zero_instance()
        zp.a = 1
        zm.a = 1
        zc.const = 1

        self._assert_polynomials_are_the_same(Polynomial(1), zp)
        self._assert_polynomials_are_the_same(Monomial(1, 0), zm)
        self._assert_polynomials_are_the_same(Constant(1), zc)

    def test_zero_polynomial_conversions(self):
        """Test that converting ZeroPolynomial to numerical types yields 0."""
        z = ZeroPolynomial()
        self.assertEqual(0, int(z))
        self.assertEqual(0.0, float(z))
        self.assertEqual(0j, complex(z))

    def test_mul_zero_poly_returns_most_permissive(self):
        """Test that multiplication never reduces permissiveness."""
        a = Polynomial(1, 2, 3) * ZeroPolynomial()
        b = Monomial(1, 2) * ZeroPolynomial()
        c = Constant(5) * ZeroPolynomial()
        d = ZeroPolynomial() * ZeroPolynomial()

        self.assertIsInstance(a, Polynomial)
        self.assertIsInstance(b, Monomial)
        self.assertIsInstance(c, Constant)
        self.assertIsInstance(d, ZeroPolynomial)

    def test_mul_constant_returns_most_permissive(self):
        """Test that multiplication never reduces permissiveness."""
        a = Polynomial(1, 2, 3) * Constant(3)
        b = Monomial(1, 3) * Constant(1)
        c = Constant(5) * Constant(4)
        d = ZeroPolynomial() * Constant(2)

        self.assertIsInstance(a, Polynomial)
        self.assertIsInstance(b, Monomial)
        self.assertIsInstance(c, Constant)
        self.assertIsInstance(d, Constant)

    def test_mul_monomial_returns_most_permissive(self):
        """Test that multiplication never reduces permissiveness."""
        a = Polynomial(1, 2, 3) * Monomial(3, 4)
        b = Monomial(1, 3) * Monomial(5, 1)
        c = Constant(5) * Monomial(6, 8)
        d = ZeroPolynomial() * Monomial(9, 2)

        self.assertIsInstance(a, Polynomial)
        self.assertIsInstance(b, Monomial)
        self.assertIsInstance(c, Monomial)
        self.assertIsInstance(d, Monomial)

    def test_mul_polymial_returns_most_permissive(self):
        """Test that multiplication never reduces permissiveness."""
        a = Polynomial(1, 2, 3) * Polynomial(3, 4, 1)
        b = Monomial(1, 3) * Polynomial(5, 1, 2)
        c = Constant(5) * Polynomial(6, 8, 1)
        d = ZeroPolynomial() * Polynomial(9, 2, 7)

        self.assertIsInstance(a, Polynomial)
        self.assertIsInstance(b, Polynomial)
        self.assertIsInstance(c, Polynomial)
        self.assertIsInstance(d, Polynomial)

    def test_permissive_zero_polynomial(self):
        """Test that permissiveness doesn't decrease with Polynomial zero."""
        a = Polynomial(1, 2, 3)
        b = Monomial(1, 2)
        c = Constant(5)
        d = ZeroPolynomial()
        zp = Polynomial()

        self.assertIsInstance(a * zp, Polynomial)
        self.assertIsInstance(b * zp, Polynomial)
        self.assertIsInstance(c * zp, Polynomial)
        self.assertIsInstance(d * zp, Polynomial)

    def test_permissive_zero_monomial(self):
        """Test that permissiveness doesn't decrease with Monomial zero."""
        a = Polynomial(1, 2, 3)
        b = Monomial(1, 2)
        c = Constant(5)
        d = ZeroPolynomial()

        zm = Monomial(0, 0)

        self.assertIsInstance(a * zm, Polynomial)
        self.assertIsInstance(b * zm, Monomial)
        self.assertIsInstance(c * zm, Monomial)
        self.assertIsInstance(d * zm, Monomial)

    def test_permissive_zero_constant(self):
        """Test that permissiveness doesn't decrease with Constant zero."""
        a = Polynomial(1, 2, 3)
        b = Monomial(1, 2)
        c = Constant(5)
        d = ZeroPolynomial()

        zc = Constant(0)

        self.assertIsInstance(a * zc, Polynomial)
        self.assertIsInstance(b * zc, Monomial)
        self.assertIsInstance(c * zc, Constant)
        self.assertIsInstance(d * zc, Constant)

    def test_permissive_zero_zero_polynomial(self):
        """Test that permissiveness doesn't decrease with ZeroPolynomial."""
        a = Polynomial(1, 2, 3)
        b = Monomial(1, 2)
        c = Constant(5)
        d = ZeroPolynomial()

        zz = ZeroPolynomial()

        self.assertIsInstance(a * zz, Polynomial)
        self.assertIsInstance(b * zz, Monomial)
        self.assertIsInstance(c * zz, Constant)
        self.assertIsInstance(d * zz, ZeroPolynomial)

    def test_frozen_vector_immutable(self):
        """Test that frozen vectors can't be modified."""
        a = ZeroPolynomial()
        b = FrozenPolynomial(1, 2, 3)

        def set_item(obj, index, value):
            obj[index] = value

        possible_errs = (TypeError, AttributeError)
        self.assertRaises(possible_errs, set_item, a._vector, 0, 5)
        self.assertRaises(possible_errs, set_item, b._vector, 0, 5)

    def test_casting_to_frozen_polynomial(self):
        """Casting polynomials is fine."""
        a = Polynomial(1, 2, 3)
        b = FrozenPolynomial.from_polynomial(a)
        self.assertEqual(a, b)

    def test_frozen_zero_instance_immutable(self):
        """Test that FrozenPolynomial returns correct zero instance."""
        fp = FrozenPolynomial(0)
        fpz = FrozenPolynomial.zero_instance()
        self._assert_polynomials_are_the_same(fp, fpz)
        self.assertRaises(AttributeError, fpz.__setitem__, 0, 1)
        self.assertRaises(AttributeError, fpz.__setattr__, "a", 1)

    def test_freezable_obeys_frozen_flag(self):
        """Test that Freezable objects behave as expected."""
        class A:
            def __init__(self):
                self._list = [1, 2, 3]

            def __setattr__(self, key, value):
                self.__dict__[key] = value

            def __setitem__(self, key, value):
                self._list[key] = value

        class B(Freezable, A):
            def __init__(self):
                super().__init__()
                self[0:2] = [4, 1]
                self._freeze()

        b = B()

        self.assertEqual([4, 1, 3], b._list)
        self.assertRaises(AttributeError, b.__setitem__, 0, 1)
        self.assertRaises(AttributeError, b.__setattr__, "a", 1)

    def test_trinomial_addition(self):
        """Test that QuadraticTrinomial behaves as expected."""
        a = QuadraticTrinomial(1, 2, 3)
        b = LinearBinomial(1, 2)
        e = QuadraticTrinomial(1, 3, 5)
        ep = Polynomial(e)
        # a + b is safe because a's degree does not change.
        self._assert_polynomials_are_the_same(e, a + b)
        # b + a requires a cast since b's degree does change.
        self._assert_polynomials_are_the_same(ep, b + a)


if __name__ == '__main__':
    unittest.main()
