# Python package defining single-variable polynomials and operations with them

[![PyPI version](https://badge.fury.io/py/py-polynomial.svg)](https://badge.fury.io/py/py-polynomial)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/py-polynomial.svg)](https://pypi.python.org/pypi/py-polynomial/)
[![PyPI license](https://img.shields.io/pypi/l/py-polynomial.svg)](https://pypi.python.org/pypi/py-polynomial/)

[![allexks](https://circleci.com/gh/allexks/py-polynomial.svg?style=svg)](https://circleci.com/gh/allexks/py-polynomial)
[![CodeFactor](https://www.codefactor.io/repository/github/allexks/py-polynomial/badge)](https://www.codefactor.io/repository/github/allexks/py-polynomial)
[![codecov](https://codecov.io/gh/allexks/py-polynomial/branch/master/graph/badge.svg)](https://codecov.io/gh/allexks/py-polynomial)

## Installation
`pip3 install py-polynomial`

## Sample functionality
``` pycon
>>> from polynomial import Polynomial as P
>>> a = P(1, 2, 3, 4)
>>> a
Polynomial(1, 2, 3, 4)

>>> str(a)
x^3 + 2x^2 + 3x + 4

>>> b = P([4 - x for x in range(4)])  # Flexible initialization
>>> str(b)
4x^3 + 3x^2 + 2x + 1

>>> b.derivative                      # First derivative
Polynomial(12, 6, 2)

>>> str(b.derivative)
12x^2 + 6x + 2

>>> str(b.nth_derivative(2))          # Second or higher derivative
24x + 6

>>> str(a + b)                        # Addition
5x^3 + 5x^2 + 5x + 5

>>> (a + b).calculate(5)              # Calculating value for a given x
780

>>> p = P(1, 2) * P(1, 2)             # Multiplication
>>> p
Polynomial(1, 4, 4)

>>> p[0] = -4                         # Accessing coefficient by degree
>>> p
Polynomial(1, 4, -4)

>>> p[1:] = [4, -1]                   # Slicing
>>> p
Polynomial(-1, 4, -4)

>>> (p.a, p.b, p.c)                   # Accessing coefficients by name convention
(-1, 4, -4)

>>> p.a, p.c = 1, 4
>>> (p.A, p.B, p.C)
(1, 4, 4)

>>> q, remainder = divmod(p, P(1, 2)) # Division and remainder
>>> q
Polynomial(1.0, 2.0)
>>> remainder
Polynomial()

>>> p // P(1, 2)
Polynomial(1.0, 2.0)

>>> P(1, 2, 3) % P(1, 2)
Polynomial(3)

>>> P(2, 1) in P(4, 3, 2, 1)          # Check whether it contains given terms
True

>>> str(P("abc"))                     # Misc
ax^2 + bx + c
```

``` pycon
>>> from polynomial import QuadraticTrinomial, Monomial
>>> y = QuadraticTrinomial(1, -2, 1)
>>> str(y)
x^2 - 2x + 1

>>> y.discriminant
0

>>> y.real_roots
(1, 1)

>>> y.real_factors
(1, Polynomial(1, -1), Polynomial(1, -1))

>>> str(Monomial(5, 3))
5x^3

>>> y += Monomial(9, 2)
>>> y
Polynomial(10, -2, 1)

>>> str(y)
10x^2 - 2x + 1

>>> (y.a, y.b, y.c)
(10, -2, 1)

>>> (y.A, y.B, y.C)
(10, -2, 1)

>>> y.complex_roots
((0.1 + 0.3j), (0.1 - 0.3j))
```
