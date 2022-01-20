# Python package defining single-variable polynomials and operations with them

[![PyPI version](https://badge.fury.io/py/py-polynomial.svg)](https://badge.fury.io/py/py-polynomial)
[![Downloads](https://static.pepy.tech/personalized-badge/py-polynomial?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/py-polynomial)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/py-polynomial.svg)](https://pypi.python.org/pypi/py-polynomial/)
[![PyPI license](https://img.shields.io/pypi/l/py-polynomial.svg)](https://pypi.python.org/pypi/py-polynomial/)

![Unit Tests](https://github.com/allexks/py-polynomial/workflows/Unit%20Tests/badge.svg)
![Code Documentation Style](https://github.com/allexks/py-polynomial/workflows/Code%20Documentation%20Style/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/allexks/py-polynomial/badge)](https://www.codefactor.io/repository/github/allexks/py-polynomial)
[![codecov](https://codecov.io/gh/allexks/py-polynomial/branch/master/graph/badge.svg)](https://codecov.io/gh/allexks/py-polynomial)

## Installation
`pip install py-polynomial`

## Documentation
[Click here for code-derived documentation and help](https://allexks.github.io/py-polynomial/)

## Quick examples
### Flexible initialization
``` pycon
>>> from polynomial import Polynomial

>>> a = Polynomial(1, 2, 3, 4)
>>> str(a)
x^3 + 2x^2 + 3x + 4

>>> b = Polynomial([4 - x for x in range(4)])
>>> str(b)
4x^3 + 3x^2 + 2x + 1
```

### First derivative
``` pycon
>>> b.derivative
Polynomial(12, 6, 2)

>>> str(b.derivative)
12x^2 + 6x + 2
```

### Second or higher derivative
``` pycon
>>> str(b.nth_derivative(2))
24x + 6
```

### Addition
``` pycon
>>> str(a + b)
5x^3 + 5x^2 + 5x + 5
```

### Calculating value for a given x
``` pycon
>>> (a + b).calculate(5)
780

>>> а(2)  #  equivalent to a.calculate(2)
26
```

### Multiplication
``` pycon
>>> p = Polynomial(1, 2) * Polynomial(1, 2)
>>> p
Polynomial(1, 4, 4)
```

### Accessing coefficient by degree
``` pycon
>>> p[0] = -4
>>> p
Polynomial(1, 4, -4)
```

### Slicing
``` pycon
>>> p[1:] = [4, -1]
>>> p
Polynomial(-1, 4, -4)
```

### Accessing coefficients by name convention
``` pycon
>>> (p.a, p.b, p.c)
(-1, 4, -4)

>>> p.a, p.c = 1, 4
>>> (p.A, p.B, p.C)
(1, 4, 4)
```

### Division and remainder
``` pycon
>>> q, remainder = divmod(p, Polynomial(1, 2))
>>> q
Polynomial(1.0, 2.0)
>>> remainder
Polynomial()

>>> p // Polynomial(1, 2)
Polynomial(1.0, 2.0)

>>> P(1, 2, 3) % Polynomial(1, 2)
Polynomial(3)
```

### Check whether it contains given terms
``` pycon
>>> Polynomial(2, 1) in Polynomial(4, 3, 2, 1)
True
```

### Definite integral
```pycon
>>> Polynomial(3, 2, 1).integral(0, 1)
3
```

### Misc
``` pycon
>>> str(Polynomial("abc"))
ax^2 + bx + c
```

### Roots and discriminants
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
