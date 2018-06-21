# Python Package for handling of various polynomials (not finished).

Sample usage:
```python3
from Polynomial import Polynomial as P

p1 = P(1,2,3,4)  # x^3 + 2x^2 + 3x + 4
p2 = P([x for x in range(4, 0, -1)])  # 4x^3 + 3x^2 + 2x + 1
new_p = p1 + p2  # 5x^3 + 5x^2 + 5x + 5

print(P(1,2) * P(1,2))  # prints x^2 + 4x + 4 [which is exactly (x+2)*(x+2)]

print(P("abc"))  # prints ax^2 + bx + c

# more functionality to be added...
```

```python3
from Trinomial import QuadraticTrinomial as QT
from Polynomial import Monomial

y = QT(1, -2, 1)  # x^2 - 2x + 1
print(y.discriminant)  # prints 0
print(y.get_roots())  # prints (1, 1)

y += Monomial(9, 2)  # adding 9x^2; y is now 10x^2 - 2x + 1
print(y.get_roots())  # ((0.1 + 0.3j), (0.1 - 0.3j)) - supports complex numbers

# more functionality to be added...
```
