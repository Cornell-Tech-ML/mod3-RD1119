"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:


# - mul
def mul(x: float, y: float) -> float:
    """Multiply

    Args:
    ----
        x: float : First value.
        y: float : Second value.

    Returns:
    -------
        float : The product of x and y.

    """
    return x * y


# - id
def id(x: float) -> float:
    """Identity

    Args:
    ----
        x: float : Value to return.

    Returns:
    -------
        float : The value x.

    """
    return x


# - add
def add(x: float, y: float) -> float:
    """Add

    Args:
    ----
        x: float : First value.
        y: float : Second value.

    Returns:
    -------
        float : The sum of x and y.

    """
    return float(x + y)


# - neg
def neg(x: float) -> float:
    """Negate

    Args:
    ----
        x: float : Value to negate.

    Returns:
    -------
        float : The negation of x.

    """
    return -x


# - lt
def lt(x: float, y: float) -> float:
    """Less than

    Args:
    ----
        x: float : First value.
        y: float : Second value.

    Returns:
    -------
        float : 1 if x is less than y, 0 otherwise

    """
    return 1.0 if x < y else 0.0


# - eq
def eq(x: float, y: float) -> float:
    """Equal

    Args:
    ----
        x: float : First value.
        y: float : Second value.

    Returns:
    -------
        float : 1.0 if x is equal to y, 0.0 otherwise.

    """
    return 1.0 if x == y else 0.0


# - max
def max(x: float, y: float) -> float:
    """Maximum

    Args:
    ----
        x: float : First value.
        y: float : Second value.

    Returns:
    -------
        float : The maximum of x and y.

    """
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Is close

    Args:
    ----
        x: float : First value.
        y: float : Second value.

    Returns:
    -------
        bool : True if x is close to y, False otherwise. The close is defined as f(x) = |x - y| < 1e-2.

    """
    return abs(x - y) < 1e-2


# - sigmoid
def sigmoid(x: float) -> float:
    """Sigmoid

    Args:
    ----
        x: float : Value to apply sigmoid to.

    Returns:
    -------
        float : The sigmoid of x.


    """
    # For sigmoid calculate as:
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """ReLU

    Args:
    ----
        x: float : Value to apply ReLU to.

    Returns:
    -------
        float : The ReLU of x.

    """
    return x if x > 0 else 0.0


# - log
def log(x: float) -> float:
    """Log

    Args:
    ----
        x: float : Value to apply log to.

    Returns:
    -------
        float : The log of x.

    """
    if x <= 0:
        raise ValueError("Log of non-positive value")
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Exp

    Args:
    ----
        x: float : Value to apply exp to.

    Returns:
    -------
        float : The exp of x.

    """
    return math.exp(x)


# - log_back
def log_back(x: float, y: float) -> float:
    """Log back

    Args:
    ----
        x: float : Value to apply log back to.
        y: float : Value to apply log back to.

    Returns:
    -------
        float : The log back of x and y.

    """
    if x <= 0:
        raise ValueError("Log of non-positive value")
    return y / x


# - inv
def inv(x: float) -> float:
    """Inverse

    Args:
    ----
        x: float : Value to apply inverse to.

    Returns:
    -------
        float : The inverse of x.

    """
    if x == 0:
        raise ValueError("Division by zero")
    return 1.0 / x


# - inv_back
def inv_back(x: float, y: float) -> float:
    """Inverse back

    Args:
    ----
        x: float : Value to apply inverse back to.
        y: float : Value to apply inverse back to.

    Returns:
    -------
        float : The inverse back of x and y.

    """
    return -1.0 / x**2 * y


# - relu_back
def relu_back(x: float, y: float) -> float:
    """ReLU back

    Args:
    ----
        x: float : Value to apply ReLU back to.
        y: float : Value to apply ReLU back to.

    Returns:
    -------
        float : The ReLU back of x and y.

    """
    return y if x > 0 else 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable

    Args:
    ----
        fn: Callable[[float], float] : Function to apply to each element.

    Returns:
    -------
        Callable[[Iterable], Iterable] : A function that applies the given function to each element of an iterable.

    """

    def map_fn(arr: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in arr]

    return map_fn


# - zipWith
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
        fn: Callable[[float, float], float] : Function to combine elements with.

    Returns:
    -------
        Callable[[Iterable[float], Iterable[float], Iterable[float]]] : A function that combines elements from two iterables using the given function.

    """

    def zipWith_fn(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(arr1, arr2)]

    return zipWith_fn


# - reduce
def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function

    Args:
    ----
        fn: Callable[[float, float], float] : Function to reduce elements with.
        start: float : Initial value to start the reduction with.

    Returns:
    -------
        Callable[[Iterable[float]], float] : A function that reduces an iterable to a single value using the given function.

    """

    def reduce_fn(arr: Iterable[float]) -> float:
        value = start
        for element in arr:
            value = fn(value, element)
        return value

    return reduce_fn


#
# Use these to implement
# - negList : negate a list
def negList(arr: Iterable[float]) -> Iterable[float]:
    """Negate a list

    Args:
    ----
        arr: Iterable[float] : List to negate.

    Returns:
    -------
        Iterable[float] : The negation of the list.

    """
    return map(neg)(arr)


# - addLists : add two lists together
def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith

    Args:
    ----
        arr1: Iterable[float] : First list.
        arr2: Iterable[float] : Second list.

    Returns:
    -------
        Iterable : The sum of the two lists.

    """
    return zipWith(add)(arr1, arr2)


# - sum: sum lists
def sum(arr: Iterable[float]) -> float:
    """Sum all elements in a list using reduce

    Args:
    ----
        arr: Iterable[float] : List to sum.

    Returns:
    -------
        float : The sum of the list.

    """
    return reduce(add, 0.0)(arr)


# - prod: take the product of lists
def prod(arr: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce

    Args:
    ----
        arr: Iterable[float] : List to take the product of.

    Returns:
    -------
        float : The product of the list.

    """
    return reduce(mul, 1.0)(arr)


# TODO: Implement for Task 0.3.