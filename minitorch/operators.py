"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x:float, y:float) -> float:
    """
    Multiply two numbers.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The product of x and y.
    """
    return x * y

def id(x:float) -> float:
    """
    Identity function.

    Args:
        x (float): The input number.

    Returns:
        float: Returns the input number unchanged.
    """
    return x

def add(x:float, y:float) -> float:
    """
    Add two numbers.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The sum of x and y.
    """
    return x + y

def neg(x:float) -> float:
    """
    Negate a number.

    Args:
        x (float): The input number.

    Returns:
        float: The negation of x.
    """
    return -x

def lt(x:float, y:float) -> bool:
    """
    Check if x is less than y.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        bool: True if x is less than y, False otherwise.
    """
    return x < y

def eq(x:float, y:float) -> bool:
    """
    Check if x is equal to y.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        bool: True if x is equal to y, False otherwise.
    """
    return x == y

def max(x:float, y:float) -> float:
    """
    Return the maximum of x and y.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The maximum of x and y.
    """
    return x if x > y else y

def is_close(x:float, y:float) -> bool:
    """
    Check if x is close to y within a tolerance of 1e-2.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        bool: True if x is close to y, False otherwise.
    """
    return abs(x - y) < 1e-2

def sigmoid(x:float) -> float:
    """
    Compute the sigmoid of x.

    Args:
        x (float): The input number.

    Returns:
        float: The sigmoid of x.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
    
def relu(x:float) -> float:
    """
    Compute the ReLU of x.

    Args:
        x (float): The input number.

    Returns:
        float: The ReLU of x.
    """
    return x if x > 0 else 0.0

def log(x:float) -> float:
    """
    Compute the natural logarithm of x.

    Args:
        x (float): The input number.

    Returns:
        float: The natural logarithm of x.
    """
    return math.log(x)

def exp(x:float) -> float:
    """
    Compute the exponential of x.

    Args:
        x (float): The input number.

    Returns:
        float: The exponential of x.
    """
    return math.exp(x)

def log_back(x:float, d:float) -> float:
    """
    Compute the gradient of the logarithm function.

    Args:
        x (float): The input number.
        d (float): The upstream gradient.

    Returns:
        float: The gradient of log at x multiplied by d.
    """
    return d / x

def inv(x:float) -> float:
    """
    Compute the inverse of x.

    Args:
        x (float): The input number.

    Returns:
        float: The inverse of x.
    """
    return 1.0 / x

def inv_back(x:float, d:float) -> float:
    """
    Compute the gradient of the inverse function.

    Args:
        x (float): The input number.
        d (float): The upstream gradient.

    Returns:
        float: The gradient of inv at x multiplied by d.
    """
    return -d / (x * x)

def relu_back(x:float, d:float) -> float:
    """
    Compute the gradient of the ReLU function.

    Args:
        x (float): The input number.
        d (float): The upstream gradient.

    Returns:
        float: The gradient of ReLU at x multiplied by d.
    """
    return d if x > 0 else 0.0

# def addLists(xs:list[float], ys:list[float]) -> list[float]:
#     """
#     Add two lists elementwise.
#     Args:
#         xs (list[float]): first list
#         ys (list[float]): second list
#     Returns:
#         list[float]: elementwise sum
#     """
#     assert len(xs) == len(ys), "Lists must be the same length"
#     return [x + y for x, y in zip(xs, ys)]


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(fn:callable) -> callable:
    """
    Map a callable over a list of numbers.

    Args:
        fn (callable): The function to apply.

    Returns:
        callable: A new function that takes a list and applies fn to each element.
    """
    def mapped(xs:Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in xs]
    return mapped

def zipWith(fn:callable) -> callable:
    """
    Zip two lists together with a function.

    Args:
        fn (callable): The function to apply.

    Returns:
        callable: A new function that takes two lists and applies fn to each pair of elements.
    """
    def zipped(xs:Iterable[float], ys:Iterable[float]) -> Iterable[float]:
        assert len(xs) == len(ys), "Lists must be the same length"
        return [fn(x, y) for x, y in zip(xs, ys)]
    return zipped

def reduce(fn:callable, initial:float) -> callable:
    """
    Reduce a list of numbers to a single number with a binary function.

    Args:
        fn (callable): The binary function to apply.
        initial (float): The initial value for the reduction.

    Returns:
        function: A new function that takes a list and reduces it using fn.
    """
    def reduced(xs:Iterable[float]) -> float:
        result = initial
        for x in xs:
            result = fn(result, x)
        return result
    return reduced

def negList(xs:Iterable[float]) -> Iterable[float]:
    """
    Negate a list of numbers.

    Args:
        xs (Iterable[float]): The list of numbers to negate.

    Returns:
        Iterable[float]: The negated list.
    """
    return map(neg)(xs)

def addLists(xs:Iterable[float], ys:Iterable[float]) -> Iterable[float]:
    """
    Add two lists of numbers elementwise.

    Args:
        xs (Iterable[float]): The first list.
        ys (Iterable[float]): The second list.

    Returns:
        Iterable[float]: The elementwise sum of the two lists.
    """
    return zipWith(add)(xs, ys)

def sum(xs:Iterable[float]) -> float:
    """
    Sum a list of numbers.

    Args:
        xs (Iterable[float]): The list of numbers to sum.

    Returns:
        float: The sum of the list.
    """
    return reduce(add, 0.0)(xs)

def prod(xs:Iterable[float]) -> float:
    """
    Take the product of a list of numbers.

    Args:
        xs (Iterable[float]): The list of numbers to multiply.

    Returns:
        float: The product of the list.
    """
    return reduce(mul, 1.0)(xs)

# TODO: Implement for Task 0.3.
