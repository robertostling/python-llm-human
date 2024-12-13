"""Demonstration of Python calls from LLM functions.

The LLM function add_without_unlucky has access to an Python addition
function, add_numbers. Internally this uses tool calls, which are wrapped
using the @python() decorator.
"""

import plh

world = plh.PLH()


@world.python()
def add_numbers(x: int, y: int) -> int:
    """Add the numbers x and y."""
    print(f'Calculator used to compute {x} + {y}')
    return x + y


@world.llm()
def add_without_unlucky(x, y) -> int:
    """Add x and y, dealing with unlucky numbers.

    If the sum is an unlucky number in western culture, return another
    nearby number instead.
    """
    pass


for x, y in [(2, 2), (8, 5)]:
    print(f'{x} + {y} = {add_without_unlucky(x, y, _temperature=1.0)}')
