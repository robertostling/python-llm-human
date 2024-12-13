# Python/LLM/Human interface

This is a small one-file library providing wrappers that make interaction
between Python, LLMs (through the OpenAI API) and humans easy with very little
boilerplate code.

## Installing

Clone this repository and install with pip:

```pip install .```

## Example

The following example shows how the two decorators `python` and `llm` can be
used to, respectively, make Python functions available as tool calls for the
LLM, and LLM calls available from Python as functions.

Note that the `add_without_unlucky` function has an empty body, and is defined
by its name, return value annotation, and docstring.

The docstring of `add_numbers` is used along with its name in the OpenAI
tools API for describing the function. The library adds wrappers to pack the
arguments and return values in pydantic types.

```
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
```

More examples can be found in the `examples/` directory.

## Few-shot prompts

Below is a preliminary and poorly tested way of producing few-shot prompts.

Functions decorated with `llm()` can take an optional `_context` argument,
which consists of messages that are inserted into the prompt. Using the
`FewShot` class it is possible to create examples of call/return sequences, to
be used as few-shot prompts.

`examples/fewshot.py` provides a simple example (without tool calls). Tool
calls can be performed in the following way:

```
world = PLH()
fs = FewShot(world)

# Call a function decorated with llm():
fs.llm_call('llm_function', argument1, ...)
# Simulate two parallel function calls made by llm_function
# Note that the tool_call() methods below will actually execute the Python
# function python_function, which is decorated with @world.python(), and make
# its return value the expected answer in the few-shot prompt.
fs.begin_tool_calls()
# Simulate call to python_function(argument1)
fs.tool_call('python_function', argument1)
# Simulate call to python_function(argument2)
fs.tool_call('python_function', argument2)
fs.end_tool_calls()
# Return from the original function
fs.llm_return('llm_function', return_value)

# Now fs.messages can be included in calls to llm_function
llm_function(arguments, _context=fs.messages)

