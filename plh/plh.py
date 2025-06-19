"""Python/LLM/human cooperation interface.

The main class PLH defines decorators (python, llm) for functions that are
defined in Python and as an LLM prompt. Necessary wrappers around the OpenAI
API are created to provide a uniform interface based on Python functions.

See demo_plh_courses.py and demo_plh_lucky_add.py for examples of how to use
this interface.
"""

import json
import inspect
import random

import openai
from pydantic import BaseModel, create_model


client = openai.OpenAI()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase."""
    return "".join(part.capitalize() for part in name.split("_"))


def pydantic_from_function(function):
    """Create a pydantic model from a type-annotated function."""
    signature = inspect.signature(function)
    model_name = snake_to_camel(function.__name__)
    model_args = {}
    for parameter in signature.parameters.values():
        if parameter.annotation is parameter.empty:
            raise TypeError(
                f"Argument {parameter.name} of {function.__name__} "
                "lacks type annotation."
            )
        model_args[parameter.name] = (
            parameter.annotation,
            ... if parameter.default is parameter.empty else parameter.default,
        )
    model = create_model(model_name, **model_args)
    return model


class FewShot:
    def __init__(self, plh):
        self.plh = plh
        self.messages = []
        self.tool_calls = []
        self.tool_results = {}

    def llm_call(self, name, *args):
        self.messages.append(self.plh.llm_call_message(name, *args))

    def llm_return(self, name, result):
        function = self.plh.llm_functions[name]
        response_format = function['response_format']
        if function['wrapped']:
            value = response_format(result=result)
        elif isinstance(result, response_format):
            value = result
        else:
            raise TypeError(
                    'Expected result to be of type '
                    f'{response_format.__name__}')
        self.messages.append(
                dict(role='assistant',
                     content=json.dumps(value.model_dump())))

    def begin_tool_calls(self):
        if self.tool_calls or self.tool_results:
            raise ValueError('Missing call to end_tool_calls()?')

    def end_tool_calls(self):
        assert self.tool_calls
        assert len(self.tool_calls) == len(self.tool_results)
        self.messages.append(
                dict(role='assistant',
                     tool_calls=self.tool_calls))
        for tool_call_id, result in self.tool_results.items():
            self.messages.append(
                    dict(role='tool',
                         content=json.dumps(result),
                         tool_call_id=tool_call_id))
        self.tool_calls = []
        self.tool_results = {}

    def tool_call(self, name, *args):
        tool = self.plh.tools[name]
        result = tool['function'](*args)
        call_id = f'call_{random.randint(0, 1000000000000):012d}'
        self.tool_results[call_id] = result
        argument = tool['parameter_type'](*args)
        self.tool_calls.append(dict(
            id=call_id,
            function=dict(
                name=name,
                arguments=json.dumps(argument.model_dump())),
            type='function'))


class PLH:
    """Python/LLM/Human interface.

    Attributes:
        tools -- dict { name: info } of tools, i.e. functions
                 marked by the @python decorator. The name used is that of
                 the argument data structure, a pydantic object. This is the
                 CamelCase variant of the snake_case name of the decorated
                 function. The info dict contains the following keys:
            identifier -- snake_case name of function
            description -- docstring of function, which will be passed to
                           the OpenAI API as the description of this tool
            function -- the original Python function
            response_format -- the return value, a pydantic type
            parameter_type -- the argument data structure, a pydantic type
                              whose name is the key of this dict
            tool -- OpenAI API representation of the tool, created by
                    the openai.pydantic_function_tool function
        llm_functions -- dict { name: info } of Python functions that wrap
                         LLM prompts. The info dict contains the following
                         keys:
            identifier -- name of function, same as the key that maps to this
                          dict
            function -- Python function itself (empty, used only for its
                        signature and docstring)
            description -- docstring of function, used as LLM prompt
            parameters -- list of parameter names
            response_format -- pydantic type of return value
            wrapped -- True if response_format is a pydantic type that
                       wraps a simple data type, if so what is actually
                       returned from the function is the `result` attribute of
                       the data type returned from the LLM.
    """
    tools: dict[str, dict]
    llm_functions: dict[str, dict]

    def __init__(self, verbosity=0):
        self.tools = {}
        self.llm_functions = {}
        self.verbosity = verbosity

    def chat_call(self, messages, model, temperature, response_format):
        """Wrapper around OpenAI's chat completion with structured output."""
        kwargs = {}
        if self.tools:
            header = (
                "In carrying out the tasks defined below, you have access "
                "to the following tools:\n\n"
            )
            descriptions = [
                f"The function {name}: {tool['description']}"
                for name, tool in self.tools.items()
            ]
            messages = [
                dict(role="system", content=header + "\n\n".join(descriptions))
            ] + messages
            kwargs['tools'] = [tool["tool"] for tool in self.tools.values()]
        completion = client.beta.chat.completions.parse(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            **kwargs
        )

        message = completion.choices[0].message

        if self.verbosity >= 3:
            print(json.dumps(message.model_dump(), indent=4))

        if not message.tool_calls:
            return completion.choices[0].message.parsed

        result_messages = []
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            if function_name not in self.tools:
                raise NameError(f"Function '{function_name}' not defined.")
            tool = self.tools[function_name]
            argument = tool_call.function.parsed_arguments
            if not isinstance(argument, tool["parameter_type"]):
                raise TypeError(
                    f"Argument to '{function_name}' expected to be "
                    f"of type {tool['parameter_type'].__name__} "
                    f"but got {type(argument).__name__}"
                )
            print_call = tool.get("print_call", False)
            ask_before_call = tool.get("ask_before_call", False)
            if self.verbosity >= 3 or print_call or ask_before_call:
                argument_json = json.dumps(argument.model_dump(), indent=4)
                print(f"{function_name}({argument_json})")
            result = None
            if ask_before_call:
                while True:
                    answer = input("Proceed with the call above? [y/N] "
                                   ).strip().casefold()
                    if answer in ('y', 'n', ''):
                        break
                # TODO: could also use IPython via IPython.embed()
                #       but then values need to be passed through mutable data
                #       structures instead.
                if answer in ("n", ""):
                    result = "N/A"
                    print(
                        "You now have the opportunity to manually define "
                        "the variable `result` instead."
                    )
                    print(
                        "The expected return type is "
                        f"{tool['response_format'].__name__}."
                    )
                    print("When finished, use the pdb command `continue`.")
                    breakpoint()
            if result is None:
                result = tool["function"](**argument.model_dump())
            if issubclass(type(result), BaseModel):
                result = result.model_dump()
            if self.verbosity >= 3 or print_call:
                print("return", json.dumps(result, indent=4))
            result_messages.append(
                dict(
                    role="tool",
                    content=json.dumps(result),
                    tool_call_id=tool_call.id,
                )
            )

        return self.chat_call(
            messages + [message] + result_messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
        )

    def python(self, **kwargs):
        """Decorator to add the function to a dict of LLM-callable tools.

        Keyword arguments are passed directly to the tool dict, which is
        interpreted it `chat_call`. Currently these values are defined:
            - ask_before_call: bool         ask for user confirmation
            - print_call: bool              print argument and return value

        The docstring of the decorated function is passed to the tools API
        as the description of this function.
        """

        def decorator(f):
            docstring = inspect.getdoc(f)
            description = docstring
            signature = inspect.signature(f)
            parameter_type = pydantic_from_function(f)
            if signature.return_annotation == signature.empty:
                raise TypeError(f"Return annotation required for {f.__name__}")
            response_format = signature.return_annotation
            self.tools[parameter_type.__name__] = (
                dict(
                    identifier=f.__name__,
                    description=description,
                    function=f,
                    response_format=response_format,
                    parameter_type=parameter_type,
                    tool=openai.pydantic_function_tool(parameter_type),
                )
                | kwargs
            )
            return f

        return decorator

    def llm_call_message(self, function_name, *args):
        fun = self.llm_functions[function_name]
        if len(args) != len(fun['parameters']):
            raise TypeError(
                    f"Got {len(args)} arguments but expected "
                    f"{fun['parameters']}")
        args_dict = dict(zip(fun['parameters'], args))
        return dict(role="user",
                    content=(
                        f"{fun['description']}\n\n"
                        f"Arguments:\n{json.dumps(args_dict)}"
                    ))

    def llm(self, model="gpt-4o-mini", temperature=0.0):
        """Decorator to make an LLM execute the function.

        The LLM is given the docstring and arguments of the function. The body
        of the function is ignored.

        Note that only plain positional arguments without default values are
        allowed. Argument type annotations are non-mandatory and are not
        checked at run-time.

        The decorated function will accept additional keyword arguments to
        control the LLM sampling:
            _model: str             model name
            _temperature: float     sampling temperature
        The _model and _temperature arguments override the defaults set by the
        `model` and `temperature` arguments to this decorator.

        The return type annotation is mandatory. Anything that is not a
        pydantic.BaseModel subclass will be wrapped in a dynamically created
        pydantic type.
        """

        def decorator(f):
            docstring = inspect.getdoc(f)
            signature = inspect.signature(f)
            parameters = signature.parameters
            function_name = f.__name__
            for parameter in parameters.values():
                if parameter.default is not parameter.empty:
                    raise TypeError(
                        "Default parameters not allowed in interpreted "
                        f"function {f.__name__}"
                    )
                if parameter.kind != parameter.POSITIONAL_OR_KEYWORD:
                    raise TypeError(
                        "Only plain positional arguments allowed in "
                        f"interpreted function {f.__name__}, but "
                        f"{parameter.name} is {parameter.kind}"
                    )
            if signature.return_annotation == signature.empty:
                raise TypeError(f"Return annotation required for {f.__name__}")
            wrapped = not issubclass(signature.return_annotation, BaseModel)
            if not wrapped:
                # Structured output with pre-existing pydantic type.
                response_format = signature.return_annotation
            else:
                # Other return type, so we need to create a wrapper.
                model_name = snake_to_camel(f.__name__) + "Result"
                response_format = create_model(
                    model_name, result=(signature.return_annotation, ...)
                )

            self.llm_functions[function_name] = dict(
                    identifier=function_name,
                    function=f,
                    description=docstring,
                    parameters=list(parameters.keys()),
                    response_format=response_format,
                    wrapped=wrapped)

            def execute_function(*args, **kwargs):
                args = [
                    x.model_dump() if issubclass(type(x), BaseModel) else x
                    for x in args
                ]
                context = kwargs.get('_context', [])
                messages = [
                    dict(
                        role="system",
                        content=(
                            "Your task is to carry out the tasks "
                            "provided by the user."
                        ))
                        ] + context + [
                            self.llm_call_message(function_name, *args)
                        ]
                response = self.chat_call(
                    messages,
                    model=kwargs.get('_model', model),
                    temperature=kwargs.get('_temperature', temperature),
                    response_format=response_format,
                )
                return response.result if wrapped else response

            return execute_function

        return decorator
