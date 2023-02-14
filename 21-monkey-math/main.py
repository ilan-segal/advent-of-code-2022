"""
https://adventofcode.com/2022/day/21
"""

from __future__ import annotations
from typing import (
    Callable,
    Literal,
    ParamSpec,
    TypeVar,
)

import abc
import dataclasses
import functools
import re
import time


T = TypeVar('T')
P = ParamSpec('P')
def time_execution(f: Callable[P, T]) -> Callable[P, T]:
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f'{f.__name__} executed in {end-start:.5} s')
        return result
    return wrapped


def get_raw_input() -> str:
    return open('21-monkey-math/input.txt', 'r').read().strip().replace('\r\n', '\n')


OperatorSymbol = Literal['+', '-', '*', '/']
BinaryOperator = Callable[[int, int], int]
_SYMBOL_TO_OPERATOR: dict[OperatorSymbol, BinaryOperator] = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x // y,
}
_SYMBOL_TO_INVERSE: dict[OperatorSymbol, BinaryOperator] = {
    '+': _SYMBOL_TO_OPERATOR['-'],
    '-': _SYMBOL_TO_OPERATOR['+'],
    '*': _SYMBOL_TO_OPERATOR['/'],
    '/': _SYMBOL_TO_OPERATOR['*'],
}
HUMAN_NAME = 'humn'


def get_operator(symbol: OperatorSymbol) -> BinaryOperator:
    if symbol not in _SYMBOL_TO_OPERATOR:
        raise ValueError(f'Invalid symbol "{symbol}", must be one of {list(_SYMBOL_TO_OPERATOR.keys())}')
    return _SYMBOL_TO_OPERATOR[symbol]


def get_inverse_operator(symbol: OperatorSymbol) -> BinaryOperator:
    if symbol not in _SYMBOL_TO_INVERSE:
        raise ValueError(f'Invalid symbol "{symbol}", must be one of {list(_SYMBOL_TO_INVERSE.keys())}')
    return _SYMBOL_TO_INVERSE[symbol]


@dataclasses.dataclass
class Binomial:
    """
    ax+b
    """

    a: float
    b: float

    def evaluate(self, *, x: float) -> float:
        return (self.a * x) + self.b
    
    def is_constant(self) -> bool:
        return self.a == 0

    def __repr__(self) -> str:
        if self.a == 0:
            multiplication_part = ''
        else:
            multiplication_part = f'{self.a}x'
        if self.b == 0:
            addition_part = ''
        elif self.b > 0:
            addition_part = f'+{self.b}'
        else:
            addition_part = f'{self.b}'
        result = f'{multiplication_part}{addition_part}'
        if result.startswith('+'):
            result = result[1:]
        if result.startswith('1x'):
            result = result[1:]
        return result

    def __add__(self, other: Binomial) -> Binomial:
        return Binomial(self.a + other.a, self.b + other.b)

    def __sub__(self, other: Binomial) -> Binomial:
        return Binomial(self.a - other.a, self.b - other.b)

    def __mul__(self, other: Binomial) -> Binomial:
        if not (self.is_constant() or other.is_constant()):
            raise ValueError(f'Cannot multiply two binomials')
        if self.is_constant():
            return Binomial(other.a * self.b, other.b * self.b)
        else:
            return Binomial(self.a * other.b, self.b * other.b)

    def __floordiv__(self, other: Binomial) -> Binomial:
        if not other.is_constant():
            raise ValueError('Cannot divide by binomial')
        return Binomial(self.a / other.b, self.b / other.b)


class Monkey(metaclass=abc.ABCMeta):

    __name_to_monkey_map: dict[str, Monkey] = dict()

    def __init__(self, name: str) -> None:
        self.name = name
        Monkey.__name_to_monkey_map[name] = self

    @abc.abstractmethod
    def evaluate(self) -> int:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def to_expression(self, variable_name: str) -> Binomial:
        pass

    @classmethod
    def get_monkey(cls, name: str) -> Monkey:
        return cls.__name_to_monkey_map[name]

    @classmethod
    def get_all_monkeys(cls) -> list[Monkey]:
        return list(cls.__name_to_monkey_map.values())

    @classmethod
    def initialize(cls) -> None:
        for monkey in cls.__name_to_monkey_map.values():
            del monkey
        cls.__name_to_monkey_map.clear()


class ConstantMonkey(Monkey):
    def __init__(self, name: str, constant: int) -> None:
        super().__init__(name)
        self.constant = constant

    def evaluate(self) -> int:
        return self.constant

    def to_expression(self, variable_name: str) -> Binomial:
        if self.name == variable_name:
            return Binomial(1, 0)
        return Binomial(0, self.constant)

    def __repr__(self) -> str:
        return f'{self.name}: {self.constant}'


class OperatorMonkey(Monkey):
    def __init__(self, name: str, operator_symbol: OperatorSymbol, dependency_names: list[str]) -> None:
        super().__init__(name)
        self.operator_symbol = operator_symbol
        self.operator = get_operator(operator_symbol)
        self.dependency_names = dependency_names

    def evaluate(self) -> int:
        dependencies = [Monkey.get_monkey(name) for name in self.dependency_names]
        evaluated_dependencies = [monkey.evaluate() for monkey in dependencies]
        return functools.reduce(self.operator, evaluated_dependencies)

    def to_expression(self, variable_name: str) -> Binomial:
        if self.name == variable_name:
            return Binomial(1, 0)
        child_expressions = [
            Monkey.get_monkey(child_name).to_expression(variable_name)
            for child_name in self.dependency_names
        ]
        return functools.reduce(self.operator, child_expressions)  # type: ignore

    def __repr__(self) -> str:
        return f'{self.name}: {(" " + self.operator_symbol + " ").join(self.dependency_names)}'


def parse_monkey(monkey_str: str) -> Monkey:
    constant_monkey_pattern = r'([a-z]+): (-?[0-9]+)'
    result = re.search(constant_monkey_pattern, monkey_str)
    if result is not None:
        monkey_name = result.group(1)
        monkey_constant = int(result.group(2))
        return ConstantMonkey(monkey_name, monkey_constant)
    operator_monkey_pattern = r'([a-z]+): ([a-z]+) (.) ([a-z]+)'
    result = re.search(operator_monkey_pattern, monkey_str)
    if result is not None:
        monkey_name = result.group(1)
        dependencies = [result.group(2), result.group(4)]
        operator_symbol = result.group(3)
        return OperatorMonkey(monkey_name, operator_symbol, dependencies)  # type: ignore
    raise ValueError(monkey_str)


def initialize_monkeys(raw_input: str) -> list[Monkey]:
    Monkey.initialize()
    return list(map(parse_monkey, raw_input.split('\n')))


@time_execution
def part_1(raw_input: str) -> None:
    initialize_monkeys(raw_input)
    print(f'part_1={Monkey.get_monkey("root").evaluate()}')


@time_execution
def part_2(raw_input: str) -> None:
    initialize_monkeys(raw_input)
    root = Monkey.get_monkey('root')
    assert isinstance(root, OperatorMonkey)
    assert len(root.dependency_names) == 2
    left, right = map(lambda m: m.to_expression('humn'), map(Monkey.get_monkey, root.dependency_names))
    """
    ax+b = cx+d  // Solve for x
    (a-c)x = d-b
    x = (d-b)/(a-c)
    """
    x = (right.b - left.b) / (left.a - right.a)
    print(f'part_2={round(x)}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)
    part_2(raw_input)