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
SYMBOL_TO_OPERATOR: dict[OperatorSymbol, Callable[[int, int], int]] = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x // y,
}


NAME_TO_MONKEY_MAP: dict[str, Monkey] = dict()


class Monkey:
    def __init__(self, name: str) -> None:
        self.name = name
        NAME_TO_MONKEY_MAP[self.name] = self

    def evaluate(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class ConstantMonkey(Monkey):
    def __init__(self, name: str, constant: int) -> None:
        super().__init__(name)
        self.__constant = constant

    def evaluate(self) -> int:
        return self.__constant

    def __repr__(self) -> str:
        return f'{self.name}: {self.__constant}'


class OperatorMonkey(Monkey):
    def __init__(self, name: str, operator_symbol: OperatorSymbol, dependency_names: list[str]) -> None:
        super().__init__(name)
        self.__operator_symbol = operator_symbol
        self.__operator = SYMBOL_TO_OPERATOR[operator_symbol]
        self.__dependency_names = dependency_names

    def evaluate(self) -> int:
        dependencies = [NAME_TO_MONKEY_MAP[name] for name in self.__dependency_names]
        evaluated_dependencies = [monkey.evaluate() for monkey in dependencies]
        return functools.reduce(self.__operator, evaluated_dependencies)

    def __repr__(self) -> str:
        return f'{self.name}: {(" " + self.__operator_symbol + " ").join(self.__dependency_names)}'


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


def get_monkeys(raw_input: str) -> list[Monkey]:
    return list(map(parse_monkey, raw_input.split('\n')))


@time_execution
def part_1(raw_input: str) -> None:
    NAME_TO_MONKEY_MAP.clear()
    get_monkeys(raw_input)
    print(f'part_1={NAME_TO_MONKEY_MAP["root"].evaluate()}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)