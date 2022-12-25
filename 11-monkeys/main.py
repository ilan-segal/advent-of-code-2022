"""
https://adventofcode.com/2022/day/11
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from functools import reduce
import re


def get_raw_input() -> str:
    return open('11-monkeys/input.txt', 'r').read().strip().replace('\r\n', '\n')


@dataclass
class Item:
    worry_level: int


@dataclass
class Monkey:
    id: int
    items: list[Item]
    worry_level_operator: Callable[[int], int]
    test_factor: int
    true_target_id: int
    false_target_id: int
    inspection_count: int = 0

    def take_turn(self, monkeys_by_id: dict[int, Monkey], extra_worrying: bool) -> None:
        M = reduce(lambda x, y: x*y, [m.test_factor for m in monkeys_by_id.values()])
        while len(self.items) > 0:
            cur_item = self.items.pop(0)
            self.__inspect_item(cur_item, extra_worrying)
            """
            It's possible for an item's worry level to blow up very quickly. We
            can keep it bounded by finding its value in a certain modulo group.
            We take advantage of the following fact:

            If:
                a is congruent to b modulo m_i for any m_1, m_2, ..., m_n
            Then:
                a is congruent to b modulo (m_1 * m_2 * ... * m_n)

            In the context of this problem, m_1 ... m_n correspond to each
            monkey's test factor. M is the product of all the test factors.
            """
            cur_item.worry_level %= M
            self.__throw_item(cur_item, monkeys_by_id)
            
    def __inspect_item(self, item: Item, extra_worrying: bool) -> None:
        cur_worry_level = item.worry_level
        cur_worry_level = self.worry_level_operator(cur_worry_level)
        if not extra_worrying:
            cur_worry_level //= 3
        item.worry_level = cur_worry_level
        self.inspection_count += 1

    def __throw_item(self, item: Item, monkeys_by_id: dict[int, Monkey]) -> None:
        is_divisible = item.worry_level % self.test_factor == 0
        target_id = self.true_target_id if is_divisible else self.false_target_id
        monkeys_by_id[target_id].items.append(item)


def create_monkey(string: str) -> Monkey:
    pattern = r'''Monkey (\d+):
  Starting items: ([\d, ]+)
  Operation: new = old ([\*/\+-]) (\d+|old)
  Test: divisible by (\d+)
    If true: throw to monkey (\d+)
    If false: throw to monkey (\d+)'''
    result = re.search(pattern, string)
    assert result is not None, f'Invalid monkey string:\n{string}'
    monkey_id = int(result.group(1))
    items = list(map(Item, map(int, result.group(2).split(', '))))
    OPERATOR_FACTORY = {
        '+': lambda x, y: x + y,
        '*': lambda x, y: x * y,
        '-': lambda x, y: x - y,
        '/': lambda x, y: x // y,
    }
    operator_symbol = result.group(3)
    operand = result.group(4)
    if operand == 'old':
        operator = lambda x: OPERATOR_FACTORY[operator_symbol](x, x)
    else:
        operand = int(operand)
        operator = lambda x: OPERATOR_FACTORY[operator_symbol](x, operand)
    test_factor = int(result.group(5))
    true_id = int(result.group(6))
    false_id = int(result.group(7))
    return Monkey(monkey_id, items, operator, test_factor, true_id, false_id)


@dataclass
class MonkeyPack:
    monkeys: list[Monkey]

    def __play_round(self, extra_worrying: bool) -> None:
        monkeys_by_id = {m.id: m for m in self.monkeys}
        for monkey in self.monkeys:
            monkey.take_turn(monkeys_by_id, extra_worrying)

    def play_n_rounds(self, n: int, extra_worrying: bool) -> None:
        for _ in range(n):
            self.__play_round(extra_worrying)

    def get_monkey_business(self) -> int:
        inspection_counts = [m.inspection_count for m in self.monkeys]
        a, b = sorted(inspection_counts)[-2:]
        return a * b


def create_monkey_pack(raw_input: str) -> MonkeyPack:
    monkey_strings = raw_input.split('\n\n')
    monkeys = list(map(create_monkey, monkey_strings))
    return MonkeyPack(monkeys)


def part_1(raw_input: str) -> None:
    monkey_pack = create_monkey_pack(raw_input)
    monkey_pack.play_n_rounds(20, False)
    monkey_business = monkey_pack.get_monkey_business()
    print(f'part_1={monkey_business}')


def part_2(raw_input: str) -> None:
    monkey_pack = create_monkey_pack(raw_input)
    monkey_pack.play_n_rounds(10_000, True)
    monkey_business = monkey_pack.get_monkey_business()
    print(f'part_2={monkey_business}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)
    part_2(raw_input)