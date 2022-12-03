"""
https://adventofcode.com/2022/day/3
"""

from functools import reduce

PUZZLE_INPUT = open('03-rucksack-reorganization/input.txt', 'r').read()


def get_item_priority(item_type: str) -> int:
    assert len(item_type) == 1, f'Expected single character item type, got {item_type}'
    assert item_type.isalpha(), f'Expected alphabetical character, got {item_type}'
    if 'a' <= item_type <= 'z':
        return ord(item_type) - ord('a') + 1
    else:
        return ord(item_type) - ord('A') + 27


def get_types_in_compartments(rucksack_items: str) -> tuple[set[str], set[str]]:
    n = len(rucksack_items)
    halves = rucksack_items[:n//2], rucksack_items[n//2:]
    return tuple(map(set, halves))


def find_common_item_in_sets(compartments: tuple[set[str], ...]) -> str:
    common_types = reduce(set.intersection, compartments)
    assert len(common_types) == 1, f'More than one common type! {common_types}'
    return list(common_types)[0]


def part_1():
    rucksacks = PUZZLE_INPUT.strip().split('\n')
    compartment_types_per_sack = map(get_types_in_compartments, rucksacks)
    common_type_per_sack = map(find_common_item_in_sets, compartment_types_per_sack)
    priorities_part_1 = map(get_item_priority, common_type_per_sack)
    print(f'{sum(priorities_part_1)=}')


def divide_into_groups(rucksacks: list[str]) -> list[tuple[set[str], set[str], set[str]]]:
    num_sacks = len(rucksacks)
    assert num_sacks % 3 == 0, f'Num of rucksacks must be divisible by 3, got {num_sacks}'
    return [tuple(map(set, rucksacks[i:i+3])) for i in range(0, num_sacks, 3)]


def part_2():
    rucksacks = PUZZLE_INPUT.strip().split('\n')
    sack_items_per_group = divide_into_groups(rucksacks)
    badges = map(find_common_item_in_sets, sack_items_per_group)
    priorities_part_2 = map(get_item_priority, badges)
    print(f'{sum(priorities_part_2)=}')


if __name__ == '__main__':
    part_1()
    part_2()