"""
https://adventofcode.com/2022/day/8
"""


def get_raw_input() -> str:
    return open('08-tree-house/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_tree_grid(raw_input: str) -> list[list[int]]:
    parse_row = lambda row: list(map(int, row))
    return list(map(parse_row, raw_input.split('\n')))


if __name__ == '__main__':
    pass