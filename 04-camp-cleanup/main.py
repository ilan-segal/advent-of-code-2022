"""
https://adventofcode.com/2022/day/4
"""


def parse_raw_input_line(line: str) -> tuple[int, int, int, int]:
    first, second = line.split(',')
    first_a, first_b = map(int, first.split('-'))
    second_a, second_b = map(int, second.split('-'))
    return first_a, first_b, second_a, second_b


def get_pairings() -> list[tuple[int, int, int, int]]:
    """
    Return list of tuples where each tuple represents a pair of elves and the
    range of section IDs they were assigned.
    """
    raw_input = open('04-camp-cleanup/input.txt', 'r').read().strip().replace('\r\n', '\n')
    return list(map(parse_raw_input_line, raw_input.split('\n')))


def pairing_has_full_containment(pairing: tuple[int, int, int, int]) -> bool:
    a, b, c, d = pairing
    return (
        a <= c <= d <= b
        or
        c <= a <= b <= d
    )


def pairing_has_overlap(pairing: tuple[int, int, int, int]) -> bool:
    a, b, c, d = pairing
    return (
        a <= c <= b
        or
        a <= d <= b
        or
        c <= a <= d
        or
        c <= b <= d
    )


def part_1():
    pairings = get_pairings()
    full_contains_count = len(list(filter(pairing_has_full_containment, pairings)))
    print(full_contains_count)


def part_2():
    pairings = get_pairings()
    overlap_count = len(list(filter(pairing_has_overlap, pairings)))
    print(overlap_count)


if __name__ == '__main__':
    part_1()
    part_2()