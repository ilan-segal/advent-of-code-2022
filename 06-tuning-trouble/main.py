"""
https://adventofcode.com/2022/day/6
"""


def get_raw_input() -> str:
    return open('06-tuning-trouble/input.txt', 'r').read().strip().replace('\r\n', '\n')


def find_unique_sequence(string: str, sequence_length: int) -> int:
    for index in range(0, len(string) - sequence_length + 1):
        cur_packet = string[index : index + sequence_length]
        if len(cur_packet) == len(set(cur_packet)):
            return index + sequence_length
    return -1


def part_1():
    signal = get_raw_input()
    print(f'part_1={find_unique_sequence(signal, 4)}')


def part_2():
    signal = get_raw_input()
    print(f'part_2={find_unique_sequence(signal, 14)}')


if __name__ == '__main__':
    part_1()
    part_2()