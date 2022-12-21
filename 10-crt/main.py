"""
https://adventofcode.com/2022/day/10
"""


CRT_DISPLAY_WIDTH = 40


def get_raw_input() -> str:
    return open('10-crt/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_register_series(commands: list[str]) -> list[int]:
    """
    One value per clock cycle
    """
    values: list[int] = []
    x_register = 1
    for command in commands:
        values.append(x_register)
        if command == 'noop':
            continue
        else:
            num_to_add = int(command.split(' ')[-1])
            values.append(x_register + num_to_add)
            x_register += num_to_add
    return ([1] + values)[:-1]


def part_1(commands: list[str]) -> None:
    signal_strength_cycles = [20 + (40 * i) for i in range(6)]
    x_register_series = get_register_series(commands)
    signal_strength_sum = sum([(cycle_num * x_register_series[cycle_num-1]) for cycle_num in signal_strength_cycles])
    print(f'part_1={signal_strength_sum}')


def should_draw_pixel(cycle_num: int, sprite_pos: int) -> bool:
    """
    Cycle num is 0-indexed here
    """
    crt_ray_pos = cycle_num % CRT_DISPLAY_WIDTH
    return abs(crt_ray_pos - sprite_pos) <= 1


def part_2(commands: list[str]) -> None:
    xs = get_register_series(commands)
    for cycle_num, x_register in enumerate(xs):
        print('#' if should_draw_pixel(cycle_num, x_register) else '.', end='')
        if (cycle_num + 1) % 40 == 0:
            print('')


if __name__ == '__main__':
    commands = get_raw_input().split('\n')
    part_1(commands)
    part_2(commands)