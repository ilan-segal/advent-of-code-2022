"""
https://adventofcode.com/2022/day/9
"""


def get_raw_input() -> str:
    return open('09-rope-bridge/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_unit_steps(raw_input: str) -> list[tuple[int, int]]:
    direction_name_to_vector: dict[str, tuple[int, int]] = {
        'U': (0, -1),
        'D': (0, 1),
        'R': (1, 0),
        'L': (-1, 0),
    }
    unit_steps: list[tuple[int, int]] = []
    for step in raw_input.split('\n'):
        direction_name, step_length = step.split(' ')
        vector = direction_name_to_vector[direction_name]
        unit_steps += [vector] * int(step_length)
    return unit_steps


def add_positions(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
    return x[0] + y[0], x[1] + y[1]


def get_new_tail_pos(head_pos: tuple[int, int], tail_pos: tuple[int, int]) -> tuple[int, int]:
    head_x, head_y = head_pos
    tail_x, tail_y = tail_pos
    x_dist = head_x - tail_x
    y_dist = head_y - tail_y
    if abs(x_dist) < 2 and abs(y_dist) < 2:
        return tail_pos
    x_dist_unit = 0 if x_dist == 0 else (x_dist // abs(x_dist))
    y_dist_unit = 0 if y_dist == 0 else (y_dist // abs(y_dist))
    return add_positions(tail_pos, (x_dist_unit, y_dist_unit))


def part_1(unit_steps: list[tuple[int, int]]) -> None:
    head_pos, tail_pos = (0, 0), (0, 0)
    prev_tail_positions: set[tuple[int, int]] = {tail_pos}
    for step in unit_steps:
        head_pos = add_positions(head_pos, step)
        tail_pos = get_new_tail_pos(head_pos, tail_pos)
        prev_tail_positions.add(tail_pos)
    num_unique_tail_positions = len(prev_tail_positions)
    print(f'part_1={num_unique_tail_positions}')


def part_2(unit_steps: list[tuple[int, int]]) -> None:
    NUM_KNOTS = 10
    knot_positions: list[tuple[int, int]] = [(0, 0) for _ in range(10)]
    prev_tail_positions = {knot_positions[-1]}
    for step in unit_steps:
        knot_positions[0] = add_positions(knot_positions[0], step)
        for knot_index in range(NUM_KNOTS - 1):
            head = knot_positions[knot_index]
            tail = knot_positions[knot_index + 1]
            new_tail = get_new_tail_pos(head, tail)
            knot_positions[knot_index + 1] = new_tail
        prev_tail_positions.add(knot_positions[-1])
    num_unique_tail_positions = len(prev_tail_positions)
    print(f'part_2={num_unique_tail_positions}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    unit_steps = get_unit_steps(raw_input)
    part_1(unit_steps)
    part_2(unit_steps)