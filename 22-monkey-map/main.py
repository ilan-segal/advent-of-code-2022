"""
https://adventofcode.com/2022/day/22
"""

from __future__ import annotations
from typing import (
    Callable,
    Literal,
    ParamSpec,
    TypeVar,
)

import enum
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
    return open('22-monkey-map/input.txt', 'r').read().replace('\r\n', '\n')


Vector2D = tuple[int, int]


def _add(a: Vector2D, b: Vector2D) -> Vector2D:
    (ax, ay), (bx, by) = a, b
    return (ax + bx), (ay + by)


def _mul(v: Vector2D, m: int) -> Vector2D:
    x, y = v
    return (m*x, m*y)


class Heading(enum.Enum):

    Right = 0
    Down = 1
    Left = 2
    Up = 3

    def get_vector(self) -> Vector2D:
        return {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1),
        }[self.value]

    def rotated(self, direction: Literal['R', 'L']) -> Heading:
        turn_direction = 1 if direction == 'R' else -1
        return Heading((self.value + turn_direction) % 4)


class PasswordBoard:

    __open_spaces: set[Vector2D]
    __closed_spaces: set[Vector2D]
    __trace_position: Vector2D
    __heading: Heading

    def __init__(self, raw_representation: str) -> None:
        self.__open_spaces = set()
        self.__closed_spaces = set()
        for y, line in enumerate(raw_representation.split('\n')):
            for x, character in enumerate(line):
                assert character in [' ', '.', '#']
                if character == '.':
                    self.__open_spaces.add((x, y))
                elif character == '#':
                    self.__closed_spaces.add((x, y))
        top_row_open_spaces = {(x, y) for (x, y) in self.__open_spaces if y == 0}
        leftmost_open_tile_x = min(x for x, _ in top_row_open_spaces)
        self.__trace_position = (leftmost_open_tile_x, 0)
        self.__heading = Heading.Right

    def __repr__(self) -> str:
        all_spaces = self.__open_spaces.union(self.__closed_spaces)
        max_x = max(x for x, _ in all_spaces)
        max_y = max(y for _, y in all_spaces)
        result = ''
        for y in range(max_y + 1):
            for x in range(max_x + 1):
                pos = x, y
                if pos == self.__trace_position:
                    result += 'O'
                elif pos in self.__open_spaces:
                    result += '.'
                elif pos in self.__closed_spaces:
                    result += '#'
                else:
                    result += ' '
            result += '\n'
        return result.rstrip()
        
    def turn(self, direction: Literal['R', 'L']) -> None:
        self.__heading = self.__heading.rotated(direction)

    def move_forward(self, distance: int) -> None:
        cur_position = self.__trace_position
        movement_vector = self.__heading.get_vector()
        for _ in range(distance):
            next_position = _add(cur_position, movement_vector)
            if not (next_position in self.__open_spaces or next_position in self.__closed_spaces):
                # Wrap around
                opposite_movement_vector = _mul(movement_vector, -1)
                next_position = cur_position
                while next_position in self.__open_spaces or next_position in self.__closed_spaces:
                    next_position = _add(next_position, opposite_movement_vector)
                # After loop terminates, we've gone off grid again. So take one more step back onto grid.
                next_position = _add(next_position, movement_vector)
            if next_position in self.__closed_spaces:
                break
            cur_position = next_position
        self.__trace_position = cur_position

    def get_password(self) -> int:
        return (
            1000 * (self.__trace_position[1] + 1)
            + 4 * (self.__trace_position[0] + 1)
            + self.__heading.value
        )


def _parsed_instructions(movement_input: str) -> list[int | Literal['R', 'L']]:
    instructions = re.findall(r'\d+|[RL]', movement_input.strip())
    result = []
    for item in instructions:
        assert isinstance(item, str)
        assert item in ['R', 'L'] or item.isnumeric()
        if item.isnumeric():
            item = int(item)
        result.append(item)
    return result


@time_execution
def part_1(raw_input: str) -> None:
    board_input, movement_input = raw_input.split('\n\n')
    board = PasswordBoard(board_input)
    for instruction in _parsed_instructions(movement_input):
        if isinstance(instruction, str):
            board.turn(instruction)
        else:
            board.move_forward(instruction)
    print(f'part_1={board.get_password()}')


@time_execution
def part_2(raw_input: str) -> None:
    pass


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)
    part_2(raw_input)