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
from geometry import (
    Vector2D,
    add_vectors,
    multiply_vectors,
    Heading,
)

import abc
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


class AbstractPasswordBoard(abc.ABC):

    _open_spaces: set[Vector2D]
    _closed_spaces: set[Vector2D]
    _trace_position: Vector2D
    _heading: Heading

    def __init__(self, raw_representation: str) -> None:
        self._open_spaces = set()
        self._closed_spaces = set()
        for y, line in enumerate(raw_representation.split('\n')):
            for x, character in enumerate(line):
                assert character in [' ', '.', '#']
                if character == '.':
                    self._open_spaces.add((x, y))
                elif character == '#':
                    self._closed_spaces.add((x, y))
        top_row_open_spaces = {(x, y) for (x, y) in self._open_spaces if y == 0}
        leftmost_open_tile_x = min(x for x, _ in top_row_open_spaces)
        self._trace_position = (leftmost_open_tile_x, 0)
        self._heading = Heading.Right

    def __repr__(self) -> str:
        all_spaces = self._open_spaces.union(self._closed_spaces)
        max_x = max(x for x, _ in all_spaces)
        max_y = max(y for _, y in all_spaces)
        result = ''
        for y in range(max_y + 1):
            for x in range(max_x + 1):
                pos = x, y
                if pos == self._trace_position:
                    result += 'O'
                elif pos in self._open_spaces:
                    result += '.'
                elif pos in self._closed_spaces:
                    result += '#'
                else:
                    result += ' '
            result += '\n'
        return result.rstrip()

    @abc.abstractmethod
    def _wrap_around(self, position: Vector2D, heading: Heading) -> tuple[Vector2D, Heading]:
        pass
        
    def turn(self, direction: Literal['R', 'L']) -> None:
        self._heading = self._heading.rotated(direction)

    def move_forward(self, distance: int) -> None:
        cur_position = self._trace_position
        for _ in range(distance):
            next_position = add_vectors(cur_position, self._heading.get_vector())
            next_heading = self._heading
            if not (next_position in self._open_spaces or next_position in self._closed_spaces):
                next_position, next_heading = self._wrap_around(cur_position, self._heading)                
            if next_position in self._closed_spaces:
                break
            cur_position = next_position
            self._heading = next_heading
        self._trace_position = cur_position

    def get_password(self) -> int:
        return (
            1000 * (self._trace_position[1] + 1)
            + 4 * (self._trace_position[0] + 1)
            + self._heading.value
        )


class FlatPasswordBoard(AbstractPasswordBoard):

    def _wrap_around(self, position: Vector2D, heading: Heading) -> tuple[Vector2D, Heading]:
        opposite_movement_vector = multiply_vectors(heading.get_vector(), -1)
        next_position = position
        while next_position in self._open_spaces or next_position in self._closed_spaces:
            next_position = add_vectors(next_position, opposite_movement_vector)
        next_position = add_vectors(next_position, heading.get_vector())
        return next_position, heading


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
    board = FlatPasswordBoard(board_input)
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