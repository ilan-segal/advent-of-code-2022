"""
https://adventofcode.com/2022/day/17
"""

import time
from typing import TypeVar, ParamSpec, Callable, Literal, Generator, Type

Position = tuple[int, int]
JetSymbol = Literal['<', '>']

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
    return open('17-pyroclastic/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_jet_sequence(raw_input: str) -> list[JetSymbol]:
    return list(raw_input)  # type: ignore


class Rock:

    _pos: Position

    def __init__(self, structure_height: int) -> None:
        self._pos = (0, 0)
        # Implementation-agnostic way to set 2 units away from left wall and 3 units above structure
        distance_from_left_wall = min(map(lambda pos: pos[0], self.get_spaces()))
        distance_from_floor = min(map(lambda pos: pos[1], self.get_spaces())) - structure_height
        self.__move((
            -distance_from_left_wall + 2,
            -distance_from_floor + 3
        ))

    def get_spaces(self) -> tuple[Position]:
        raise NotImplementedError
            
    def __move(self, vector: tuple[int, int]) -> None:
        (px, py), (vx, vy) = self._pos, vector
        px += vx
        py += vy
        self._pos = px, py

    def __collides_with_obstacle(self, occupied_spaces: set[Position], min_x: int, max_x: int) -> bool:
        spaces = self.get_spaces()
        out_of_bounds = lambda pos: pos[0] < min_x or pos[0] > max_x
        return (
            any(map(out_of_bounds, spaces)) 
            or len(occupied_spaces.intersection(spaces)) > 0
            or min(pos[1] for pos in spaces) < 0
        )

    def advance(self, jet: JetSymbol, occupied_spaces: set[Position], min_x: int, max_x: int) -> bool:
        """
        Return True iff rock can continue falling, False otherwise
        """
        jet_blow_x_direction = {'>': 1, '<': -1}[jet]
        self.__move((jet_blow_x_direction, 0))
        if self.__collides_with_obstacle(occupied_spaces, min_x, max_x):
            self.__move((-jet_blow_x_direction, 0))
        self.__move((0, -1))
        if self.__collides_with_obstacle(occupied_spaces, min_x, max_x):
            self.__move((0, 1))
            return False
        return True

    def __repr__(self) -> str:
        return f'{type(self).__name__}: {self.get_spaces()}'


class HorizontalRock(Rock):
    def get_spaces(self) -> tuple[Position]:
        x, y = self._pos
        return tuple((x+i, y) for i in range(4))
    

class VerticalRock(Rock):
    def get_spaces(self) -> tuple[Position]:
        x, y = self._pos
        return tuple((x, y+i) for i in range(4))


class PlusRock(Rock):
    def get_spaces(self) -> tuple[Position]:
        x, y = self._pos
        return tuple((x+i, y) for i in [-1, 0, 1]) + tuple((x, y+i) for i in [-1, 1])


class JayRock(Rock):
    def get_spaces(self) -> tuple[Position]:
        x, y = self._pos
        return tuple((x+i, y) for i in [-2, -1, 0]) + tuple((x, y+i) for i in [2, 1, 0])


class SquareRock(Rock):
    def get_spaces(self) -> tuple[Position]:
        x, y = self._pos
        return tuple((x+a, y+b) for a in [0, 1] for b in [0, 1])


ROCK_SEQUENCE: list[Type[Rock]] = [
    HorizontalRock, PlusRock, JayRock, VerticalRock, SquareRock
]
CHAMBER_WIDTH = 7

class Chamber:

    occupied_points: set[Position]
    cur_rock: Rock
    last_rock: Rock
    structure_height: int
    num_dropped: int
    __rock_sequence_next_index: int
    __jet_sequence: Generator[JetSymbol, None, None]

    def __init__(self, jet_sequence: list[JetSymbol]) -> None:
        self.occupied_points = set()
        self.__rock_sequence_next_index = 0
        self.structure_height = 0
        self.cur_rock = None  # type: ignore
        self.cur_rock = self.__get_next_rock()
        def jet_sequence_gen() -> Generator[JetSymbol, None, None]:
            cur_idx = 0
            while True:
                yield jet_sequence[cur_idx]
                cur_idx = (cur_idx + 1) % len(jet_sequence)
        self.__jet_sequence = jet_sequence_gen()
        self.last_rock = None  # type: ignore
        self.num_dropped = 0

    def drop_rock(self) -> None:
        while self.cur_rock.advance(
            jet=next(self.__jet_sequence), 
            occupied_spaces=self.occupied_points,
            min_x=0,
            max_x=CHAMBER_WIDTH-1,
        ): pass #print(self.__cur_rock)
        self.occupied_points = self.occupied_points.union(self.cur_rock.get_spaces())
        self.structure_height = max(pos[1] for pos in self.occupied_points) + 1
        self.cur_rock = self.__get_next_rock()
        self.num_dropped += 1

    def __get_next_rock(self) -> Rock:
        self.last_rock = self.cur_rock
        next_rock = ROCK_SEQUENCE[self.__rock_sequence_next_index](self.structure_height)
        self.__rock_sequence_next_index += 1
        if self.__rock_sequence_next_index >= len(ROCK_SEQUENCE):
            self.__rock_sequence_next_index = 0
        return next_rock

    def __repr__(self) -> str:
        cur_rock_points = self.cur_rock.get_spaces()
        all_points = set(cur_rock_points).union(self.occupied_points)
        window_height = max(pos[1] for pos in all_points)
        result = ''
        for y in range(window_height, -1, -1):
            result += '|'
            for x in range(CHAMBER_WIDTH):
                if (x, y) in cur_rock_points:
                    result += '@'
                elif (x, y) in self.occupied_points:
                    result += '#'
                else:
                    result += '.'
            result += '|\n'
        result += '+' + ('-' * CHAMBER_WIDTH) + '+\n'
        return result


def find_period(jet_sequence: list[JetSymbol]) -> tuple[int, int]:
    """
    Returns period length and start of periodicity.
    Periodicity here is defined as the behaviour by which a chamber of falling
    rocks will be guaranteed to demonstrate the same increase in structure height
    over a constant period length.

    E.g. If this function returns (14, 11) then the chamber height becomes periodic
    after dropping the 11th rock, after which point the structure height increases
    by a constant amount with every 14 rocks dropped.
    """
    queue_length = len(ROCK_SEQUENCE)
    most_recent_rock_configurations: list[tuple[tuple[int, int], ...]] = []
    # Key: x-coordinates of 7 rocks
    # Value: num rocks dropped
    past_rock_configurations: dict[tuple[tuple[tuple[int, int], ...], ...], int] = {}
    chamber = Chamber(jet_sequence)
    while True:
        chamber.drop_rock()
        min_y = min(pos[1] for pos in chamber.last_rock.get_spaces())
        most_recent_rock_configurations.append(
            tuple((pos[0], pos[1]-min_y) for pos in chamber.last_rock.get_spaces())
        )
        while len(most_recent_rock_configurations) > queue_length:
            most_recent_rock_configurations.pop(0)
        key = tuple(most_recent_rock_configurations)
        if (
            chamber.structure_height > 3127
            and (period_start := past_rock_configurations.get(key)) is not None 
            and (period_length := (chamber.num_dropped - period_start)) % len(ROCK_SEQUENCE) == 0
        ):
            # TODO: Found start of periodic behaviour
            assert period_length % len(ROCK_SEQUENCE) == 0
            return period_length, period_start
        else:
            past_rock_configurations[key] = chamber.num_dropped


def get_height_after_n_drops(jet_sequence: list[JetSymbol], n: int) -> int:
    period_length, period_start = find_period(jet_sequence)
    print(f'{len(jet_sequence)=} {len(ROCK_SEQUENCE)=} {period_length=} {period_start=}')
    chamber = Chamber(jet_sequence)
    for _ in range(period_start):
        chamber.drop_rock()
    height_before_first_period = chamber.structure_height
    for _ in range(period_length):
        chamber.drop_rock()
    height_after_first_period = chamber.structure_height
    height_change_per_period_elapsed = height_after_first_period - height_before_first_period
    remaining_drops = n - chamber.num_dropped
    # Align end of n drops with period
    drops_to_align = remaining_drops % period_length
    for _ in range(drops_to_align):
        chamber.drop_rock()
    remaining_drops = n - chamber.num_dropped
    assert remaining_drops % len(ROCK_SEQUENCE) == 0, f'{remaining_drops=} {period_length=}'
    assert remaining_drops % period_length == 0, f'{remaining_drops=} {period_length=}'
    num_periods_remaining = remaining_drops // period_length
    return chamber.structure_height + (num_periods_remaining * height_change_per_period_elapsed)

    
@time_execution
def part_1(raw_input: str) -> None:
    jet_sequence = get_jet_sequence(raw_input)
    height = get_height_after_n_drops(jet_sequence, 2022)
    print(f'part_1={height}')

    
@time_execution
def part_2(raw_input: str) -> None:
    jet_sequence = get_jet_sequence(raw_input)
    height = get_height_after_n_drops(jet_sequence, 1_000_000_000_000)
    print(f'part_2={height}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)
    part_2(raw_input)