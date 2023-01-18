"""
https://adventofcode.com/2022/day/16
"""

from functools import cache
import heapq
import re
import time
from typing import TypeVar, ParamSpec, Callable, Iterable


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
    return open('16-proboscidea/input.txt', 'r').read().strip().replace('\r\n', '\n')


def resolve_path(start_label: str, end_label: str, adjacency_map: dict[str, list[str]]) -> list[str]:
    paths: list[tuple[int, list[str]]] = []
    heapq.heappush(paths, (0, [start_label]))
    while len(paths) > 0:
        cur_path_length, cur_path = heapq.heappop(paths)
        if cur_path[-1] == end_label:
            return cur_path
        for next_label in adjacency_map[cur_path[-1]]:
            if next_label in cur_path:
                continue
            heapq.heappush(paths, (cur_path_length + 1, cur_path + [next_label]))
    return []


# DistanceMap = dict[tuple[str, str], int]
PathMap = dict[tuple[str, str], list[str]]


def create_volcano_state(raw_input: str) -> tuple[PathMap, dict[str, int]]:
    adjacency_map: dict[str, list[str]] = dict()
    optimizable_flows_by_label: dict[str, int] = dict()
    valve_labels = []
    for valve_string in raw_input.split('\n'):
        pattern = r'Valve ([A-Z]+) has flow rate=(\d+); tunnels? leads? to valves? (.+)'
        result = re.search(pattern, valve_string)
        assert result is not None, f'Invalid string {valve_string}'
        valve_label = result.group(1)
        valve_flow_rate = int(result.group(2))
        valve_tunnels = result.group(3).split(', ')
        if valve_flow_rate > 0:
            optimizable_flows_by_label[valve_label] = valve_flow_rate
        valve_labels.append(valve_label)
        adjacency_map[valve_label] = valve_tunnels
    distances: PathMap = dict()
    for start in valve_labels:
        for end in valve_labels:
            distances[start, end] = resolve_path(start, end, adjacency_map)
    return distances, optimizable_flows_by_label


path_map, optimizable_flows_by_label = create_volcano_state(get_raw_input())
QUIET = True
def my_print(*values: object) -> None:
    if not QUIET:
        print(*values)

MEMORY: dict[tuple[tuple[str, ...], tuple[str, ...], int], int] = dict()
MEMORY_ZERO: set[tuple[tuple[str, ...], tuple[str, ...], int]] = set()
def get_pressure_released(
    starting_spaces: tuple[str, ...],
    remaining_valves: tuple[str, ...],
    time_remaining: int) -> int:

    key = (starting_spaces, remaining_valves, time_remaining)
    if key in MEMORY_ZERO:
        return 0
    if key in MEMORY:
        return MEMORY[key]

    num_agents = len(starting_spaces)
    num_valves = len(remaining_valves)
    if time_remaining <= 0 or num_agents == 0 or num_valves == 0:
        return 0

    extra_agents = num_agents - num_valves
    if extra_agents > 0:
        # We have more agents than valves, so some agents are gonna nap for their next step
        next_valves: tuple[str | None, ...] = remaining_valves + (None,) * extra_agents
    else:
        next_valves: tuple[str | None, ...] = remaining_valves

    outcomes: list[int] = [0]
    destination_permutations = get_non_repeating_product(next_valves, num_agents)
    for destinations in destination_permutations:
        starts_ends = zip(starting_spaces, destinations)
        paths: list[list[str] | None] = list(map(path_map.get, starts_ends))  # type: ignore
        real_paths = [path for path in paths if path is not None]
        if len(real_paths) == 0:
            continue
        time_to_open_next_valve = min(map(len, real_paths))
        time_remaining_after_valve_open = time_remaining - time_to_open_next_valve
        pressure_released = 0
        next_starting_spaces: list[str] = []
        next_remaining_valves = set(remaining_valves)
        for path in real_paths:
            if len(path) != time_to_open_next_valve:
                next_starting_spaces.append(path[time_to_open_next_valve])
            else:
                opened_valve_label = path[-1]
                next_remaining_valves.remove(opened_valve_label)
                next_starting_spaces.append(opened_valve_label)
                rate = optimizable_flows_by_label[opened_valve_label]
                pressure_released += rate * time_remaining_after_valve_open
        outcome = pressure_released + get_pressure_released(
            tuple(next_starting_spaces), 
            tuple(next_remaining_valves), 
            time_remaining_after_valve_open
        )
        outcomes.append(outcome)
    ret = max(outcomes)
    if ret == 0:
        MEMORY_ZERO.add(key)
    else:
        MEMORY[key] = ret
    return ret


@cache
def get_non_repeating_product(l: Iterable[T], n: int) -> list[tuple[T, ...]]:
    """
    Returns l^n excluding tuples with repeated elements.
    """
    if n == 0:
        return []
    if n == 1:
        return [(el,) for el in l]
    sub_results = get_non_repeating_product(l, n-1)
    result: list[tuple[T, ...]] = []
    for sub_result in sub_results:
        for el in l:
            if el in sub_result and el is not None:
                continue
            result.append(sub_result + (el,))
    return result


@time_execution
def part_1() -> None:
    utility = get_pressure_released(('AA',), tuple(optimizable_flows_by_label.keys()), 30)
    print(f'part_1={utility}')


@time_execution
def part_2() -> None:
    utility = get_pressure_released(('AA', 'AA'), tuple(optimizable_flows_by_label.keys()), 26)
    print(f'part_2={utility}')


if __name__ == '__main__':
    part_1()
    part_2()