"""
https://adventofcode.com/2022/day/16
"""

from functools import cache
import heapq
import re
import time
from typing import TypeVar, ParamSpec, Callable

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


def resolve_distance(start_label: str, end_label: str, adjacency_map: dict[str, list[str]]) -> int:
    paths: list[tuple[int, list[str]]] = []
    heapq.heappush(paths, (0, [start_label]))
    while len(paths) > 0:
        cur_path_length, cur_path = heapq.heappop(paths)
        if cur_path[-1] == end_label:
            return cur_path_length
        for next_label in adjacency_map[cur_path[-1]]:
            if next_label in cur_path:
                continue
            heapq.heappush(paths, (cur_path_length + 1, cur_path + [next_label]))
    return -1


DistanceMap = dict[tuple[str, str], int]


def create_volcano_state(raw_input: str) -> tuple[DistanceMap, dict[str, int]]:
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
    distances: DistanceMap = dict()
    for start in valve_labels:
        for end in valve_labels:
            if start == end:
                continue
            distances[start, end] = resolve_distance(start, end, adjacency_map)
    return distances, optimizable_flows_by_label


MAX_TIME = 30

# @cache
def get_pressure_released(current_valve: str, t: int, remaining_rates: dict[str, int], distances: DistanceMap) -> int:
    if t >= MAX_TIME or len(remaining_rates) == 0:
        return 0
    if current_valve in remaining_rates:
        pressure_released = remaining_rates[current_valve] * (MAX_TIME - t)
        t += 1
    else:
        pressure_released = 0
    remaining_rates = {valve: rate for valve, rate in remaining_rates.items() if valve != current_valve}
    return pressure_released + max([0] + [
        get_pressure_released(
            next_valve, 
            t + distances[(current_valve, next_valve)],
            remaining_rates,  # type: ignore
            distances,  # type: ignore
        )
        for next_valve in remaining_rates.keys()
    ])


@time_execution
def part_1(raw_input: str) -> None:
    distances, optimizable_flows_by_label = create_volcano_state(raw_input)
    print(distances)
    print(optimizable_flows_by_label)
    utility = get_pressure_released('AA', 1, optimizable_flows_by_label, distances)  # type: ignore
    print(f'part_1={utility}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)