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
            if start == end:
                continue
            distances[start, end] = resolve_path(start, end, adjacency_map)
    return distances, optimizable_flows_by_label


# @cache
def get_pressure_released(current_valve: str, t: int, remaining_rates: dict[str, int], paths: PathMap) -> int:
    MAX_TIME = 30
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
            t + len(paths[(current_valve, next_valve)]) - 1,
            remaining_rates,
            paths,
        )
        for next_valve in remaining_rates.keys()
    ])


def get_non_repeating_product(l: list[T], n: int) -> list[tuple[T, ...]]:
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


def get_distance(start: str | None, end: str | None, path_map: PathMap) -> int | float:
    if start is None or end is None:
        return float('inf')
    if (start, end) not in path_map:
        return float('inf')
    return len(path_map[(start, end)]) - 1


def get_minimum_distance(starts: list[str | None], ends: list[str | None], path_map: PathMap) -> int | float:
    zipped = zip(starts, ends)
    distance = lambda t: get_distance(t[0], t[1], path_map)
    return min(map(distance, zipped))


def get_pressure_released_multiple_agents(
    agent_current_valves: list[str | None],
    t: int,
    remaining_rates: dict[str, int],
    path_map: PathMap) -> int:

    MAX_TIME = 26
    if t >= MAX_TIME or len(remaining_rates) == 0:
        return 0
    remaining_destinations = list(remaining_rates.keys())
    num_destinations = len(remaining_rates)
    num_agents = len(agent_current_valves)
    if num_agents > num_destinations:
        # We have more agents than closed valves. Some agents will take a nap (current valve = None)
        remaining_destinations = remaining_destinations + [None]*(num_agents - num_destinations)
    
    def napper_stays_napping(positions: tuple[str | None, str | None]) -> bool:
        return positions[0] is not None or positions[1] is None

    destination_permutations = get_non_repeating_product(remaining_destinations, num_agents)  # type: ignore
    # Deduplicate
    destination_permutations = list(set(destination_permutations))
    subsequent_results: list[int] = [0]
    for destinations in destination_permutations:
        zipped_start_end = zip(agent_current_valves, destinations)
        nappers_stay_napping = map(napper_stays_napping, zipped_start_end)
        if not all(nappers_stay_napping):
            # Don't wake them up!
            continue
        paths: map[list[str] | None] = map(path_map.get, zipped_start_end)  # type: ignore
        calculate_distance = lambda t: get_distance(t[0], t[1], path_map)
        distances = map(calculate_distance, zipped_start_end)
        minimum_distance = min(distances)
    return max(subsequent_results)


@time_execution
def part_1(raw_input: str) -> None:
    distances, optimizable_flows_by_label = create_volcano_state(raw_input)
    # print(distances)
    # print(optimizable_flows_by_label)
    utility = get_pressure_released('AA', 1, optimizable_flows_by_label, distances)
    print(f'part_1={utility}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)