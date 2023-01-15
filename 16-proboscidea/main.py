"""
https://adventofcode.com/2022/day/16
"""

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


def cache(f: Callable[P, T]) -> Callable[P, T]:
    memo: dict[str, T] = dict()
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        key = str(args)+','+str(kwargs)
        if key in memo:
            return memo[key]
        result = f(*args, **kwargs)
        memo[key] = result
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


path_map, optimizable_flows_by_label = create_volcano_state(get_raw_input())


@cache
def get_pressure_released(
    agent_current_valves: list[str | None],
    t: int,
    remaining_rates: dict[str, int],
    max_time: int) -> int:

    # print('#'*10)
    # print(f'{agent_current_valves=}')
    # print(f'{t=}')
    # print(f'{remaining_rates=}')

    if t >= max_time or len(remaining_rates) == 0:
        # print('shorting')
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
        zipped_start_end = list(zip(agent_current_valves, destinations))
        # print(f'{zipped_start_end=}')
        nappers_stay_napping = map(napper_stays_napping, zipped_start_end)
        if not all(nappers_stay_napping):
            # Don't wake them up!
            # print('Dont wake em up')
            continue
        # print(f'{destinations=}')
        paths: list[list[str] | None] = list(map(path_map.get, zipped_start_end))  # type: ignore
        # print(f'{paths=}')
        calculate_distance = lambda t: get_distance(t[0], t[1], path_map)
        distances = list(map(calculate_distance, zipped_start_end))
        # print(f'{distances=}')
        minimum_distance = int(min(distances))
        length_of_activated_path = minimum_distance + 1
        actual_destinations: list[str | None] = []
        activated_valves: list[str] = []
        # Step forward to next time an agent activates a valve. The other agents
        # move as far as they can in that amount of elapsed time.
        for path in paths:
            if path is None:
                actual_destinations.append(None)
            elif len(path) == length_of_activated_path:
                # Track the valves which get activated by the end of the step
                actual_destinations.append(path[-1])
                activated_valves.append(path[-1])
            else:
                actual_destinations.append(path[length_of_activated_path])
        assert all(map(remaining_rates.__contains__, activated_valves))
        pressure_released = (max_time - t - minimum_distance) * sum(map(remaining_rates.get, activated_valves))  # type: ignore
        new_remaining = {k: v for k, v in remaining_rates.items() if k not in activated_valves}
        subsequent_results.append(pressure_released + get_pressure_released(actual_destinations, t+length_of_activated_path, new_remaining, max_time))
    return max(subsequent_results)


@time_execution
def part_1() -> None:
    utility = get_pressure_released(['AA'], 1, optimizable_flows_by_label, 30)
    print(f'part_1={utility}')


if __name__ == '__main__':
    part_1()