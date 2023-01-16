"""
https://adventofcode.com/2022/day/16
"""

from functools import cache
import heapq
import re
import time
from typing import TypeVar, ParamSpec, Callable
from tqdm import tqdm


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


path_map, optimizable_flows_by_label = create_volcano_state(get_raw_input())
QUIET = True
def my_print(*values: object) -> None:
    if not QUIET:
        print(*values)


@cache
def get_non_repeating_product(l: tuple[T, ...], n: int) -> tuple[tuple[T, ...]]:
    """
    Returns l^n excluding tuples with repeated elements.
    """
    if n == 0:
        return tuple()
    if n == 1:
        return tuple((el,) for el in l)
    sub_results = get_non_repeating_product(l, n-1)
    result: list[tuple[T, ...]] = []
    for sub_result in sub_results:
        for el in l:
            if el in sub_result and el is not None:
                continue
            result.append(sub_result + (el,))
    return tuple(result)


@cache
def get_distance(start: str | None, end: str | None) -> int | float:
    if start is None or end is None:
        return float('inf')
    if (start, end) not in path_map:
        return float('inf')
    return len(path_map[(start, end)]) - 1


@cache
def get_minimum_distance(starts: list[str | None], ends: list[str | None]) -> int | float:
    zipped = zip(starts, ends)
    distance = lambda t: get_distance(t[0], t[1])
    return min(map(distance, zipped))


cached_results: dict[tuple[str,...], int] = dict()


def get_pressure_released(
    agent_current_valves: tuple[str | None, ...],
    remaining_rates: dict[str, int],
    remaining_time: int) -> int:

    key = (
        tuple(sorted(map(str, agent_current_valves))), 
        tuple(remaining_rates.keys()),
        remaining_time,
    )
    if key in cached_results:
        # print(f'Cache hit: {key} -> {cached_results[key]}')
        return cached_results[key]

    my_print('#'*10)
    my_print(f'{agent_current_valves=}')
    my_print(f'{remaining_rates=}')
    my_print(f'{remaining_time=}')

    if remaining_time <= 1 or len(remaining_rates) == 0:
        my_print('shorting')
        cached_results[key] = 0
        return 0
    remaining_destinations = tuple(remaining_rates.keys())
    num_destinations = len(remaining_destinations)
    num_agents = len(agent_current_valves)
    if num_agents > num_destinations:
        # We have more agents than closed valves. Some agents will take a nap (current valve = None)
        remaining_destinations = remaining_destinations + (None,)*(num_agents - num_destinations)
    
    def napper_stays_napping(positions: tuple[str | None, str | None]) -> bool:
        return positions[0] is not None or positions[1] is None

    destination_permutations = get_non_repeating_product(remaining_destinations, num_agents)  # type: ignore
    # Deduplicate
    destination_permutations = list(set(destination_permutations))
    subsequent_results: list[int] = [0]
    for destinations in destination_permutations:
        zipped_start_end = tuple(zip(agent_current_valves, destinations))
        nappers_stay_napping = map(napper_stays_napping, zipped_start_end)  # type: ignore
        my_print(f'{destinations=}')
        if not all(nappers_stay_napping):
            # Don't wake them up!
            my_print('Dont wake em up')
            continue
        paths: list[list[str] | None] = list(map(path_map.get, zipped_start_end))  # type: ignore
        my_print(f'{paths=}')
        calculate_distance = lambda t: get_distance(t[0], t[1])
        distances = list(map(calculate_distance, zipped_start_end))
        my_print(f'{distances=}')
        if type(min(distances)) == float:
            continue
        minimum_distance = int(min(distances))
        time_to_activate = minimum_distance + 1
        if remaining_time - time_to_activate <= 0:
            continue
        actual_destinations: list[str | None] = []
        activated_valves: list[str] = []
        # Step forward to next time an agent activates a valve. The other agents
        # move as far as they can in that amount of elapsed time.
        for path in paths:
            if path is None:
                actual_destinations.append(None)
            elif len(path) == time_to_activate:
                # Track the valves which get activated by the end of the step
                actual_destinations.append(path[-1])
                activated_valves.append(path[-1])
            else:
                actual_destinations.append(path[time_to_activate])
        assert len(activated_valves) == len(set(activated_valves))
        assert all(map(remaining_rates.__contains__, activated_valves))
        pressure_released = (remaining_time - time_to_activate) * sum(map(remaining_rates.get, activated_valves))  # type: ignore
        new_remaining = {k: v for k, v in remaining_rates.items() if k not in activated_valves}
        if len(new_remaining) > 0 and remaining_time - time_to_activate > 1:
            pressure_released += get_pressure_released(
                tuple(d for d in actual_destinations if d is not None),
                new_remaining,
                remaining_time - time_to_activate,
            )
        subsequent_results.append(pressure_released)
    result = max(subsequent_results)
    my_print(f'{result=}')
    cached_results[key] = result
    return result


@time_execution
def part_1() -> None:
    utility = get_pressure_released(('AA',), optimizable_flows_by_label, 30)
    print(f'part_1={utility}')


@time_execution
def part_2() -> None:
    # Populate the cache for minimal recursive calls
    MAX_TIME = 26
    # for max_time in tqdm(range(1, MAX_TIME)):
    #     get_pressure_released(('AA', 'AA'), optimizable_flows_by_label, max_time)
    utility = get_pressure_released(('AA', 'AA'), optimizable_flows_by_label, MAX_TIME)
    print(f'part_2={utility}')


if __name__ == '__main__':
    part_1()
    part_2()