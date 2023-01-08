"""
https://adventofcode.com/2022/day/16

For some valve v_i, let r_i be that valve's flow rate.
Let p_i(t) be v_i's total pressure released if opened at time t.
Then: p_i(t) = r_i * (30 - t)

Let t_i be the time at which valve v_i is opened. Then we want to maximize:
    
    Σ p_i(t_i) = Σ r_i * (30 - t)

Subject to the following constraints:

    - 1 <= t_i <= 30
    - | t_i - t_j | >= 1 + d_ij where d_ij is the distance between v_i and v_j

I believe this can be solved with Integer Programming. A naive objective function
would be as follows:

z = (r_0 * (30 - t_0)) + (r_1 * (30 - t_1)) + ...
  = (r_0 * 30) - (r_0 * t_0) + (r_1 * 30) - (r_1 * t_0) + ...

Since r_i is a constant and we simply want to find optimal t_i, we can simplify
our objective function to:

z = - (r_0 * t_0 + r_1 * t_1 + ...)

Thus our integer programming problem can be formulated as:

Optimize:
    z = - (r_0 * t_0 + r_1 * t_1 + ...)
Subject to:
    t_i >= 1
    -t_i >= -30
    ...
    t_i - t_j + 60 * B_ij >= 1 + d_ij           # https://lpsolve.sourceforge.net/5.1/absolute.htm
    t_j - t_i + 60 * (1 - B_ij) >= 1 + d_ij     # The original constraint:
    B_ij >= 0                                   # | t_i - t_j | >= 1 + d_ij
    -B_ij >= -1
    ...

If there are N optimizable valves then there are 2N + 4N^2 - 4N = 4N^2 - 2N constraints.
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


@time_execution
def part_1(raw_input: str) -> None:
    distances, optimizable_flows_by_label = create_volcano_state(raw_input)
    print(distances)
    print(optimizable_flows_by_label)
        

if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)