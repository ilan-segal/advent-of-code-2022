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
    30 >= t_i >= 1
    ...
    t_i - t_j + 60 * B_ij >= 1 + d_ij           # https://lpsolve.sourceforge.net/5.1/absolute.htm
    t_j - t_i + 60 * (1 - B_ij) >= 1 + d_ij     # The original constraint:
    1 >= B_ij >= 0                              # | t_i - t_j | >= 1 + d_ij
    ...

How many constraints and variables are we optimizing?

    Let there be N optimizable valves (valves with flow greater than 0).
    Each such valve v_i entails the following constraints:
        1 <= t_i <= 30
    This adds N constraints and N decision variables.
    There are also the following constraints for each i, j:
        1 + d_ij <= t_i - t_j + 60 * B_ij
        -59 + d_ij <= t_j - t_i - 60 * B_ij
        0 <= B_ij <= 1
    We do not include the above constraints for j, i or i, i because that would be redundant.
    Thus this adds N*(N-1)*3/2 constraints and N*(N-1)/2 decision variables.

    Thus we have:
        N + N*(N-1)/2 decision variables (O(N^2))
        N + N*(N-1)*3/2 constraints (O(N^2))
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


from scipy.optimize import (
    Bounds,
    LinearConstraint,
    milp,
    OptimizeResult,
)
import numpy as np


def optimize_valve_times(distance_map: DistanceMap, valve_rate_by_label: dict[str, int]) -> OptimizeResult:
    N = len(valve_rate_by_label)
    num_B = (N * (N - 1)) // 2
    num_variables = N + num_B
    decision_bounds = Bounds(
        np.array([1] * N + [0] * num_B),  # type: ignore
        np.array([30] * N + [1] * num_B),  # type: ignore
    )
    constraints: list[LinearConstraint] = []
    seen_pairs: set[tuple[str, str]] = set()
    labels = list(valve_rate_by_label.keys())
    for ai, a in enumerate(labels):
        for bi, b in enumerate(labels):
            if a == b:
                continue
            if (a, b) in seen_pairs or (b, a) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            distance = distance_map[a, b]
            # Constraint: 1 + d_ij <= t_i - t_j + 60 * B_ij
            A = np.zeros((1, num_variables))
            A[0][ai] = 1
            A[0][bi] = -1
            A[0][N + len(seen_pairs) - 1] = 60
            constraints.append(LinearConstraint(A, lb=1+distance))
            # Constraint: -59 + d_ij <= t_j - t_i - 60 * B_ij
            A = np.zeros((1, num_variables))
            A[0][bi] = 1
            A[0][ai] = -1
            A[0][N + len(seen_pairs) - 1] = -60
            constraints.append(LinearConstraint(A, lb=-59+distance))
    return milp(
        c=[valve_rate_by_label[label] for label in labels],
        integrality=1,
        bounds=decision_bounds,
        constraints=constraints,
        options={'disp': True},
    )


@time_execution
def part_1(raw_input: str) -> None:
    distances, optimizable_flows_by_label = create_volcano_state(raw_input)
    print(distances)
    print(optimizable_flows_by_label)
    result = optimize_valve_times(distances, optimizable_flows_by_label)
    print(result)


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)