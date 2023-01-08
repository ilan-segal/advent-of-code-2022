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

If there are N optimizable valves then there are 2N * 4N^2 - 4N = 4N^2 - 2N constraints.
"""


from __future__ import annotations
from dataclasses import dataclass
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


@dataclass
class Valve:
    label: str
    flow_rate: int
    __is_open: bool = False

    def get_flow(self) -> int:
        return self.flow_rate * self.__is_open

    def is_optimized(self) -> bool:
        return self.flow_rate == 0 or self.__is_open

    def open(self) -> Valve:
        assert not self.__is_open, f'Valve {self.label} is already open.'
        return Valve(self.label, self.flow_rate, True)


@dataclass
class VolcanoState:
    valves_by_label: dict[str, Valve]
    tunnels: dict[str, list[str]]
    cur_valve_label: str = 'AA'
    released_pressure: int = 0
    minutes_passed: int = 0

    def __get_minute_flow(self) -> int:
        return sum(valve.get_flow() for valve in self.valves_by_label.values())

    def move_to(self, valve_label: str) -> VolcanoState:
        assert valve_label != self.cur_valve_label, f'Already at valve {valve_label}'
        assert valve_label in self.tunnels[self.cur_valve_label], f'Cannot access {valve_label} from {self.cur_valve_label}'
        new_released_pressure = self.released_pressure + self.__get_minute_flow()
        return VolcanoState(self.valves_by_label, self.tunnels, valve_label, new_released_pressure, self.minutes_passed + 1)

    def open_current_valve(self) -> VolcanoState:
        new_released_pressure = self.released_pressure + self.__get_minute_flow()
        new_dict = self.valves_by_label.copy()
        new_valve = new_dict[self.cur_valve_label].open()
        new_dict[self.cur_valve_label] = new_valve
        return VolcanoState(new_dict, self.tunnels, self.cur_valve_label, new_released_pressure, self.minutes_passed + 1)

    def is_optimized(self) -> bool:
        return all(valve.is_optimized() for valve in self.valves_by_label.values())


def create_volcano_state(raw_input: str) -> VolcanoState:
    valves_by_label: dict[str, Valve] = dict()
    tunnels: dict[str, list[str]] = dict()
    for valve_string in raw_input.split('\n'):
        pattern = r'Valve ([A-Z]+) has flow rate=(\d+); tunnels? leads? to valves? (.+)'
        result = re.search(pattern, valve_string)
        assert result is not None, f'Invalid string {valve_string}'
        valve_label = result.group(1)
        valve_flow_rate = int(result.group(2))
        valve_tunnels = result.group(3).split(', ')
        valve = Valve(valve_label, valve_flow_rate)
        valves_by_label[valve_label] = valve
        tunnels[valve_label] = valve_tunnels
    return VolcanoState(valves_by_label, tunnels)


@time_execution
def part_1(raw_input: str) -> None:
    init_state = create_volcano_state(raw_input)
    print(init_state)
        

if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)