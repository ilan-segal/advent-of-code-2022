"""
https://adventofcode.com/2022/day/19

They're minerals, Marie.

Trying to use ILP for this one. (https://en.wikipedia.org/wiki/Integer_programming)

Let's define some terms:

    M = {Ore, Clay, Obsidian}  # The materials we care to track
    R = {Ore, Clay, Obsidian, Geode}  # The possible types of robots
    
Notice M ⊂ R. This will be useful later. We define the cost of building a robot
using any particular material as:

    C(r,m) where r ∈ R, m ∈ M

Then for minutes t ∈ {1, 2, ..., 23} we want to track the following variables:

    -> A(r,t) where r ∈ R
        1 if a robot of type r was made during minute t, 0 otherwise.

    -> N(r,t) where r ∈ R
        Number of robots of type r before minute t elapses.

    -> Q(m,t) where m ∈ M
        Number of material m before minute t elapses.

We have the following constraints for each minute t:

    For each r ∈ R:     0 <= A(r,t) <= 1                        (4 constraints)
                        0 <= ∑_r A(r,t) <= 1                    (1 constraint)
    For each r ∈ R:     0 <= N(r,t) - ∑_(t'<t) A(r,t') <= 0     (4 constraints)

We also have to constrain Q(m,t) as follows:

    Q(m,t) = [∑_(t'<t) N(m,t')*(t-t')] - [∑_(t'<t) ∑_r A(r,t')*C(r,m)]
    0 <= Q(m,t)  # We cannot go negative, i.e. we cannot spend materials we don't have

We can skip the definition of Q(m,t) entirely and cut straight to this constraint
on A(r,t) and N(r,t):

    For each m ∈ M:     0 <= [∑_(t'<t) N(m,t')*(t-t')] - [∑_(t'<t) ∑_r A(r,t')*C(r,m)]  (3 constraints)

Thus we count the number of variables and constraints:

    Num. Variables = 8*T
    Num. Constraints = 12*T

...where T is the number of optimizable minutes (1,2,...,23). We don't count the
24th minute because any robots made during this minute don't contribute to the
objective function. Speaking of, our objective function is this:

    maximize: ∑_t N(Geode,t)*(24-t)
"""

from dataclasses import dataclass
import re
import time
from typing import TypeVar, ParamSpec, Callable, Literal

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
    return open('19-minerals/input.txt', 'r').read().strip().replace('\r\n', '\n')



MaterialType = Literal['Ore', 'Clay', 'Obsidian', 'Geode']
CostDict = dict[MaterialType, dict[MaterialType, int]]


M: list[MaterialType] = ['Ore', 'Clay', 'Obsidian']
R: list[MaterialType] = ['Ore', 'Clay', 'Obsidian', 'Geode']


@dataclass
class Blueprint:
    id: int
    cost: CostDict


def get_blueprint(raw_input_line: str) -> Blueprint:
    raw_input_line = ' '.join(raw_input_line.split())
    pattern = r'Blueprint (\d+): Each ore robot costs (\d+) ore. Each clay robot costs (\d+) ore. Each obsidian robot costs (\d+) ore and (\d+) clay. Each geode robot costs (\d+) ore and (\d+) obsidian.'
    result = re.search(pattern, raw_input_line)
    assert result is not None, f'Invalid string {raw_input_line} for pattern {pattern}'
    get_group = lambda group_num: int(result.group(group_num))
    blueprint_id = get_group(1)
    cost: CostDict = {
        'Ore': {'Ore': get_group(2)},
        'Clay': {'Ore': get_group(3)},
        'Obsidian': {
            'Ore': get_group(4),
            'Clay': get_group(5),
        },
        'Geode': {
            'Ore': get_group(6),
            'Obsidian': get_group(7),
        },
    }
    return Blueprint(blueprint_id, cost)


def get_blueprints(raw_input: str) -> list[Blueprint]:
    inputs = raw_input.split('Blueprint')
    inputs = filter(len, inputs)
    inputs = map(lambda s: 'Blueprint' + s, inputs)
    return list(map(get_blueprint, inputs))


from pulp import LpMaximize, LpProblem, lpSum, LpVariable, LpAffineExpression


class LpMatrix:
    """
    Represents a time series of values across different material types.
    """

    _rows: int
    _cols: int
    _terms: list[list[LpVariable | LpAffineExpression | int]]

    def __init__(self, num_materials: int, num_t: int) -> None:
        self._rows = num_materials
        self._cols = num_t
        self._terms = [[None for _ in range(num_t)] for _ in range(num_materials)]  # type: ignore

    def __getitem__(self, index: tuple[MaterialType, int]) -> LpVariable | LpAffineExpression | int:
        # print(f'{index=}')
        material, t = index
        r = R.index(material)
        return self._terms[r][t-1]

    def __setitem__(self,  index: tuple[MaterialType, int], value: LpVariable | LpAffineExpression | int) -> None:
        material, t = index
        r = R.index(material)
        self._terms[r][t-1] = value


def get_optimal_geodes(blueprint: Blueprint) -> int:
    problem = LpProblem(name=f'blueprint-{blueprint.id}', sense=LpMaximize)
    T_RANGE = range(1, 24)
    # Variables
    C = blueprint.cost
    A = LpMatrix(len(R), len(T_RANGE))
    for r in R:
        for t in T_RANGE:
            # print(f'Assigning {r=} {t=}')
            A[r, t] = LpVariable(name=f'A_{r}_t_{t:02}', cat='Binary')
    N = LpMatrix(len(R), len(T_RANGE))
    for r in R:
        for t in T_RANGE:
            if t == 1:
                if r == 'Ore':
                    N[r, t] = 1
                else:
                    N[r, t] = 0
            else:
                N[r, t] = lpSum(A[r, t_prime] for t_prime in range(1,t))
            # if r == 'Ore' and t == 1:
            #     N[r, t] += 1
    # Constraints
    for t in T_RANGE:
        problem += (sum(A[r, t] for r in R) <= 1, f'at_most_one_robot_at_t_{t:02}')
        for m in M:
            Q_m_t = (
                lpSum(N[m, t_prime] * (t - t_prime) for t_prime in range(1,t))
                -
                lpSum(A[r, t_prime] * C.get(r, dict()).get(m, 0) for t_prime in range(1,t) for r in R)
            )
            problem += (
                0 <= Q_m_t, f'constrain_{m}_at_t_{t:02}'
            )
    problem.setObjective(lpSum(N['Geode', t] * (len(T_RANGE) - t + 1) for t in T_RANGE))
    print(repr(problem))
    # TODO
    result: int = problem.solve()
    for v in problem.variables():
        print(f'{v}={v.value()}')
    return problem.objective.value()  # type: ignore

if __name__ == '__main__':
    raw_input = get_raw_input()
    for blueprint in get_blueprints(raw_input)[:1]:
        print(blueprint)
        print(get_optimal_geodes(blueprint))