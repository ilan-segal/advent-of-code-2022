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
        1 if a robot of type r was made after t, 0 otherwise.

    -> N(r,t) where r ∈ R
        Number of robots of type r after t.

    -> Q(m,t) where m ∈ M
        Number of material m after t.

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
    return open('19-minerals/input.txt', 'r').read().strip().replace('\r\n', '\n')


M = ['Ore', 'Clay', 'Obsidian']
R = ['Ore', 'Clay', 'Obsidian', 'Geode']
MATERIAL_TO_INDEX = {r: R.index(r) for r in R}


@dataclass
class Blueprint:
    id: int
    cost: dict[str, dict[str, int]]


def get_blueprint(raw_input_line: str) -> Blueprint:
    raw_input_line = raw_input_line.replace('\n\n', ' ')