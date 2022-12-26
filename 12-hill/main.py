"""
https://adventofcode.com/2022/day/12
"""


from __future__ import annotations
from dataclasses import dataclass, field
import heapq
from typing import Callable


def get_raw_input() -> str:
    return open('12-hill/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_height_grid(raw_input: str) -> list[list[int]]:
    raw_input = raw_input.replace('S', 'a').replace('E', 'z')
    return [[ord(c) - ord('a') for c in row] for row in raw_input.split('\n')]


Coordinates = tuple[int, int]


def get_start_and_end(raw_input: str) -> tuple[Coordinates, Coordinates]:
    S, E = None, None
    for row_i, row in enumerate(raw_input.split('\n')):
        for col_i, element in enumerate(row):
            if element == 'S':
                S = (row_i, col_i)
            elif element == 'E':
                E = (row_i, col_i)
    assert S is not None, 'No start found'
    assert E is not None, 'No end found'
    return S, E


def get_adjacency_map(height_grid: list[list[int]]) -> dict[Coordinates, list[Coordinates]]:

    num_rows = len(height_grid)
    num_cols = len(height_grid[0])

    def is_out_of_bounds(row: int, col: int) -> bool:
        return (
            row < 0
            or row >= num_rows
            or col < 0
            or col >= num_cols
        )

    def is_accessible(start: Coordinates, end: Coordinates) -> bool:
        if is_out_of_bounds(*start) or is_out_of_bounds(*end):
            return False
        s_r, s_c = start
        e_r, e_c = end
        start_height = height_grid[s_r][s_c]
        end_height = height_grid[e_r][e_c]
        return end_height <= start_height + 1

    adjacency_map: dict[Coordinates, list[Coordinates]] = {}
    for row in range(num_rows):
        for col in range(num_cols):
            start = row, col
            adjacency_map[start] = list()
            for row_direction, col_direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                end = (row + row_direction), (col + col_direction)
                if not is_accessible(start, end):
                    continue
                adjacency_map[start].append(end)
    return adjacency_map


@dataclass
class Node:

    value: int
    position: Coordinates
    reachable_neighbors: list[Node] = field(default_factory=list)
    parent: Node | None = None
    weight: int | float = float('inf')

    def __lt__(self, other: Node) -> bool:
        return self.weight < other.weight

    def __eq__(self, other: Node) -> bool:
        return self.position == other.position

    def __repr__(self) -> str:
        if self.parent is None:
            return f'Node(weight={self.weight}, pos={self.position})'
        else:
            return f'Node(weight={self.weight}, pos={self.position}, parent={self.parent.position})'

    def __str__(self) -> str:
        if self.parent is None:
            return '•'
        direction_to_parent = (
            self.parent.position[0] - self.position[0],
            self.parent.position[1] - self.position[1],
        )
        arrow_set = '⭠ ⭢ ⭡ ⭣'.split(' ')
        if direction_to_parent[0] < 0:
            return arrow_set[2]
        elif direction_to_parent[0] > 0:
            return arrow_set[3]
        elif direction_to_parent[1] < 0:
            return arrow_set[0]
        else:
            return arrow_set[1]

    def __hash__(self) -> int:
        return hash(id(self))

    def get_path(self) -> list[Node]:
        if self.parent is None:
            return [self]
        return self.parent.get_path() + [self]

    def reset(self) -> None:
        self.parent = None
        self.weight = float('inf')

    def set_parent_if_better(self, parent: Node) -> None:
        LINK_WEIGHT = 1
        if parent.weight + LINK_WEIGHT >= self.weight:
            return
        self.parent = parent
        self.weight = parent.weight + LINK_WEIGHT


class Field:

    __nodes: list[list[Node]]

    def __init__(self, height_grid: list[list[int]]) -> None:
        adjacency_map = get_adjacency_map(height_grid)
        num_rows, num_cols = len(height_grid), len(height_grid[0])
        self.__nodes = [[Node(height_grid[row][col], (row, col)) for col in range(num_cols)] for row in range(num_rows)]
        for (x, y), neighbor_positions in adjacency_map.items():
            neighbor_nodes = list(map(self.__getitem__, neighbor_positions))
            self[x, y].reachable_neighbors = neighbor_nodes

    def __str__(self) -> str:
        return '\n'.join(''.join(map(str, row)) for row in self.__nodes)

    def __getitem__(self, pos: Coordinates) -> Node:
        row, col = pos
        return self.__nodes[row][col]

    def find_path(self,
        start: Coordinates, 
        end: Coordinates | Callable[[Node], bool],
    ) -> list[Node] | None:

        goal_predicate: Callable[[Node], bool] = end if callable(end) else (lambda node: node.position == end)

        for row in self.__nodes:
            for node in row:
                node.reset()
        start_node = self[start]
        start_node.weight = 0
        closed_nodes: set[Node] = {start_node}
        open_nodes: list[Node] = start_node.reachable_neighbors
        for node in open_nodes:
            node.set_parent_if_better(start_node)
        heapq.heapify(open_nodes)
        while len(open_nodes) > 0:
            cur_node = heapq.heappop(open_nodes)
            for neighbor in cur_node.reachable_neighbors:
                if neighbor in closed_nodes:
                    continue
                neighbor.set_parent_if_better(cur_node)
                if neighbor not in open_nodes:
                    heapq.heappush(open_nodes, neighbor)
            if goal_predicate(cur_node):
                return cur_node.get_path()
            closed_nodes.add(cur_node)
        return None


def part_1(raw_input: str) -> None:

    height_grid = get_height_grid(raw_input)
    field = Field(height_grid)

    start_pos, end_pos = get_start_and_end(raw_input)
    path = field.find_path(start_pos, end_pos)
    # print(str(field))
    assert path is not None, f'No path found from {start_pos} to {end_pos}'
    num_steps = len(path) - 1
    print(f'part_1={num_steps}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)