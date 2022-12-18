"""
https://adventofcode.com/2022/day/8
"""


import numpy as np
from numpy import typing as npt


TreeGrid = npt.NDArray[np.int8]
VisibilityGrid = npt.NDArray[np.bool8]


def get_raw_input() -> str:
    return open('08-tree-house/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_tree_grid(raw_input: str) -> TreeGrid:
    rows: list[TreeGrid] = []
    for row in raw_input.split('\n'):
        rows.append(np.fromiter(row, dtype=int))
    return np.stack(rows, axis=0)


def get_left_to_right_visibility_of_row(grid_row: TreeGrid) -> VisibilityGrid:
    running_max_height = np.maximum.accumulate(np.append([-1], grid_row))[:-1]
    return np.greater(grid_row, running_max_height)


def get_visibility_of_row(grid_row: TreeGrid) -> VisibilityGrid:
    left_to_right_visibility = get_left_to_right_visibility_of_row(grid_row)
    right_to_left_visibility = get_left_to_right_visibility_of_row(grid_row[::-1])[::-1]
    return np.logical_or(left_to_right_visibility, right_to_left_visibility)


def get_visibility_of_grid(grid: TreeGrid) -> VisibilityGrid:
    horizontal_visibility = np.apply_along_axis(get_visibility_of_row, 1, grid)
    print(horizontal_visibility.astype(int))
    vertical_visibility = np.apply_along_axis(get_visibility_of_row, 1, grid.T).T
    print(vertical_visibility.astype(int))
    return np.logical_or(horizontal_visibility, vertical_visibility)


def part_1(tree_grid: TreeGrid) -> None:
    print(tree_grid)
    visibility_grid = get_visibility_of_grid(tree_grid)
    print(visibility_grid.astype(int))
    n_visibile = np.sum(visibility_grid)
    print(f'part_1={n_visibile}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    tree_grid = get_tree_grid(raw_input)
    part_1(tree_grid)