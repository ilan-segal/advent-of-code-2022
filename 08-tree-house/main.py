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
    vertical_visibility = np.apply_along_axis(get_visibility_of_row, 1, grid.T).T
    return np.logical_or(horizontal_visibility, vertical_visibility)


def part_1(tree_grid: TreeGrid) -> None:
    visibility_grid = get_visibility_of_grid(tree_grid)
    n_visibile = np.sum(visibility_grid)
    print(f'part_1={n_visibile}')


def get_left_to_right_view_distance_of_row(tree_grid: TreeGrid) -> TreeGrid:
    view_distances = np.zeros_like(tree_grid)
    pending_view_dist_indexes: set[int] = set()
    num_trees = view_distances.shape[0]
    for i in range(num_trees):
        still_pending: set[int] = set()
        for pending_index in pending_view_dist_indexes:
            if tree_grid[i] >= tree_grid[pending_index]:
                view_distances[pending_index] = i - pending_index
            else:
                still_pending.add(pending_index)
        pending_view_dist_indexes = still_pending
        pending_view_dist_indexes.add(i)
    # Reached end of grid
    for i in pending_view_dist_indexes:
        view_distances[i] = num_trees - i - 1
    return view_distances


def get_left_to_right_view_distance_of_grid(tree_grid: TreeGrid) -> TreeGrid:
    return np.apply_along_axis(get_left_to_right_view_distance_of_row, 1, tree_grid)


def get_scenic_score_of_grid(tree_grid: TreeGrid) -> TreeGrid:
    left_to_right = get_left_to_right_view_distance_of_grid(tree_grid)
    right_to_left = np.flip(get_left_to_right_view_distance_of_grid(np.flip(tree_grid, axis=1)), axis=1)
    up_to_down = get_left_to_right_view_distance_of_grid(tree_grid.T).T
    down_to_up = np.flip(get_left_to_right_view_distance_of_grid(np.flip(tree_grid.T, axis=1)), axis=1).T
    return left_to_right * right_to_left * up_to_down * down_to_up


def part_2(tree_grid: TreeGrid) -> None:
    scenic_score = get_scenic_score_of_grid(tree_grid)
    best_scenic_score = np.max(scenic_score)
    print(f'part_2={best_scenic_score}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    tree_grid = get_tree_grid(raw_input)
    # tree_grid = tree_grid[:10, :10]
    part_1(tree_grid)
    part_2(tree_grid)