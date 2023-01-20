"""
https://adventofcode.com/2022/day/18
"""

import time
from typing import TypeVar, ParamSpec, Callable
import heapq

Coordinates = tuple[int, int, int]

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
    return open('18-boiling-boulders/input.txt', 'r').read().strip().replace('\r\n', '\n')


def parse_voxel_position(raw_input_line: str) -> Coordinates:
    return tuple(map(int, raw_input_line.split(',')))


def get_voxel_positions(raw_input: str) -> tuple[Coordinates]:
    return tuple(map(parse_voxel_position, raw_input.splitlines()))


def get_neighbors(pos: Coordinates) -> list[Coordinates]:
    x, y, z = pos
    return (
        [(x+i, y, z) for i in [-1, 1]] 
        + [(x, y+i, z) for i in [-1, 1]] 
        + [(x, y, z+i) for i in [-1, 1]] 
    )


def get_surface_area(voxels: tuple[Coordinates]) -> int:
    surface_area_per_voxel: dict[Coordinates, int] = {voxel: 6 for voxel in voxels}
    for voxel in voxels:
        for neighbor in get_neighbors(voxel):
            if neighbor not in surface_area_per_voxel:
                continue
            surface_area_per_voxel[neighbor] -= 1
    assert not any(map(lambda x: x < 0, surface_area_per_voxel.values()))
    return sum(surface_area_per_voxel.values())
    
    
@time_execution
def part_1(raw_input: str) -> None:
    voxels = get_voxel_positions(raw_input)
    surface_area = get_surface_area(voxels)
    print(f'part_1={surface_area}')


def manhattan_distance(a: Coordinates, b: Coordinates) -> int:
    return sum(map(lambda p: abs(p[0] - p[1]), zip(a, b)))


proven_outside_points: set[Coordinates] = set()
proven_inside_points: set[Coordinates] = set()
def resolve_path(start: Coordinates, end: Coordinates, obstacles: set[Coordinates]) -> list[Coordinates]:
    paths: list[tuple[int, list[Coordinates]]] = []
    seen_coords: set[Coordinates] = set()
    heapq.heappush(paths, (manhattan_distance(start, end), [start]))
    while len(paths) > 0:
        cur_path_length, cur_path = heapq.heappop(paths)
        if cur_path[-1] in proven_outside_points:
            return cur_path + [end]
        if cur_path[-1] in proven_inside_points:
            return []
        if cur_path[-1] == end:
            proven_outside_points.add(start)
            return cur_path
        for next_label in get_neighbors(cur_path[-1]):
            if next_label in cur_path or next_label in obstacles or next_label in seen_coords:
                continue
            heapq.heappush(paths, (
            # A* Manhattan distance heuristic
                cur_path_length + manhattan_distance(next_label, end),
                cur_path + [next_label],
            ))
            seen_coords.add(next_label)
    proven_inside_points.add(start)
    return []


def get_surface_area_no_interior(voxels: tuple[Coordinates]) -> int:
    exterior_neighbors_by_voxel: dict[Coordinates, list[Coordinates]] = {
        v: get_neighbors(v) for v in voxels
    }
    for voxel in voxels:
        for neighbor in get_neighbors(voxel):
            if neighbor not in exterior_neighbors_by_voxel:
                continue
            exterior_neighbors_by_voxel[neighbor].remove(voxel)
    leftmost_exterior_voxel: Coordinates | None = None
    for voxel in voxels:
        if exterior_neighbors_by_voxel[voxel] == []:
            continue
        if (leftmost_exterior_voxel is None 
            or voxel[0] < leftmost_exterior_voxel[0]):
            leftmost_exterior_voxel = voxel
    assert leftmost_exterior_voxel is not None
    x, y, z = leftmost_exterior_voxel
    outside_point = (x - 1, y, z)
    obstacles = set(voxels)
    surface_area = 0
    for voxel, exterior_neighbors in exterior_neighbors_by_voxel.items():
        for neighbor in exterior_neighbors:
            if resolve_path(neighbor, outside_point, obstacles) != []:
                surface_area += 1
    return surface_area

    
@time_execution
def part_2(raw_input: str) -> None:
    voxels = get_voxel_positions(raw_input)
    surface_area = get_surface_area_no_interior(voxels)
    print(f'part_2={surface_area}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)
    part_2(raw_input)