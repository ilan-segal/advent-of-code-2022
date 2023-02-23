from __future__ import annotations
from typing import Literal

import enum


Vector2D = tuple[int, int]


def add_vectors(a: Vector2D, b: Vector2D) -> Vector2D:
    (ax, ay), (bx, by) = a, b
    return (ax + bx), (ay + by)


def multiply_vectors(v: Vector2D, m: int) -> Vector2D:
    x, y = v
    return (m*x, m*y)


class Heading(enum.Enum):

    Right = 0
    Down = 1
    Left = 2
    Up = 3

    def get_vector(self) -> Vector2D:
        return {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1),
        }[self.value]

    def rotated(self, direction: Literal['R', 'L']) -> Heading:
        turn_direction = 1 if direction == 'R' else -1
        return Heading((self.value + turn_direction) % 4)


FaceId = int

class Face:

    _neighbors: dict[Heading, Face]
    _id: FaceId

    _EXISTING_IDS: set[FaceId] = set()

    @classmethod
    def generate_id(cls) -> FaceId:
        result = max(cls._EXISTING_IDS, default=-1) + 1
        cls._EXISTING_IDS.add(result)
        return result

    def __init__(self) -> None:
        self._neighbors = dict()
        self._id = Face.generate_id()

    def add_neighbor(self, edge: Heading, neighbor: Face, neighbor_edge: Heading) -> None:
        assert edge not in self._neighbors, 'Self edge already occupied'
        assert neighbor_edge not in neighbor._neighbors, 'Nieghbor edge already occupied'
        self._neighbors[edge] = neighbor
        neighbor._neighbors[neighbor_edge] = self

    def _get_neighbors_adjacent_to_edge(self, edge: Heading) -> tuple[Face | None, Face | None]:
        """
        CW from edge then CCW from edge
        """
        return (
            self._neighbors.get(edge.rotated('L')),
            self._neighbors.get(edge.rotated('R')),
        )


class Cube:

    _faces: dict[FaceId, Face]

    def __init__(self, flattened_cube: str) -> None:
        self._faces = dict()
        self.__process_flattened_cube(flattened_cube)

    def __process_flattened_cube(self, flattened_cube: str) -> None:
        pass