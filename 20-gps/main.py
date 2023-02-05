"""
https://adventofcode.com/2022/day/20
"""

from __future__ import annotations
from typing import (
    Callable,
    Generic,
    Iterable,
    ParamSpec,
    TypeVar,
)

import dataclasses
import time


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
    return open('20-gps/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_values(raw_input: str) -> list[int]:
    return list(map(int, raw_input.split('\n')))


@dataclasses.dataclass
class Node(Generic[T]):
    value: T
    next: Node[T] | None = dataclasses.field(default=None, init=False, repr=False)
    prev: Node[T] | None = dataclasses.field(default=None, init=False, repr=False)

    def __eq__(self, __o: object) -> bool:
        return id(__o) == id(self)


class Ring(Generic[T], Iterable[T]):
    head: Node[T] | None = None
    count: int = 0

    def __init__(self, values: Iterable[T]) -> None:
        for value in values:
            self._append(value)
        
    def _insert(self, inserted: Node[T], prev_node: Node[T]) -> None:
        assert prev_node.prev is not None and prev_node.next is not None
        inserted.prev = prev_node
        inserted.next = prev_node.next
        prev_node.next.prev = inserted
        prev_node.next = inserted

    def _append(self, value: T) -> None:
        self.count += 1
        node = Node(value)
        if self.head is None:
            node.next = node
            node.prev = node
            self.head = node
            return
        assert self.head.prev is not None
        self._insert(node, self.head.prev)

    def _get_nodes(self) -> list[Node[T]]:
        result: list[Node[T]] = []
        cur = self.head
        while True:
            if cur is None:
                return result
            result.append(cur)
            cur = cur.next
            if cur == self.head:
                return result

    def _get_values(self) -> list[T]:
        return list(map(lambda node: node.value, self._get_nodes()))

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._get_values()})'

    def __len__(self) -> int:
        return len(self._get_nodes())

    def __iter__(self) -> Iterable[T]:
        return iter(self._get_values())

    def _get_from_offset(self, node: Node[T], offset: int) -> Node[T]:
        distance = abs(offset) #% self.count
        if distance == 0:
            return node
        for _ in range(distance):
            next_node = node.prev if offset < 0 else node.next
            assert next_node is not None
            node = next_node
        return node
        
              
class MixableRing(Ring[int]):

    def mix(self, n: int = 1) -> None:
        nodes = self._get_nodes()
        # print(self)
        for _ in range(n):
            for node in nodes:
                self.__move_node(node)
            # print(self)

    def __move_node(self, node: Node[int]) -> None:
        if node.value == 0:
            return
        if node == self.head:
            self.head = node.next
        prev_node = node.prev
        next_node = node.next
        assert prev_node is not None and next_node is not None
        # Close hole left by moved node
        prev_node.next = next_node
        next_node.prev = prev_node
        new_prev_node = self._get_from_offset(node, node.value)
        if node.value < 0:
            new_prev_node = new_prev_node.prev
        if new_prev_node == node.prev:
            return
        # Insert node at new position
        assert new_prev_node is not None
        self._insert(node, new_prev_node)

    def get_coordinates(self) -> tuple[int, int, int]:
        zero_node = self.head
        while True:
            assert zero_node is not None
            if zero_node.value == 0:
                break
            zero_node = zero_node.next
            assert zero_node != self.head, f'Exhaused nodes searching for 0 value'
        return tuple(self._get_from_offset(zero_node, offset).value for offset in [1000, 2000, 3000])


if __name__ == '__main__':
    raw_input = get_raw_input()
    values = get_values(raw_input)
    linked_list = MixableRing(values)
    orig_sorted = sorted(linked_list)
    linked_list.mix()
    new_sorted = sorted(linked_list)
    print(f'{orig_sorted==new_sorted}')
    print(f'part_1={sum(linked_list.get_coordinates())}')