"""
https://adventofcode.com/2022/day/7
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class File:
    name: str
    size: int

    def __len__(self) -> int:
        return self.size


@dataclass
class Directory:
    name: str
    parent: Optional[Directory]
    children: list[File | Directory] = []

    def get_child_by_name(self, name: str) -> File | Directory | None:
        for child in self.children:
            if child.name == name:
                return child
        return None

    def __len__(self) -> int:
        return sum(map(len, self.children))


def get_raw_input() -> str:
    return open('07-no-space/input.txt', 'r').read().strip().replace('\r\n', '\n')


def parse_ls(command_output: str, parent_directory: Directory) -> list[File | Directory]:
    items = []
    for line in command_output.split('\n'):
        item_description, item_name = line.split(' ')
        if item_description == 'dir':
            items.append(Directory(item_name, parent_directory))
        else:
            file_size = int(item_description)
            items.append(File(item_name, file_size))
    return items


def build_directory_tree(raw_commands: str) -> Directory:
    assert raw_commands.startswith('$ cd /'), \
        'Commands must start with navigation to root directory'
    raw_commands = raw_commands.replace('$ cd /\n', '')
    commands_and_outputs = raw_commands.split('$ ')
    cur_directory = Directory('/', None)
    for command_and_output in commands_and_outputs:
        if command_and_output.startswith('$ ls'):
            output = command_and_output.replace('$ ls\n', '')
            