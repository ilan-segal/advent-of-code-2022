"""
https://adventofcode.com/2022/day/7
"""


from __future__ import annotations
from dataclasses import dataclass, field
from functools import reduce
from typing import Optional


DEPTH_CHAR = '  '


@dataclass
class File:

    name: str
    size: int

    def get_flattened_tree(self) -> list[File]:
        return [self]

    def __len__(self) -> int:
        return self.size

    def __repr__(self, depth=0) -> str:
        return DEPTH_CHAR * depth + f'- {self.size}\t{self.name}'

@dataclass
class Directory:
    
    name: str
    parent: Optional[Directory]
    children: list[File | Directory] = field(default_factory=list)

    def get_child_by_name(self, name: str) -> Optional[File | Directory]:
        for child in self.children:
            if child.name == name:
                return child
        return None

    def get_flattened_tree(self) -> list[File | Directory]:
        flattened_children = [child.get_flattened_tree() for child in self.children] 
        return list(reduce(list.__add__, flattened_children, [self]))

    def __len__(self) -> int:
        return sum(map(len, self.children))

    def __repr__(self, depth=0) -> str:
        ret = DEPTH_CHAR * depth + f'- dir {self.name}'
        for child in self.children:
            ret += '\n' + child.__repr__(depth+1)
        return ret


def get_raw_input() -> str:
    return open('07-no-space/input.txt', 'r').read().strip().replace('\r\n', '\n')


def parse_ls_items(command_output: str, parent_directory: Directory) -> list[File | Directory]:
    items = []
    for line in command_output.split('\n'):
        item_description, item_name = line.split(' ', maxsplit=1)
        if item_description == 'dir':
            items.append(Directory(item_name, parent_directory))
        else:
            file_size = int(item_description)
            items.append(File(item_name, file_size))
    return items


def get_cd_destination(command: str, cur_directory: Directory) -> Directory:
    if command == 'cd ..' and cur_directory.parent is not None:
        return cur_directory.parent
    destination_name = command.split(' ', maxsplit=1)[1]
    destination = cur_directory.get_child_by_name(destination_name)
    assert isinstance(destination, Directory), \
        f'Cannot cd into {destination_name}: Expected Directory got {type(destination)}'
    return destination


def build_directory_tree(raw_commands: str) -> Directory:
    assert raw_commands.startswith('$ cd /'), \
        'Commands must start with navigation to root directory'
    raw_commands = raw_commands.replace('$ cd /\n', '')
    commands_and_outputs = map(str.strip, raw_commands.split('$ '))
    root = Directory('/', None)
    cur_directory = root
    for command_and_output in commands_and_outputs:
        if command_and_output == '': continue
        if command_and_output.startswith('ls'):
            output = command_and_output.replace('ls\n', '')
            cur_dir_children = parse_ls_items(output, cur_directory)
            cur_directory.children.extend(cur_dir_children)
        elif command_and_output.startswith('cd'):
            cur_directory = get_cd_destination(command_and_output, cur_directory)
        else:
            assert False, f'Invalid {command_and_output=}'
    return root


def part_1(root: Directory):
    flattened_tree = root.get_flattened_tree()
    directories = filter(lambda item: isinstance(item, Directory), flattened_tree)
    item_sizes = map(len, directories)
    MAX_SIZE = 100_000
    filtered_sizes = filter(MAX_SIZE.__ge__, item_sizes)
    total_size = sum(filtered_sizes)
    print(f'part_1={total_size}')


def part_2(root: Directory):
    DISK_SPACE = 70_000_000
    REQUIRED_SPACE = 30_000_000
    used_space = len(root)
    free_space = DISK_SPACE - used_space
    space_to_free = REQUIRED_SPACE - free_space
    flattened_tree = root.get_flattened_tree()
    directories = filter(lambda item: isinstance(item, Directory), flattened_tree)
    directory_sizes = map(len, directories)
    deletion_candidate_sizes = filter(space_to_free.__le__, directory_sizes)
    min_deletion_size = min(deletion_candidate_sizes)
    print(f'part_1={min_deletion_size}')


if __name__ == '__main__':
    raw_commands = get_raw_input()
    root = build_directory_tree(raw_commands)
    part_1(root)
    part_2(root)