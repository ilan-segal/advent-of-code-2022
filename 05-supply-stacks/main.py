"""
https://adventofcode.com/2022/day/5/input
"""


def get_raw_input() -> str:
    return open('05-supply-stacks/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_stacks(raw_input: str) -> list[list[str]]:
    start_of_moves_index = raw_input.find('move')
    stacks_raw_input = raw_input[:start_of_moves_index].strip()
    raw_input_lines = stacks_raw_input.split('\n')
    num_stacks = (len(raw_input_lines[-1]) + 2) // 4
    stacks = [list() for _ in range(num_stacks)]
    stack_input_lines = raw_input_lines[:-1]
    max_stack_height = len(stack_input_lines)
    for height in range(max_stack_height):
        line_index = -(height + 1)
        for stack_index in range(num_stacks):
            start = stack_index * 4
            end = start + 4
            crate = stack_input_lines[line_index][start:end]
            if crate.strip() == '':
                continue
            crate_symbol = crate[1]
            stacks[stack_index].append(crate_symbol)
    return stacks


def get_instructions(raw_input: str) -> list[tuple[int, int, int]]:
    start_of_moves_index = raw_input.find('move')
    instructions_raw_input = raw_input[start_of_moves_index:].strip()
    instructions = []
    for line in instructions_raw_input.split('\n'):
        line = (
            line.replace('move ', '')
            .replace('from ', '')
            .replace('to ', '')
        )
        instructions.append(tuple(map(int, line.split(' '))))
    return instructions


def get_top_crates(stacks: list[list[str]]) -> str:
    tops = []
    for stack in stacks:
        if len(stack) == 0: continue
        tops.append(stack[-1])
    return ''.join(tops)


def part_1_apply_instructions(stacks: list[list[str]], instructions: list[tuple[int, int, int]]):
    for qty, move_from, move_to in instructions:
        move_from -= 1
        move_to -= 1
        for _ in range(qty):
            stacks[move_to].append(stacks[move_from].pop())


def part_1():
    raw_input = get_raw_input()
    stacks = get_stacks(raw_input)
    instructions = get_instructions(raw_input)
    part_1_apply_instructions(stacks, instructions)
    part_1_tops = get_top_crates(stacks)
    print(f'{part_1_tops=}')


def part_2_apply_instructions(stacks: list[list[str]], instructions: list[tuple[int, int, int]]):
    for qty, move_from, move_to in instructions:
        move_from -= 1
        move_to -= 1
        crates_to_move = stacks[move_from][-qty:]
        stacks[move_from] = stacks[move_from][:-qty]
        stacks[move_to].extend(crates_to_move)


def part_2():
    raw_input = get_raw_input()
    stacks = get_stacks(raw_input)
    instructions = get_instructions(raw_input)
    part_2_apply_instructions(stacks, instructions)
    part_2_tops = get_top_crates(stacks)
    print(f'{part_2_tops=}')


if __name__ == '__main__':
    part_1()
    part_2()