"""
https://adventofcode.com/2022/day/1
"""


def get_calories_per_elf(calories_raw: str) -> list[int]:
    # In case of carriage return
    calories_raw = calories_raw.replace('\r', '')
    calories_split = calories_raw.split('\n\n')
    aggregate = lambda calories: sum(map(int, filter(len, calories.split('\n'))))
    return list(map(aggregate, calories_split))


def get_multi_line_input() -> str:
    END_TOKEN = 'END'
    print(f'When done, enter "{END_TOKEN}"')
    result = ''
    while True:
        cur_input = input()
        if cur_input == 'END':
            return result
        result += '\n' + cur_input


def get_input_from_file(filename: str) -> str:
    return open(filename, 'r').read()


def main():
    # calories_raw = get_multi_line_input()
    calories_raw = get_input_from_file('1-calorie-counting/input.txt')
    calories_per_elf = get_calories_per_elf(calories_raw)
    sorted_calories = sorted(calories_per_elf)
    highest_calories = sorted_calories[-1]
    three_highest_calories = sum(sorted_calories[-3:])
    print(highest_calories)
    print(three_highest_calories)


if __name__ == '__main__':
    main()