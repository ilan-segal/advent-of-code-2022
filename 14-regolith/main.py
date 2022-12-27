"""
https://adventofcode.com/2022/day/14
"""


Coordinates = tuple[int, int]


def get_raw_input() -> str:
    return open('14-regolith/input.txt', 'r').read().strip().replace('\r\n', '\n')


class CaveSystem:

    __blocked_spaces: set[Coordinates]
    __sanded_spaces: set[Coordinates]
    __lowest_point: int
    __infinite_floor: bool

    def __init__(self, wall_points_formatted: str, infinite_floor: bool) -> None:
        self.__blocked_spaces = set()
        for line in wall_points_formatted.split('\n'):
            split_line = line.split(' -> ')
            coords = map(lambda pair_str: tuple(map(int, pair_str.split(','))), split_line)
            self.__add_wall(list(coords))
        self.__lowest_point = max(map(lambda coord: coord[1], self.__blocked_spaces))
        if infinite_floor:
            self.__lowest_point += 2
        self.__sanded_spaces = set()
        self.__infinite_floor = infinite_floor

    def __add_wall(self, wall_points: list[Coordinates]) -> None:
        for i in range(len(wall_points) - 1):
            self.__add_wall_segment(wall_points[i], wall_points[i + 1])

    def __add_wall_segment(self, start_point: Coordinates, end_point: Coordinates) -> None:
        x_direction = end_point[0] - start_point[0]
        y_direction = end_point[1] - start_point[1]
        if x_direction != 0:
            x_direction //= abs(x_direction)
        if y_direction != 0:
            y_direction //= abs(y_direction)
        cur_x, cur_y = start_point
        while True:
            self.__blocked_spaces.add((cur_x, cur_y))
            if (cur_x, cur_y) == end_point:
                break
            cur_x += x_direction
            cur_y += y_direction

    def __space_is_available(self, space_coords: Coordinates) -> bool:
        return (
            space_coords not in self.__blocked_spaces and
            space_coords not in self.__sanded_spaces and
            (not self.__infinite_floor or space_coords[1] != self.__lowest_point)
        )

    def drop_sand_from(self, position: Coordinates) -> bool:
        """
        Return True iff sand comes to rest inside system.
        """
        if not self.__space_is_available(position):
            return False
        cur_x, cur_y = position
        while True:
            if cur_y > self.__lowest_point:
                return False
            elif self.__space_is_available((cur_x, cur_y + 1)):
                cur_y += 1
            elif self.__space_is_available((cur_x - 1, cur_y + 1)):
                cur_x -= 1
                cur_y += 1
            elif self.__space_is_available((cur_x + 1, cur_y + 1)):
                cur_x += 1
                cur_y += 1
            else:
                self.__sanded_spaces.add((cur_x, cur_y))
                return True

    def fill_to_capacity_from(self, position: Coordinates) -> None:
        while self.drop_sand_from(position):
            pass

    def get_sand_count(self) -> int:
        return len(self.__sanded_spaces)

    def __str__(self) -> str:
        significant_spaces = self.__sanded_spaces.union(self.__blocked_spaces)
        min_x = min(coords[0] for coords in significant_spaces)
        max_x = max(coords[0] for coords in significant_spaces)
        min_y = min(coords[1] for coords in significant_spaces)
        max_y = self.__lowest_point
        string = ''
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (x, y) in self.__sanded_spaces:
                    string += 'o'
                elif not self.__space_is_available((x, y)):
                    string += '#'
                else:
                    string += '.'
            string += '\n'
        return string.strip()


def part_1(raw_input: str) -> None:
    cave_system = CaveSystem(raw_input, infinite_floor=False)
    # print(str(cave_system))
    cave_system.fill_to_capacity_from((500, 0))
    # print(str(cave_system))
    capacity = cave_system.get_sand_count()
    print(f'part_1={capacity}')


def part_2(raw_input: str) -> None:
    cave_system = CaveSystem(raw_input, infinite_floor=True)
    # print(str(cave_system))
    cave_system.fill_to_capacity_from((500, 0))
    # print(str(cave_system))
    capacity = cave_system.get_sand_count()
    print(f'part_1={capacity}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)
    part_2(raw_input)