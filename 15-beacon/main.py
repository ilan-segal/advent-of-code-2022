"""
https://adventofcode.com/2022/day/1
"""


import re


Coords = tuple[int, int]


def get_raw_input() -> str:
    return open('15-beacon/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_sensor_beacon_map(raw_input: str) -> dict[Coords, Coords]:
    sensor_beacon_map = dict()
    pattern = r'Sensor at x=(-?\d+), y=(-?\d+): closest beacon is at x=(-?\d+), y=(-?\d+)'
    for line in raw_input.split('\n'):
        result = re.search(pattern, line)
        assert result is not None, f'Invalid string: {line}'
        sensor_x, sensor_y, beacon_x, beacon_y = map(int, [result.group(i) for i in range(1, 5)])
        sensor_beacon_map[(sensor_x, sensor_y)] = (beacon_x, beacon_y)
    return sensor_beacon_map


def get_beacon_exclusion_range_at_y(y: int, sensor: Coords, beacon: Coords) -> None | tuple[int, int]:
    # Manhattan distance
    sensor_beacon_distance = sum(map(abs, [s - b for s, b in zip(sensor, beacon)]))  # type: ignore
    y_diff = abs(sensor[1] - y)
    remaining_distance = sensor_beacon_distance - y_diff
    if remaining_distance <= 0:
        return None
    range_min = sensor[0] - remaining_distance
    range_max = sensor[0] + remaining_distance
    return range_min, range_max


def get_exclusion_size_at_y(y: int, sensor_beacon_map: dict[Coords, Coords]) -> int:
    ranges = [get_beacon_exclusion_range_at_y(y, sensor, beacon) for sensor, beacon in sensor_beacon_map.items()]
    ranges = [r for r in ranges if r is not None]
    range_min = min(x for r in ranges for x in r)
    range_max = max(x for r in ranges for x in r)
    mask = [False for _ in range(range_max - range_min + 1)]
    for a, b in ranges:
        for x in range(a + range_min, b + range_min + 1):
            if (x, y) in sensor_beacon_map.keys() or (x, y) in sensor_beacon_map.values():
                continue
            mask[x] = True
    return sum(mask)


def part_1(raw_input: str) -> None:
    sensor_beacon_map = get_sensor_beacon_map(raw_input)
    row = 2_000_000
    # row = 10
    exclusion_size = get_exclusion_size_at_y(row, sensor_beacon_map)
    print(f'part_1={exclusion_size}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    part_1(raw_input)