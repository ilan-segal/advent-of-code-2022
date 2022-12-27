"""
https://adventofcode.com/2022/day/13
"""


from enum import Enum, auto
import json
from typing import Literal


SignalElement = int | list['SignalElement']


def get_raw_input() -> str:
    return open('13-distress-signal/input.txt', 'r').read().strip().replace('\r\n', '\n')


def get_signal_pairs(raw_input: str) -> list[tuple[list[SignalElement], list[SignalElement]]]:
    parse_pair = lambda pair_str: tuple(map(json.loads, pair_str.split('\n')))
    return list(map(parse_pair, raw_input.split('\n\n')))


class SignalOrder(Enum):
    OutOfOrder = auto()
    NoDecision = auto()
    InOrder = auto()


def determine_order_recursive(left_signal: SignalElement, right_signal: SignalElement) -> SignalOrder:
    # LIST VS LIST
    if isinstance(left_signal, list) and isinstance(right_signal, list):
        num_elements = max(len(left_signal), len(right_signal))
        for i in range(num_elements):
            if i >= len(left_signal):
                return SignalOrder.InOrder
            elif i >= len(right_signal):
                return SignalOrder.OutOfOrder
            order = determine_order_recursive(left_signal[i], right_signal[i])
            if order != SignalOrder.NoDecision:
                return order
        return SignalOrder.NoDecision
    # INTEGER VS INTEGER
    elif isinstance(left_signal, int) and isinstance(right_signal, int):
        if left_signal < right_signal:
            return SignalOrder.InOrder
        elif left_signal > right_signal:
            return SignalOrder.OutOfOrder
        return SignalOrder.NoDecision
    # INTEGER VS LIST
    elif isinstance(left_signal, int):
        return determine_order_recursive([left_signal], right_signal)
    # LIST VS INTEGER
    elif isinstance(right_signal, int):
        return determine_order_recursive(left_signal, [right_signal])
    else:
        raise TypeError(type(left_signal), type(right_signal))


def determine_signal_order(
    left_signal: list[SignalElement], 
    right_signal: list[SignalElement],
) -> Literal[SignalOrder.InOrder, SignalOrder.OutOfOrder]:
    order = determine_order_recursive(left_signal, right_signal)
    assert order != SignalOrder.NoDecision
    return order


def part_1(signal_pairs: list[tuple[list[SignalElement], list[SignalElement]]]) -> None:
    index_sum = 0
    for index, (left_signal, right_signal) in enumerate(signal_pairs):
        if determine_signal_order(left_signal, right_signal) == SignalOrder.InOrder:
            index_sum += index + 1
    print(f'part_1={index_sum}')


def get_sorted_signals(signals: list[list[SignalElement]]) -> list[list[SignalElement]]:
    """
    Python's `sorted` function uses a key parameter rather than an ordering
    parameter like JS, for example. We'll just implement mergesort to ensure
    signals are ordered properly.
    """
    n = len(signals)
    if n < 2:
        return signals
    a = get_sorted_signals(signals[:n//2])
    b = get_sorted_signals(signals[n//2:])
    merged = []
    while len(a) > 0 and len(b) > 0:
        if determine_signal_order(a[0], b[0]) == SignalOrder.InOrder:
            merged.append(a.pop(0))
        else:
            merged.append(b.pop(0))
    return merged + a + b


def part_2(signals: list[list[SignalElement]]) -> None:
    sorted_signals = get_sorted_signals(signals + [[[2]]] + [[[6]]])
    first_divider_index = sorted_signals.index([[2]]) + 1
    second_divider_index = sorted_signals.index([[6]]) + 1
    decoder_key = first_divider_index * second_divider_index
    print(f'part_2={decoder_key}')


if __name__ == '__main__':
    raw_input = get_raw_input()
    signal_pairs = get_signal_pairs(raw_input)
    part_1(signal_pairs)
    signals = [signal for pair in signal_pairs for signal in pair]
    part_2(signals)