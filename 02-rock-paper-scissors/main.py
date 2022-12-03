"""
https://adventofcode.com/2022/day/2
"""

from enum import Enum


class Hand(Enum):
    Rock = 0
    Paper = 1
    Scissors = 2

HANDS_BY_CODE = {
    'A': Hand.Rock,
    'B': Hand.Paper,
    'C': Hand.Scissors,
}

class Outcome(Enum):
    Loss = 0
    Draw = 3
    Win = 6

OUTCOMES_BY_CODE = {
    'X': Outcome.Loss,
    'Y': Outcome.Draw,
    'Z': Outcome.Win,
}

    
def determine_match_outcome(your_hand: Hand, opponent_hand: Hand) -> Outcome:
    # In modulo 3, every hand loses to the hand above it and wins against the hand below it
    your_hand_value = your_hand.value
    opponent_hand_value = opponent_hand.value
    return {
        (your_hand_value + 1) % 3: Outcome.Loss,
        (your_hand_value - 1) % 3: Outcome.Win,
        your_hand_value: Outcome.Draw,
    }.get(opponent_hand_value)


def determine_score_of_match(your_hand: Hand, opponent_hand: Hand) -> int:
    outcome_score = determine_match_outcome(your_hand, opponent_hand).value
    your_hand_score = your_hand.value + 1
    return outcome_score + your_hand_score


def get_hands_part_1(strategy: str) -> list[list[Hand]]:
    strategy = strategy.strip()
    # Normalize inputs
    REPLACEMENTS = {
        'X': 'A',
        'Y': 'B',
        'Z': 'C',
    }
    for old, new in REPLACEMENTS.items():
        strategy = strategy.replace(old, new)
    strategy_lines = strategy.split('\n')
    return list(map(lambda line: list(map(HANDS_BY_CODE.get, line.split())), strategy_lines))


def determine_strategy_score(strategy_hands: list[list[Hand]]) -> int:
    total_score = 0
    for opponent_hand, your_hand in strategy_hands:
        total_score += determine_score_of_match(your_hand, opponent_hand)
    return total_score


def get_hand_from_opponent_and_outcome(opponent: Hand, outcome: Outcome) -> Hand:
    your_hand_value = opponent.value
    # In modulo 3, every hand loses to the hand above it and wins against the hand below it
    match outcome:
        case Outcome.Loss:
            your_hand_value -= 1
        case Outcome.Win:
            your_hand_value += 1
    your_hand_value = your_hand_value % 3
    return Hand(your_hand_value)


def get_opponent_and_outcome(strategy: str) -> list[tuple[Hand, Outcome]]:
    return_list = []
    for line in strategy.strip().split('\n'):
        opponent_hand_code, outcome_code = line.split()
        opponent_hand = HANDS_BY_CODE[opponent_hand_code]
        outcome = OUTCOMES_BY_CODE[outcome_code]
        return_list.append((opponent_hand, outcome))
    return return_list


def get_hands_part_2(strategy: str) -> list[list[Hand]]:
    opponent_hand_and_outcome = get_opponent_and_outcome(strategy)
    return [
        [opponent_hand, get_hand_from_opponent_and_outcome(opponent_hand, outcome)] 
        for opponent_hand, outcome in opponent_hand_and_outcome
    ]


def main():
    strategy = open('02-rock-paper-scissors/input.txt', 'r').read()
    part_1_hands = get_hands_part_1(strategy)
    part_1_score = determine_strategy_score(part_1_hands)
    part_2_hands = get_hands_part_2(strategy)
    part_2_score = determine_strategy_score(part_2_hands)
    print(f'{part_1_score=}')
    print(f'{part_2_score=}')


if __name__ == '__main__':
    main()