
from typing import List, Optional, Tuple

from truco_setup.cards import Card


def compute_envido_score(cards: List[Card]) -> int:
    if len(cards) != 3:
        raise ValueError("Envido score must be computed on exactly 3 cards.")

    suit_map = {}
    for c in cards:
        single_digit = c.value if c.value < 10 else 0
        suit_map.setdefault(c.suit, []).append(single_digit)

    best_score = 0
    for suit, values in suit_map.items():
        if len(values) == 2:
            score = 20 + sum(values)
            if score > best_score:
                best_score = score
        elif len(values) == 3:
            sorted_vals = sorted(values, reverse=True)
            score = 20 + sorted_vals[0] + sorted_vals[1]
            if score > best_score:
                best_score = score

    if best_score == 0:
        single_digit_vals = [c.value if c.value < 10 else 0 for c in cards]
        best_score = max(single_digit_vals)

    return best_score


def determine_card_round_winner(cardA: Card, cardB: Card) -> Optional[int]:
    if cardA > cardB:
        return 0
    elif cardB > cardA:
        return 1
    else:
        return None


def create_partial_observation_state(
    full_state: dict,
    viewing_player: int
) -> dict:
    obs = dict(full_state)

    opp = 1 - viewing_player

    hidden_hand = ["[Hidden]" for _ in obs["hands"][opp]]
    obs["hands"][opp] = hidden_hand

    return obs