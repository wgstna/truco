from typing import List, Optional, Tuple
import random

SUITS = ["Espada", "Basto", "Oro", "Copa"]

CARD_RANK_OVERRIDES = {
    (1, "Espada"): 14,  # Ancho de Espada (Macho)
    (1, "Basto"): 13,  # Ancho de Basto (Hembra)
    (7, "Espada"): 12,  # Siete de Espada
    (7, "Oro"): 11,  # Siete de Oro
}


def get_card_rank(value: int, suit: str) -> int:
    if suit not in SUITS:
        raise ValueError(f"Unrecognized suit: {suit}")

    if value not in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]:
        raise ValueError(f"Invalid Truco card value: {value}")

    if (value, suit) in CARD_RANK_OVERRIDES:
        return CARD_RANK_OVERRIDES[(value, suit)]

    if value == 3:
        return 10
    elif value == 2:
        return 9
    elif value == 1:
        # Must be Oro/Copa => 8
        return 8
    elif value == 12:
        return 7
    elif value == 11:
        return 6
    elif value == 10:
        return 5
    elif value == 7:
        # Must be Basto or Copa => 4
        return 4
    elif value == 6:
        return 3
    elif value == 5:
        return 2
    elif value == 4:
        return 1

    raise ValueError(f"Unexpected card value/suit: ({value}, {suit})")

class Card:

    __slots__ = ['value', 'suit', 'power']

    def __init__(self, value: int, suit: str):
        self.value = value
        self.suit = suit
        self.power = get_card_rank(value, suit)

    def __str__(self) -> str:
        return f"{self.value} de {self.suit}"

    def __repr__(self) -> str:
        return f"Card(value={self.value}, suit='{self.suit}', power={self.power})"

    def __lt__(self, other) -> bool:
        return self.power < other.power

    def __gt__(self, other) -> bool:
        return self.power > other.power

    def __eq__(self, other) -> bool:
        return self.power == other.power


class Deck:

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._random = random.Random(seed)
        self.cards: List[Card] = []
        self.reset()

    def _build_deck(self) -> List[Card]:
        deck = []
        for suit in SUITS:
            for val in range(1, 13):
                # Exclude 8 and 9
                if val in [8, 9]:
                    continue
                deck.append(Card(val, suit))
        return deck

    def shuffle(self) -> None:
        self._random.shuffle(self.cards)

    def draw(self, n: int = 1) -> List[Card]:
        if n > len(self.cards):
            raise ValueError(f"Cannot draw {n} cards; only {len(self.cards)} remain.")
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def reset(self) -> None:
        self._random = random.Random(self.seed)
        self.cards = self._build_deck()
        self.shuffle()

    def count(self) -> int:
        return len(self.cards)