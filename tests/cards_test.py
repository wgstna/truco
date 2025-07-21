import pytest
from truco_setup.cards import SUITS, CARD_RANK_OVERRIDES, get_card_rank, Card, Deck


class TestGetCardRank:
    def test_special_cards(self):
        assert get_card_rank(1, "Espada") == 14
        assert get_card_rank(1, "Basto") == 13
        assert get_card_rank(7, "Espada") == 12
        assert get_card_rank(7, "Oro") == 11

    def test_regular_rankings(self):
        for suit in SUITS:
            assert get_card_rank(3, suit) == 10

        for suit in SUITS:
            assert get_card_rank(2, suit) == 9

        assert get_card_rank(1, "Oro") == 8
        assert get_card_rank(1, "Copa") == 8

        for suit in SUITS:
            assert get_card_rank(12, suit) == 7

        for suit in SUITS:
            assert get_card_rank(11, suit) == 6

        for suit in SUITS:
            assert get_card_rank(10, suit) == 5

        assert get_card_rank(7, "Basto") == 4
        assert get_card_rank(7, "Copa") == 4

        for suit in SUITS:
            assert get_card_rank(6, suit) == 3

        for suit in SUITS:
            assert get_card_rank(5, suit) == 2

        for suit in SUITS:
            assert get_card_rank(4, suit) == 1

    def test_invalid_suit(self):
        with pytest.raises(ValueError, match="Unrecognized suit"):
            get_card_rank(5, "InvalidSuit")

    def test_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid Truco card value"):
            get_card_rank(8, "Espada")

        with pytest.raises(ValueError, match="Invalid Truco card value"):
            get_card_rank(9, "Copa")

        with pytest.raises(ValueError, match="Invalid Truco card value"):
            get_card_rank(0, "Espada")

        with pytest.raises(ValueError, match="Invalid Truco card value"):
            get_card_rank(13, "Oro")


class TestCard:
    def test_card_creation(self):
        card = Card(1, "Espada")
        assert card.value == 1
        assert card.suit == "Espada"
        assert card.power == 14

    def test_card_str_representation(self):
        card = Card(7, "Oro")
        assert str(card) == "7 de Oro"

    def test_card_repr(self):
        card = Card(3, "Copa")
        assert repr(card) == "Card(value=3, suit='Copa', power=10)"

    def test_card_comparison_less_than(self):
        card1 = Card(4, "Copa")
        card2 = Card(3, "Espada")
        assert card1 < card2
        assert not card2 < card1

    def test_card_comparison_greater_than(self):
        card1 = Card(1, "Espada")
        card2 = Card(7, "Oro")
        assert card1 > card2
        assert not card2 > card1

    def test_card_comparison_equal(self):
        card1 = Card(6, "Espada")
        card2 = Card(6, "Oro")
        assert card1 == card2

    def test_special_cards_comparison(self):
        ancho_espada = Card(1, "Espada")
        ancho_basto = Card(1, "Basto")
        siete_espada = Card(7, "Espada")
        siete_oro = Card(7, "Oro")
        regular_3 = Card(3, "Copa")

        assert ancho_espada > ancho_basto
        assert ancho_basto > siete_espada
        assert siete_espada > siete_oro
        assert siete_oro > regular_3


class TestDeck:
    def test_deck_initialization(self):
        deck = Deck()
        assert len(deck.cards) == 40
        assert deck.seed is None

    def test_deck_with_seed(self):
        deck1 = Deck(seed=42)
        deck2 = Deck(seed=42)
        cards1 = [str(card) for card in deck1.cards[:5]]
        cards2 = [str(card) for card in deck2.cards[:5]]
        assert cards1 == cards2

    def test_build_deck(self):
        deck = Deck()
        assert deck.count() == 40
        card_strings = [str(card) for card in deck.cards]
        for suit in SUITS:
            for value in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]:
                assert f"{value} de {suit}" in card_strings

    def test_shuffle(self):
        deck = Deck(seed=123)
        original_order = [str(card) for card in deck.cards]
        deck.shuffle()
        shuffled_order = [str(card) for card in deck.cards]
        assert original_order != shuffled_order
        assert set(original_order) == set(shuffled_order)

    def test_draw_single_card(self):
        deck = Deck()
        initial_count = deck.count()
        drawn = deck.draw(1)
        assert len(drawn) == 1
        assert isinstance(drawn[0], Card)
        assert deck.count() == initial_count - 1

    def test_draw_multiple_cards(self):
        deck = Deck()
        drawn = deck.draw(3)
        assert len(drawn) == 3
        assert all(isinstance(card, Card) for card in drawn)
        assert deck.count() == 37

    def test_draw_too_many_cards(self):
        deck = Deck()
        with pytest.raises(ValueError, match="Cannot draw 41 cards; only 40 remain"):
            deck.draw(41)

    def test_draw_all_cards(self):
        deck = Deck()
        drawn = deck.draw(40)
        assert len(drawn) == 40
        assert deck.count() == 0
        with pytest.raises(ValueError):
            deck.draw(1)

    def test_reset(self):
        deck = Deck(seed=999)
        deck.draw(10)
        assert deck.count() == 30
        deck.reset()
        assert deck.count() == 40
        deck2 = Deck(seed=999)
        assert [str(card) for card in deck.cards] == [str(card) for card in deck2.cards]

    def test_count(self):
        deck = Deck()
        assert deck.count() == 40
        deck.draw(5)
        assert deck.count() == 35
        deck.draw(10)
        assert deck.count() == 25

    def test_deck_randomness(self):
        deck1 = Deck()
        deck2 = Deck()
        order1 = [str(card) for card in deck1.cards[:10]]
        order2 = [str(card) for card in deck2.cards[:10]]
        assert order1 != order2