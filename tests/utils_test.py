import pytest
from truco_setup.utils import compute_envido_score, determine_card_round_winner, create_partial_observation_state
from truco_setup.cards import Card


class TestComputeEnvidoScore:
    def test_same_suit_two_cards(self):
        cards = [
            Card(5, "Espada"),
            Card(6, "Espada"),
            Card(2, "Oro")
        ]
        assert compute_envido_score(cards) == 20 + 5 + 6

    def test_same_suit_three_cards(self):
        cards = [
            Card(5, "Espada"),
            Card(6, "Espada"),
            Card(7, "Espada")
        ]
        assert compute_envido_score(cards) == 20 + 7 + 6

    def test_face_cards_count_as_zero(self):
        cards = [
            Card(10, "Espada"),
            Card(11, "Espada"),
            Card(5, "Oro")
        ]
        assert compute_envido_score(cards) == 20 + 0 + 0

    def test_mixed_face_and_number_cards(self):
        cards = [
            Card(7, "Espada"),
            Card(12, "Espada"),
            Card(3, "Oro")
        ]
        assert compute_envido_score(cards) == 20 + 7 + 0

    def test_no_same_suit_cards(self):
        cards = [
            Card(7, "Espada"),
            Card(6, "Oro"),
            Card(5, "Copa")
        ]
        assert compute_envido_score(cards) == 7

    def test_all_face_cards_different_suits(self):
        cards = [
            Card(10, "Espada"),
            Card(11, "Oro"),
            Card(12, "Copa")
        ]
        assert compute_envido_score(cards) == 0

    def test_invalid_number_of_cards(self):
        with pytest.raises(ValueError, match="Envido score must be computed on exactly 3 cards"):
            compute_envido_score([Card(5, "Espada"), Card(6, "Oro")])

        with pytest.raises(ValueError, match="Envido score must be computed on exactly 3 cards"):
            compute_envido_score([Card(5, "Espada"), Card(6, "Oro"), Card(7, "Copa"), Card(3, "Basto")])


class TestDetermineCardRoundWinner:
    def test_card_a_wins(self):
        card_a = Card(1, "Espada")
        card_b = Card(7, "Oro")
        assert determine_card_round_winner(card_a, card_b) == 0

    def test_card_b_wins(self):
        card_a = Card(4, "Copa")
        card_b = Card(3, "Espada")
        assert determine_card_round_winner(card_a, card_b) == 1

    def test_tie(self):
        card_a = Card(6, "Espada")
        card_b = Card(6, "Oro")
        assert determine_card_round_winner(card_a, card_b) is None

    def test_special_cards(self):
        # Test special rankings
        ancho_espada = Card(1, "Espada")
        ancho_basto = Card(1, "Basto")
        siete_espada = Card(7, "Espada")
        siete_oro = Card(7, "Oro")

        assert determine_card_round_winner(ancho_espada, ancho_basto) == 0
        assert determine_card_round_winner(ancho_basto, siete_espada) == 0
        assert determine_card_round_winner(siete_espada, siete_oro) == 0
        assert determine_card_round_winner(siete_oro, Card(3, "Copa")) == 0


class TestCreatePartialObservationState:
    def test_hides_opponent_hand(self):
        full_state = {
            "hands": [
                ["1 de Espada", "2 de Oro", "3 de Copa"],
                ["7 de Espada", "10 de Basto", "11 de Copa"]
            ],
            "scores": [15, 20],
            "current_player": 0
        }

        obs = create_partial_observation_state(full_state, 0)
        assert obs["hands"][0] == ["1 de Espada", "2 de Oro", "3 de Copa"]
        assert obs["hands"][1] == ["[Hidden]", "[Hidden]", "[Hidden]"]
        assert obs["scores"] == [15, 20]

    def test_hides_opponent_hand_player_1(self):
        full_state = {
            "hands": [
                ["1 de Espada", "2 de Oro", "3 de Copa"],
                ["7 de Espada", "10 de Basto", "11 de Copa"]
            ],
            "scores": [15, 20],
            "current_player": 1
        }

        obs = create_partial_observation_state(full_state, 1)
        assert obs["hands"][0] == ["[Hidden]", "[Hidden]", "[Hidden]"]
        assert obs["hands"][1] == ["7 de Espada", "10 de Basto", "11 de Copa"]

    def test_preserves_other_state(self):
        full_state = {
            "hands": [["1 de Espada"], ["7 de Espada"]],
            "scores": [10, 15],
            "current_round": 2,
            "truco_stage": "Truco",
            "envido_pending": True
        }

        obs = create_partial_observation_state(full_state, 0)
        assert obs["scores"] == [10, 15]
        assert obs["current_round"] == 2
        assert obs["truco_stage"] == "Truco"
        assert obs["envido_pending"] == True

    def test_empty_hands(self):
        full_state = {
            "hands": [[], []],
            "scores": [25, 28]
        }

        obs = create_partial_observation_state(full_state, 0)
        assert obs["hands"][0] == []
        assert obs["hands"][1] == []