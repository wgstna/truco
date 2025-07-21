import pytest
from unittest.mock import MagicMock, patch
from truco_setup.basic_agents import RandomAgent, SimpleRuleBasedAgent, IntermediateAgent

def create_minimal_observation(current_player=0, **kwargs):
    obs = {
        "current_player": current_player,
        "hands": [[], []],
        "scores": [0, 0],
        "current_round": 1,
        "envido_finished": False,
        "round_cards_played": [[], []],
    }
    obs.update(kwargs)
    return obs


class TestRandomAgent:
    def test_choose_action_single_option(self):
        agent = RandomAgent()
        valid_actions = [{"type": "play_card", "card_index": 0}]
        action = agent.choose_action({}, valid_actions)
        assert action == valid_actions[0]

    def test_choose_action_multiple_options(self):
        agent = RandomAgent()
        valid_actions = [
            {"type": "play_card", "card_index": 0},
            {"type": "play_card", "card_index": 1},
            {"type": "call_envido", "which": "Envido"}
        ]
        actions_chosen = set()
        for _ in range(50):
            action = agent.choose_action({}, valid_actions)
            assert action in valid_actions
            actions_chosen.add(str(action))

        assert len(actions_chosen) > 1

    def test_record_reward(self):
        agent = RandomAgent()
        agent.record_reward(10.5)
        agent.record_reward(-5.0)

    def test_finish_episode_and_update(self):
        agent = RandomAgent()
        agent.finish_episode_and_update()


class TestSimpleRuleBasedAgent:
    def setup_method(self):
        self.agent = SimpleRuleBasedAgent()

    def test_card_value_map_creation(self):
        assert self.agent._get_card_value("1 de Espada") == 14
        assert self.agent._get_card_value("1 de Basto") == 13
        assert self.agent._get_card_value("7 de Espada") == 12
        assert self.agent._get_card_value("7 de Oro") == 11

        assert self.agent._get_card_value("3 de Copa") == 10
        assert self.agent._get_card_value("2 de Basto") == 9
        assert self.agent._get_card_value("4 de Oro") == 1

    def test_get_envido_score_two_same_suit(self):
        hand = ["5 de Espada", "7 de Espada", "10 de Oro"]
        score = self.agent._get_envido_score(hand)
        assert score == 20 + 5 + 7

    def test_get_envido_score_face_cards(self):
        hand = ["10 de Espada", "12 de Espada", "5 de Oro"]
        score = self.agent._get_envido_score(hand)
        assert score == 20 + 0 + 0

    def test_get_envido_score_no_same_suit(self):
        hand = ["7 de Espada", "6 de Oro", "5 de Copa"]
        score = self.agent._get_envido_score(hand)
        assert score == 7

    def test_handle_envido_response_high_score(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["7 de Espada", "6 de Espada", "5 de Oro"],
                []
            ],
            "envido_pending": True,
            "envido_caller": 1,
            "scores": [10, 15],
            "current_round": 1,
            "envido_finished": False
        }
        valid_actions = [
            {"type": "accept_envido"},
            {"type": "reject_envido"},
            {"type": "raise_envido", "which": "RealEnvido"}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] in ["raise_envido", "accept_envido"]

    def test_handle_envido_response_medium_score(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["5 de Espada", "4 de Espada", "3 de Oro"],
                []
            ],
            "envido_pending": True,
            "envido_caller": 1,
            "scores": [10, 15],
            "current_round": 1,
            "envido_finished": False
        }
        valid_actions = [
            {"type": "accept_envido"},
            {"type": "reject_envido"}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] == "accept_envido"

    def test_handle_envido_response_low_score(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["10 de Espada", "11 de Oro", "12 de Copa"],
                []
            ],
            "envido_pending": True,
            "envido_caller": 1,
            "scores": [10, 15],
            "current_round": 1,
            "envido_finished": False
        }
        valid_actions = [
            {"type": "accept_envido"},
            {"type": "reject_envido"}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] == "reject_envido"

    def test_handle_truco_response_high_cards(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["1 de Espada", "7 de Espada", "3 de Oro"],
                []
            ],
            "truco_pending": True,
            "truco_caller": 1,
            "scores": [10, 15],
            "current_round": 1,
            "envido_finished": False
        }
        valid_actions = [
            {"type": "accept_truco"},
            {"type": "reject_truco"},
            {"type": "raise_truco", "which": "ReTruco"}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] in ["raise_truco", "accept_truco"]

    def test_consider_envido_high_score(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["7 de Espada", "6 de Espada", "5 de Oro"],
                []
            ],
            "current_round": 1,
            "envido_finished": False,
            "scores": [10, 15],
            "round_cards_played": [[], []],
            "envido_pending": False,
            "truco_pending": False
        }
        valid_actions = [
            {"type": "call_envido", "which": "Envido"},
            {"type": "play_card", "card_index": 0}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] == "call_envido"

    def test_consider_truco_high_cards(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["1 de Espada", "1 de Basto", "3 de Oro"],
                []
            ],
            "current_round": 2,
            "envido_finished": True,
            "scores": [20, 22],
            "round_cards_played": [[], []],
            "envido_pending": False,
            "truco_pending": False
        }
        valid_actions = [
            {"type": "call_truco", "which": "Truco"},
            {"type": "play_card", "card_index": 0}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] == "call_truco"

    def test_choose_card_to_play_first_player(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["1 de Espada", "4 de Copa", "7 de Oro"],
                []
            ],
            "round_cards_played": [[], []],
            "scores": [10, 15],
            "current_round": 1,
            "envido_finished": False
        }
        valid_actions = [
            {"type": "play_card", "card_index": 0},
            {"type": "play_card", "card_index": 1},
            {"type": "play_card", "card_index": 2}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] == "play_card"
        assert action["card_index"] == 2


class TestIntermediateAgent:
    def setup_method(self):
        self.agent = IntermediateAgent(aggression_level=0.5)

    def test_initialization(self):
        assert self.agent.aggression_level == 0.5
        assert self.agent.bluff_probability == 0.15
        assert self.agent.opponent_stats["total_games"] == 0

    def test_get_envido_score(self):
        hand = ["5 de Espada", "7 de Espada", "10 de Oro"]
        score = self.agent._get_envido_score(hand)
        assert score == 32

    def test_bluff_envido(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["4 de Espada", "5 de Oro", "6 de Copa"],
                []
            ],
            "current_round": 1,
            "scores": [10, 15],
            "envido_finished": False,
            "round_cards_played": [[], []]
        }
        valid_actions = [
            {"type": "call_envido", "which": "Envido"},
            {"type": "play_card", "card_index": 0}
        ]

        with patch('random.random', return_value=0.05):
            action = self.agent.choose_action(obs, valid_actions)
            assert action["type"] == "call_envido"

    def test_bluff_truco(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["4 de Copa", "5 de Oro", "6 de Basto"],
                []
            ],
            "current_round": 2,
            "envido_finished": True,
            "scores": [20, 22],
            "round_cards_played": [[], []],
            "truco_pending": False,
            "envido_pending": False
        }
        valid_actions = [
            {"type": "call_truco", "which": "Truco"},
            {"type": "play_card", "card_index": 0}
        ]

        with patch('random.random', return_value=0.05):
            action = self.agent.choose_action(obs, valid_actions)
            assert action["type"] in ["call_truco", "play_card"]

    def test_would_win_against(self):
        assert self.agent._would_win_against("1 de Espada", "7 de Oro") == True
        assert self.agent._would_win_against("4 de Copa", "3 de Espada") == False
        assert self.agent._would_win_against("6 de Espada", "6 de Oro") == False

    def test_choose_card_responding_to_opponent(self):
        obs = {
            "current_player": 0,
            "hands": [
                ["1 de Espada", "4 de Copa", "7 de Oro"],
                []
            ],
            "round_cards_played": [[], ["6 de Basto"]],
            "scores": [10, 15],
            "current_round": 1,
            "envido_finished": False
        }
        valid_actions = [
            {"type": "play_card", "card_index": 0},
            {"type": "play_card", "card_index": 1},
            {"type": "play_card", "card_index": 2}
        ]

        action = self.agent.choose_action(obs, valid_actions)
        assert action["type"] == "play_card"
        assert action["card_index"] == 2

    def test_finish_episode_updates_stats(self):
        initial_games = self.agent.opponent_stats["total_games"]
        self.agent.finish_episode_and_update()
        assert self.agent.opponent_stats["total_games"] == initial_games + 1

    def test_empty_valid_actions(self):
        obs = {"current_player": 0}
        action = self.agent.choose_action(obs, [])
        assert action is None

    def test_malformed_observation(self):
        obs = {}
        valid_actions = [{"type": "play_card", "card_index": 0}]
        action = self.agent.choose_action(obs, valid_actions)
        assert action is not None
