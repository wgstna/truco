import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from truco_setup.truco_environment import TrucoEnvironment
from truco_setup.basic_agents import RandomAgent, SimpleRuleBasedAgent
from truco_setup.cards import Card

class TestTrucoEnvironmentInitialization:
    def test_default_initialization(self):
        env = TrucoEnvironment()
        assert env.master_seed is None
        assert env.verbose is False
        assert env.use_encoded_cards is False
        assert env.difficulty_level == 0
        assert env.enable_curriculum is False
        assert env.reward_scale == 1.0
        assert env.scores == [0, 0]
        assert env.max_points == 30
        assert env.game_over is False

    def test_initialization_with_parameters(self):
        env = TrucoEnvironment(
            seed=42,
            verbose=True,
            use_encoded_cards=True,
            difficulty_level=3,
            enable_curriculum=True,
            reward_scale=2.0
        )
        assert env.master_seed == 42
        assert env.verbose is True
        assert env.use_encoded_cards is True
        assert env.difficulty_level == 3
        assert env.enable_curriculum is True
        assert env.reward_scale == 2.0

    def test_card_encoding_setup(self):
        env = TrucoEnvironment()
        assert env.num_cards == 41
        assert env.hidden_card_id == 0

        assert "1 de Espada" in env.card_to_id
        assert "7 de Oro" in env.card_to_id
        assert "12 de Copa" in env.card_to_id

        assert env.card_strength["1 de Espada"] == 14
        assert env.card_strength["1 de Basto"] == 13
        assert env.card_strength["7 de Espada"] == 12
        assert env.card_strength["7 de Oro"] == 11


class TestGameState:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42)

    def test_reset(self):
        self.env.scores = [15, 20]
        self.env.game_over = True
        self.env.hands_played = 5

        obs = self.env.reset()

        assert self.env.scores == [0, 0]
        assert self.env.game_over is False
        assert self.env.hands_played == 1
        assert self.env.current_hand_active is True
        assert len(self.env.hands[0]) == 3
        assert len(self.env.hands[1]) == 3

        assert "scores" in obs
        assert "hands" in obs
        assert "current_player" in obs

    def test_start_new_hand(self):
        self.env.reset()
        initial_hands_played = self.env.hands_played

        self.env._start_new_hand()

        assert self.env.hands_played == initial_hands_played + 1
        assert len(self.env.hands[0]) == 3
        assert len(self.env.hands[1]) == 3
        assert self.env.current_round == 1
        assert self.env.round_wins == [0, 0]
        assert self.env.envido_finished is False
        assert self.env.truco_stage is None

    def test_dealer_rotation(self):
        self.env.reset()
        initial_dealer = self.env.dealer
        initial_mano = self.env.mano

        self.env._rotate_dealer_and_start_new()

        assert self.env.dealer == initial_mano
        assert self.env.mano == initial_dealer


class TestValidActions:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42)
        self.env.reset()

    def test_initial_valid_actions(self):
        actions = self.env.get_valid_actions()

        action_types = [a["type"] for a in actions]
        assert "play_card" in action_types
        assert "call_envido" in action_types

        play_card_actions = [a for a in actions if a["type"] == "play_card"]
        assert len(play_card_actions) == 3

    def test_envido_response_actions(self):
        self.env.envido_pending = True
        self.env.envido_caller = 0
        self.env.envido_stage = "Envido"
        self.env.current_player = 1

        actions = self.env.get_valid_actions()
        action_types = [a["type"] for a in actions]

        assert "accept_envido" in action_types
        assert "reject_envido" in action_types
        assert "raise_envido" in action_types

    def test_truco_response_actions(self):
        self.env.truco_pending = True
        self.env.truco_caller = 0
        self.env.truco_stage = "Truco"
        self.env.current_player = 1

        actions = self.env.get_valid_actions()
        action_types = [a["type"] for a in actions]

        assert "accept_truco" in action_types
        assert "reject_truco" in action_types
        assert "raise_truco" in action_types

    def test_no_envido_after_first_round(self):
        self.env.current_round = 2

        actions = self.env.get_valid_actions()
        action_types = [a["type"] for a in actions]

        assert "call_envido" not in action_types

    def test_no_envido_if_finished(self):
        self.env.envido_finished = True

        actions = self.env.get_valid_actions()
        action_types = [a["type"] for a in actions]

        assert "call_envido" not in action_types


class TestCardPlaying:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42, verbose=False)
        self.env.reset()

    def test_play_card_valid(self):
        initial_hand_size = len(self.env.hands[self.env.current_player])

        action = {"type": "play_card", "card_index": 0}
        obs, reward, done, info = self.env.step(action)

        assert len(self.env.hands[1 - self.env.current_player]) == initial_hand_size - 1

        assert len(self.env.round_cards_played[1 - self.env.current_player]) == 1

        assert self.env.current_player != self.env.mano

    def test_play_card_invalid_index(self):
        action = {"type": "play_card", "card_index": 5}
        obs, reward, done, info = self.env.step(action)

        assert reward == -5.0
        assert "error" in info

    def test_round_winner_determination(self):
        self.env.hands[0] = [Card(1, "Espada"), Card(4, "Copa"), Card(5, "Oro")]
        self.env.hands[1] = [Card(7, "Oro"), Card(6, "Basto"), Card(3, "Copa")]

        self.env.current_player = 0
        self.env.step({"type": "play_card", "card_index": 1})

        self.env.current_player = 1
        self.env.step({"type": "play_card", "card_index": 2})

        assert self.env.round_wins[1] == 1
        assert self.env.round_wins[0] == 0


class TestEnvido:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42, verbose=False)
        self.env.reset()

    def test_call_envido(self):
        action = {"type": "call_envido", "which": "Envido"}
        obs, reward, done, info = self.env.step(action)

        assert self.env.envido_pending is True
        assert self.env.envido_stage == "Envido"
        assert self.env.envido_caller == 1 - self.env.current_player

    def test_accept_envido(self):
        self.env.current_player = 0
        self.env.step({"type": "call_envido", "which": "Envido"})

        self.env.step({"type": "accept_envido"})

        assert self.env.envido_finished is True
        assert self.env.envido_pending is False

        assert sum(self.env.scores) > 0

    def test_reject_envido(self):
        self.env.current_player = 0
        self.env.step({"type": "call_envido", "which": "Envido"})

        self.env.step({"type": "reject_envido"})

        assert self.env.envido_finished is True
        assert self.env.envido_pending is False

        assert self.env.scores[0] == 1

    def test_raise_envido(self):
        self.env.current_player = 0
        self.env.step({"type": "call_envido", "which": "Envido"})

        self.env.step({"type": "raise_envido", "which": "RealEnvido"})

        assert self.env.envido_stage == "RealEnvido"
        assert self.env.envido_pending is True
        assert self.env.envido_caller == 1

    def test_envido_score_calculation(self):
        self.env.original_hands[0] = [
            Card(5, "Espada"),
            Card(6, "Espada"),
            Card(3, "Oro")
        ]

        self.env.original_hands[1] = [
            Card(7, "Copa"),
            Card(2, "Copa"),
            Card(1, "Basto")
        ]

        self.env.current_player = 0
        self.env.step({"type": "call_envido", "which": "Envido"})

        self.env.step({"type": "accept_envido"})

        assert self.env.scores[0] == 2
        assert self.env.scores[1] == 0


class TestTruco:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42, verbose=False)
        self.env.reset()

    def test_call_truco(self):
        action = {"type": "call_truco", "which": "Truco"}
        obs, reward, done, info = self.env.step(action)

        assert self.env.truco_pending is True
        assert self.env.truco_stage == "Truco"
        assert self.env.truco_caller == 1 - self.env.current_player

    def test_accept_truco(self):
        self.env.current_player = 0
        self.env.step({"type": "call_truco", "which": "Truco"})

        self.env.step({"type": "accept_truco"})

        assert self.env.truco_pending is False
        assert self.env.truco_stage == "Truco"

    def test_reject_truco(self):
        self.env.current_player = 0
        self.env.step({"type": "call_truco", "which": "Truco"})

        self.env.step({"type": "reject_truco"})
        assert self.env.scores[0] == 1
        assert self.env.current_hand_active is True

    def test_raise_truco(self):
        self.env.current_player = 0
        self.env.step({"type": "call_truco", "which": "Truco"})

        self.env.step({"type": "raise_truco", "which": "ReTruco"})

        assert self.env.truco_stage == "ReTruco"
        assert self.env.truco_pending is True
        assert self.env.truco_caller == 1


class TestGameEnd:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42, verbose=False)

    def test_game_ends_at_30_points(self):
        self.env.reset()
        self.env.scores = [29, 10]

        self.env.hands[0] = [Card(1, "Espada")]
        self.env.hands[1] = [Card(4, "Copa")]
        self.env.round_wins = [1, 0]

        self.env.current_player = 0
        self.env.step({"type": "play_card", "card_index": 0})
        self.env.step({"type": "play_card", "card_index": 0})

        assert self.env.game_over is True
        assert self.env.scores[0] >= 30


class TestObservations:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42, use_encoded_cards=False)
        self.env.reset()

    def test_partial_observation(self):
        obs = self.env._get_observation(full_info=False)

        current_p = self.env.current_player
        opponent = 1 - current_p

        assert all(card != "Hidden Card" for card in obs["hands"][0])

        assert all(card == "Hidden Card" for card in obs["hands"][1])

        assert "scores" in obs
        assert "current_round" in obs
        assert "round_cards_played" in obs

    def test_full_observation(self):
        obs = self.env._get_observation(full_info=True)

        assert all(card != "Hidden Card" for card in obs["hands"][0])
        assert all(card != "Hidden Card" for card in obs["hands"][1])

    def test_encoded_cards_observation(self):
        env = TrucoEnvironment(seed=42, use_encoded_cards=True)
        env.reset()

        obs = env._get_observation(full_info=False)

        assert all(isinstance(card, int) for card in obs["hands"][0])
        assert all(card == env.hidden_card_id for card in obs["hands"][1])


class TestRewards:
    def setup_method(self):
        self.env = TrucoEnvironment(seed=42, verbose=False, reward_scale=1.0)

    def test_reward_scaling(self):
        env_scaled = TrucoEnvironment(seed=42, verbose=False, reward_scale=2.0)
        env_scaled.reset()

        self.env.reset()
        self.env._add_partial_reward(0, 1.0, "test")
        env_scaled._add_partial_reward(0, 1.0, "test")

        assert env_scaled.partial_rewards[0] == 2.0
        assert self.env.partial_rewards[0] == 1.0

    def test_partial_rewards_accumulation(self):
        self.env.reset()

        self.env._add_partial_reward(0, 0.5, "test1")
        self.env._add_partial_reward(0, 0.3, "test2")

        assert self.env.partial_rewards[0] == 0.8
        assert self.env.total_game_rewards[0] == 0.8

    def test_rewards_cleared_after_step(self):
        self.env.reset()

        self.env._add_partial_reward(self.env.current_player, 1.0, "test")
        initial_reward = self.env.partial_rewards[self.env.current_player]

        obs, reward, done, info = self.env.step({"type": "play_card", "card_index": 0})

        assert reward == initial_reward
        assert self.env.partial_rewards[1 - self.env.current_player] == 0.0


class TestPlayFullGame:
    def test_game_completes(self):
        env = TrucoEnvironment(seed=42, verbose=False)
        agent0 = RandomAgent()
        agent1 = RandomAgent()

        result = env.play_full_game(agent0, agent1)

        assert "winner" in result
        assert result["winner"] in [0, 1]
        assert "scores" in result
        assert max(result["scores"]) >= 30
        assert "hand_history" in result
        assert "total_rewards" in result

    def test_deterministic_with_seed(self):
        env1 = TrucoEnvironment(seed=123, use_encoded_cards=False)
        env2 = TrucoEnvironment(seed=123, use_encoded_cards=False)

        agent0_1 = RandomAgent()
        agent1_1 = RandomAgent()
        agent0_2 = RandomAgent()
        agent1_2 = RandomAgent()

        import random
        random.seed(456)
        result1 = env1.play_full_game(agent0_1, agent1_1)

        random.seed(456)
        result2 = env2.play_full_game(agent0_2, agent1_2)

        assert result1["winner"] == result2["winner"]
        assert result1["scores"] == result2["scores"]

    def test_simple_agent_compatibility(self):
        env = TrucoEnvironment(seed=42, use_encoded_cards=False, verbose=False)
        agent0 = SimpleRuleBasedAgent()
        agent1 = RandomAgent()

        result = env.play_full_game(agent0, agent1)

        assert "winner" in result
        assert result["winner"] in [0, 1]
        assert max(result["scores"]) >= 30


class TestCurriculumLearning:
    def test_set_difficulty(self):
        env = TrucoEnvironment(enable_curriculum=True)

        env.set_curriculum_difficulty(3)
        assert env.difficulty_level == 3

        env.set_curriculum_difficulty(10)
        assert env.difficulty_level == 5

        env.set_curriculum_difficulty(-1)
        assert env.difficulty_level == 0

    def test_get_curriculum_opponent(self):
        env = TrucoEnvironment(enable_curriculum=True)

        env.set_curriculum_difficulty(0)
        opponent = env.get_curriculum_opponent()
        assert isinstance(opponent, RandomAgent)

        env.set_curriculum_difficulty(1)
        opponent = env.get_curriculum_opponent()
        assert isinstance(opponent, SimpleRuleBasedAgent)


class TestStateEncoding:
    def test_get_state_encoding(self):
        env = TrucoEnvironment(seed=42)
        env.reset()

        encoding = env.get_state_encoding()

        assert isinstance(encoding, np.ndarray)
        assert encoding.dtype == np.float32
        assert len(encoding) > 0
        assert 0 <= encoding[0] <= 1
        assert 0 <= encoding[1] <= 1

    def test_get_action_mask(self):
        env = TrucoEnvironment(seed=42)
        env.reset()

        mask = env.get_action_mask()

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == 20
        assert np.sum(mask) > 0
        assert mask[0] or mask[1] or mask[2]


class TestMetricsAndStatistics:
    def test_episode_statistics(self):
        env = TrucoEnvironment(seed=42)
        env.reset()

        env.step({"type": "play_card", "card_index": 0})
        env.step({"type": "play_card", "card_index": 0})

        stats = env.get_episode_statistics()

        assert isinstance(stats, dict)
        assert "player_0_hands_played" in stats
        assert "player_1_hands_played" in stats
        assert "player_0_total_rewards" in stats
        assert "player_1_total_rewards" in stats

    def test_save_performance_metrics(self, tmp_path):
        env = TrucoEnvironment(seed=42)
        env.reset()

        filepath = tmp_path / "metrics.json"
        env.save_performance_metrics(str(filepath))

        assert filepath.exists()

        import json
        with open(filepath) as f:
            data = json.load(f)

        assert "games_played" in data
        assert "win_rate" in data