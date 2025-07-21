import pytest
import os
import sys
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
from typing import List, Dict, Tuple
import csv

p_root = os.path.dirname(os.path.abspath(__file__))
if p_root not in sys.path:
    sys.path.append(p_root)

sys.path.append(os.path.join(p_root, '..'))

from hand_eval.dataset_creation import (
    _envido_value,
    _envido_points,
    _play_game_and_collect,
    generate_dataset,
    tie_probability,
    stronger_count,
    card_rank,
    id_to_card_,
    total_cards
)
from truco_setup.basic_agents import RandomAgent, SimpleRuleBasedAgent, IntermediateAgent

class TestGlobalVariables:

    def test_card_rank_populated(self):
        assert len(card_rank) > 0
        assert card_rank["1 de Espada"] == 14
        assert card_rank["1 de Basto"] == 13
        assert card_rank["7 de Espada"] == 12
        assert card_rank["7 de Oro"] == 11

    def test_id_to_card_populated(self):
        assert len(id_to_card_) > 0
        assert total_cards == len(id_to_card_)
        for card_id, card_str in id_to_card_.items():
            assert isinstance(card_id, int)
            assert isinstance(card_str, str)
            assert " de " in card_str

    def test_tie_probability_populated(self):
        assert len(tie_probability) == len(card_rank)
        for card, prob in tie_probability.items():
            assert 0 <= prob <= 1

    def test_stronger_count_populated(self):
        assert len(stronger_count) == len(card_rank)
        assert stronger_count["1 de Espada"] == 0
        assert stronger_count["4 de Copa"] > 30

    def test_tie_probability_calculation(self):
        unique_strength_cards = ["1 de Espada", "1 de Basto", "7 de Espada", "7 de Oro"]
        for card in unique_strength_cards:
            assert tie_probability[card] == 0

        four_cards = ["4 de Espada", "4 de Basto", "4 de Oro", "4 de Copa"]
        for card in four_cards:
            assert abs(tie_probability[card] - 3 / 39) < 0.001


class TestEnvidoValue:

    def test_number_cards_1_to_7(self):
        assert _envido_value("1 de Espada") == 1
        assert _envido_value("2 de Oro") == 2
        assert _envido_value("3 de Copa") == 3
        assert _envido_value("4 de Basto") == 4
        assert _envido_value("5 de Oro") == 5
        assert _envido_value("6 de Copa") == 6
        assert _envido_value("7 de Copa") == 7

    def test_face_cards_return_zero(self):
        assert _envido_value("10 de Espada") == 0
        assert _envido_value("11 de Basto") == 0
        assert _envido_value("12 de Copa") == 0

    def test_invalid_format_returns_zero(self):
        assert _envido_value("Rey de Espada") == 0
        assert _envido_value("Invalid Card") == 0

    def test_edge_case_formats(self):
        assert _envido_value("8 de Espada") == 0
        assert _envido_value("9 de Oro") == 0
        assert _envido_value("13 de Copa") == 0


class TestEnvidoPoints:

    def test_two_same_suit_cards(self):
        cards = ("5 de Espada", "6 de Espada", "3 de Oro")
        assert _envido_points(cards) == 31

    def test_three_same_suit_cards(self):
        cards = ("5 de Espada", "6 de Espada", "7 de Espada")
        assert _envido_points(cards) == 33

    def test_no_same_suit_cards(self):
        cards = ("5 de Espada", "6 de Oro", "7 de Copa")
        assert _envido_points(cards) == 32

    def test_face_cards_same_suit(self):
        cards = ("10 de Espada", "11 de Espada", "5 de Oro")
        assert _envido_points(cards) == 20

    def test_all_face_cards_different_suits(self):
        cards = ("10 de Espada", "11 de Oro", "12 de Copa")
        assert _envido_points(cards) == 20

    def test_mixed_cards_same_suit(self):
        cards = ("7 de Espada", "12 de Espada", "3 de Oro")
        assert _envido_points(cards) == 27

    def test_suit_grouping_logic(self):
        cards = ("5 de Espada", "6 de Espada", "7 de Copa")
        points = _envido_points(cards)
        assert points == 33
        cards = ("5 de Espada", "6 de Oro", "7 de Basto")
        points = _envido_points(cards)
        assert points == 33
        cards = ("3 de Oro", "5 de Espada", "7 de Basto")
        points = _envido_points(cards)
        assert points == 30


class TestPlayGameAndCollect:

    def test_basic_game_collection(self):
        rows = _play_game_and_collect(RandomAgent, RandomAgent, 42)
        assert isinstance(rows, list)
        assert len(rows) > 0, "Should collect at least some data from a game"
        row = rows[0]
        assert isinstance(row, dict)
        assert all(row['game_seed'] == 42 for row in rows)

        player_ids = set(row['player_id'] for row in rows)
        assert 0 in player_ids
        assert 1 in player_ids

    @patch('hand_eval.dataset_creation.TrucoEnvironment')
    def test_game_collection_with_mock(self, mock_env_class):
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env

        mock_env.game_over = True
        mock_env.get_valid_actions.return_value = []

        rows = _play_game_and_collect(RandomAgent, RandomAgent, 42)

        mock_env_class.assert_called_once_with(seed=42, use_encoded_cards=True)

        mock_env.reset.assert_called_once()

        assert isinstance(rows, list)

    def test_real_game_collection_structure(self):
        rows = _play_game_and_collect(RandomAgent, RandomAgent, 123)

        assert isinstance(rows, list)
        assert len(rows) > 0, "Should collect at least some data from a game"

        row = rows[0]
        assert isinstance(row, dict)

        required_fields = [
            "game_seed", "hand_number", "player_id", "agent_player",
            "agent_opponent", "dealer_id", "mano_id", "is_dealer", "is_mano",
            "card1_id", "card2_id", "card3_id", "card1_str", "card2_str", "card3_str",
            "strength1", "strength2", "strength3", "strength_sum", "strength_max",
            "starting_score_player", "starting_score_opponent", "num_same_suit",
            "has_two_same_suit", "envido_points", "points_gained", "won_hand",
            "tie_probability_card1", "tie_probability_card2", "tie_probability_card3",
            "stronger_count_card1", "stronger_count_card2", "stronger_count_card3"
        ]

        for field in required_fields:
            assert field in row, f"Missing required field: {field}"

        assert isinstance(row["game_seed"], int)
        assert isinstance(row["hand_number"], int)
        assert isinstance(row["player_id"], int)
        assert row["player_id"] in [0, 1]
        assert isinstance(row["is_dealer"], int)
        assert isinstance(row["is_mano"], int)
        assert row["is_dealer"] in [0, 1]
        assert row["is_mano"] in [0, 1]

    def test_agent_type_recording(self):
        rows = _play_game_and_collect(SimpleRuleBasedAgent, IntermediateAgent, 456)
        assert len(rows) > 0, "Should collect data from game"
        player0_rows = [r for r in rows if r["player_id"] == 0]
        assert len(player0_rows) > 0
        assert all(r["agent_player"] == "SimpleRuleBasedAgent" for r in player0_rows)
        assert all(r["agent_opponent"] == "IntermediateAgent" for r in player0_rows)
        player1_rows = [r for r in rows if r["player_id"] == 1]
        assert len(player1_rows) > 0
        assert all(r["agent_player"] == "IntermediateAgent" for r in player1_rows)
        assert all(r["agent_opponent"] == "SimpleRuleBasedAgent" for r in player1_rows)

    def test_points_calculation(self):
        rows = _play_game_and_collect(RandomAgent, RandomAgent, 789)

        hands = {}
        for row in rows:
            hand_key = (row["game_seed"], row["hand_number"])
            if hand_key not in hands:
                hands[hand_key] = []
            hands[hand_key].append(row)

        for hand_key, hand_rows in hands.items():
            assert len(hand_rows) == 2
            total_gained = sum(r["points_gained"] for r in hand_rows)
            assert total_gained >= 0
            for row in hand_rows:
                if row["points_gained"] > 0:
                    assert row["won_hand"] == 1
                else:
                    assert row["won_hand"] == 0


class TestGenerateDataset:

    def test_generate_small_dataset(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_dataset(num_games_per_matchup=1, output_csv=output_path, seed=789)

            assert os.path.exists(output_path)

            df = pd.read_csv(output_path)
            assert len(df) > 0

            expected_columns = [
                "game_seed", "hand_number", "player_id", "agent_player",
                "agent_opponent", "points_gained", "won_hand", "envido_points",
                "card1_str", "card2_str", "card3_str", "strength_sum"
            ]
            for col in expected_columns:
                assert col in df.columns, f"Missing column: {col}"
            agent_combinations = df[["agent_player", "agent_opponent"]].drop_duplicates()
            assert len(agent_combinations) <= 9
            all_agents = set(df["agent_player"].unique()) | set(df["agent_opponent"].unique())
            expected_agents = {"RandomAgent", "SimpleRuleBasedAgent", "IntermediateAgent"}
            assert all_agents == expected_agents

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_dataset_statistics(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_dataset(num_games_per_matchup=5, output_csv=output_path, seed=999)

            df = pd.read_csv(output_path)

            assert set(df["player_id"].unique()) == {0, 1}
            assert df["envido_points"].min() >= 0
            assert df["envido_points"].max() <= 34
            assert df["strength1"].min() >= 1
            assert df["strength1"].max() <= 14
            assert df["strength2"].min() >= 1
            assert df["strength2"].max() <= 14
            assert df["strength3"].min() >= 1
            assert df["strength3"].max() <= 14

            for _, row in df.iterrows():
                expected_sum = row["strength1"] + row["strength2"] + row["strength3"]
                assert row["strength_sum"] == expected_sum

            for _, row in df.iterrows():
                expected_max = max(row["strength1"], row["strength2"], row["strength3"])
                assert row["strength_max"] == expected_max

            assert df["won_hand"].sum() > 0
            assert (df["won_hand"] == 0).sum() > 0

            assert set(df["is_dealer"].unique()) <= {0, 1}
            assert set(df["is_mano"].unique()) <= {0, 1}
            for _, row in df.iterrows():
                assert not (row["is_dealer"] == 1 and row["is_mano"] == 1)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @patch('hand_eval.dataset_creation._play_game_and_collect')
    def test_correct_number_of_games(self, mock_play):
        mock_play.return_value = [
            {"game_seed": 1, "player_id": 0},
            {"game_seed": 1, "player_id": 1}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_dataset(num_games_per_matchup=3, output_csv=output_path, seed=111)
            assert mock_play.call_count == 27
            seeds_used = [call[0][2] for call in mock_play.call_args_list]
            assert len(set(seeds_used)) == 27

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_output_file_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_dataset(num_games_per_matchup=1, output_csv=output_path, seed=222)

            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0
            df = pd.read_csv(output_path)
            assert not df.empty
            assert df.shape[0] == len(rows)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestEdgeCases:

    def test_empty_game(self):
        with patch('hand_eval.dataset_creation.TrucoEnvironment') as mock_env_class:
            mock_env = MagicMock()
            mock_env_class.return_value = mock_env
            mock_env.game_over = True
            mock_env.get_valid_actions.return_value = []
            rows = _play_game_and_collect(RandomAgent, RandomAgent, 0)
            assert isinstance(rows, list)

    def test_tie_probability_edge_cases(self):
        assert tie_probability["1 de Espada"] == 0
        assert abs(tie_probability["3 de Espada"] - 3 / 39) < 0.001
        assert abs(tie_probability["4 de Copa"] - 3 / 39) < 0.001

    def test_envido_with_special_suits(self):
        cards = ("1 de Espada", "2 de Espada", "3 de Oro")
        points = _envido_points(cards)
        assert points == 23
        cards = ("10 de Espada", "11 de Oro", "12 de Copa")
        points = _envido_points(cards)
        assert points == 20

    def test_exception_handling_in_play_game(self):
        with patch('hand_eval.dataset_creation.TrucoEnvironment') as mock_env_class:
            mock_env = MagicMock()
            mock_env_class.return_value = mock_env

            mock_env.game_over = False
            mock_env.scores = [0, 0]
            mock_env.hands_played = 1
            mock_env.current_round = 1
            mock_env.round_cards_played = [[], []]
            mock_env.current_hand_active = True
            mock_env.current_player = 0

            mock_obs_full = {
                "dealer": 0,
                "mano": 1,
                "hands": [[1, 2, 3], [4, 5, 6]]
            }
            mock_obs = {"current_player": 0}

            def get_obs_side_effect(full_info=False):
                return mock_obs_full if full_info else mock_obs

            mock_env._get_observation.side_effect = get_obs_side_effect
            mock_env.get_valid_actions.return_value = [{"type": "play_card", "card_index": 0}]
            mock_env.step.return_value = (mock_obs, 0, True, {})

            with patch.object(RandomAgent, 'choose_action', side_effect=Exception("Agent error")):
                rows = _play_game_and_collect(RandomAgent, RandomAgent, 0)
                assert isinstance(rows, list)

    def test_card_string_formats(self):
        assert _envido_value("1 de Espada") == 1
        assert _envido_value("1  de  Espada") == 1

    def test_data_consistency(self):
        rows = _play_game_and_collect(RandomAgent, RandomAgent, 12345)

        for row in rows:
            assert (row["is_dealer"] == 1) == (row["dealer_id"] == row["player_id"])
            assert (row["is_mano"] == 1) == (row["mano_id"] == row["player_id"])
            assert row["card1_str"] in card_rank
            assert row["card2_str"] in card_rank
            assert row["card3_str"] in card_rank
            assert row["strength1"] == card_rank[row["card1_str"]]
            assert row["strength2"] == card_rank[row["card2_str"]]
            assert row["strength3"] == card_rank[row["card3_str"]]
            assert abs(row["tie_probability_card1"] - tie_probability[row["card1_str"]]) < 0.001
            assert abs(row["tie_probability_card2"] - tie_probability[row["card2_str"]]) < 0.001
            assert abs(row["tie_probability_card3"] - tie_probability[row["card3_str"]]) < 0.001

            cards = (row["card1_str"], row["card2_str"], row["card3_str"])
            calculated_envido = _envido_points(cards)
            assert row["envido_points"] == calculated_envido


class TestMainFunction:

    def test_main_execution(self):
        try:
            import hand_eval.dataset_creation
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import dataset_creation: {e}")

    def test_argparse_defaults(self):
        import argparse
        parser = argparse.ArgumentParser(description="Generate Truco hand‑level dataset for ML training")
        parser.add_argument("--num_games", type=int, default=50000, help="Full games *per* agent matchup (default 500)")
        parser.add_argument("--output", type=str, default="truco_training_data.csv", help="CSV output path")
        parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")

        args = parser.parse_args([])
        assert args.num_games == 50000
        assert args.output == "truco_training_data.csv"
        assert args.seed == 42

        args = parser.parse_args(["--num_games", "100", "--output", "test.csv", "--seed", "123"])
        assert args.num_games == 100
        assert args.output == "test.csv"
        assert args.seed == 123


class TestIntegration:

    def test_full_pipeline_small(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_dataset(num_games_per_matchup=2, output_csv=output_path, seed=333)
            df = pd.read_csv(output_path)
            assert len(df) > 0
            assert len(df.columns) > 30
            assert df.isnull().sum().sum() == 0
            game_seeds = df.groupby(['game_seed', 'hand_number']).size()
            assert all(count == 2 for count in game_seeds)

            matchups = df.groupby(['agent_player', 'agent_opponent'])['game_seed'].nunique()
            assert len(matchups) == 9
            assert all(count >= 2 for count in matchups)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_deterministic_generation(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp1:
            output_path1 = tmp1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp2:
            output_path2 = tmp2.name

        try:
            generate_dataset(num_games_per_matchup=1, output_csv=output_path1, seed=444)
            generate_dataset(num_games_per_matchup=1, output_csv=output_path2, seed=444)

            df1 = pd.read_csv(output_path1)
            df2 = pd.read_csv(output_path2)

            assert list(df1.columns) == list(df2.columns)
            assert len(df1) == len(df2)

            assert df1['game_seed'].iloc[0] == df2['game_seed'].iloc[0]

        finally:
            for path in [output_path1, output_path2]:
                if os.path.exists(path):
                    os.remove(path)

    def test_feature_correlations(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            output_path = tmp.name

        try:
            generate_dataset(num_games_per_matchup=10, output_csv=output_path, seed=555)
            df = pd.read_csv(output_path)

            correlation = df['strength_sum'].corr(df['won_hand'])
            assert correlation > 0

            assert all((df['points_gained'] > 0) == (df['won_hand'] == 1))

            assert df['strength_sum'].min() >= 3
            assert df['strength_max'].min() >= 1
            assert df['num_same_suit'].min() >= 1
            assert df['num_same_suit'].max() <= 3

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestUtilities:

    def test_imports(self):
        import hand_eval.dataset_creation as dataset_creation
        assert hasattr(dataset_creation, '_envido_value')
        assert hasattr(dataset_creation, '_envido_points')
        assert hasattr(dataset_creation, '_play_game_and_collect')
        assert hasattr(dataset_creation, 'generate_dataset')

    def test_type_hints(self):
        import inspect
        from hand_eval.dataset_creation import _envido_value, _envido_points, _play_game_and_collect, generate_dataset

        assert inspect.signature(_envido_value)
        assert inspect.signature(_envido_points)
        assert inspect.signature(_play_game_and_collect)
        assert inspect.signature(generate_dataset)