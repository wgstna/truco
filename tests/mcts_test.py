import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import copy
import math
from collections import defaultdict

import sys

sys.modules['truco_setup'] = MagicMock()
sys.modules['truco_setup.truco_environment'] = MagicMock()
sys.modules['truco_setup.basic_agents'] = MagicMock()

from game_sim.mcts import (
    SimplifiedTrucoState,
    MCTSNode,
    DeterminizedMCTS,
    ISMCTSNode,
    ISMCTS,
    TrucoStateAdapter
)


class TestSimplifiedTrucoState:

    def test_initialization(self):
        state = SimplifiedTrucoState()
        assert len(state.deck) == 40
        assert state.hands == [[], []]
        assert state.played_cards == [[], [], []]
        assert state.current_round == 0
        assert state.round_winners == [-1, -1, -1]
        assert state.scores == [0, 0]
        assert state.truco_state == 'NONE'
        assert state.current_bet == 1
        assert state.current_player == 0
        assert state.game_over is False
        assert state.pending_truco is False

    def test_create_deck(self):
        state = SimplifiedTrucoState()
        deck = state._create_deck()

        assert len(deck) == 40

        suits = set(card[1] for card in deck)
        assert suits == {'espadas', 'bastos', 'oros', 'copas'}

        numbers = set(card[0] for card in deck)
        expected_numbers = {'1', '2', '3', '4', '5', '6', '7', '10', '11', '12'}
        assert numbers == expected_numbers

    def test_deal_cards(self):
        state = SimplifiedTrucoState()
        state.deal_cards()

        assert len(state.hands[0]) == 3
        assert len(state.hands[1]) == 3

        all_dealt = state.hands[0] + state.hands[1]
        assert len(all_dealt) == len(set(all_dealt))

    def test_get_card_value(self):
        state = SimplifiedTrucoState()

        assert state.get_card_value(('1', 'espadas')) == 14
        assert state.get_card_value(('1', 'bastos')) == 13
        assert state.get_card_value(('7', 'espadas')) == 12
        assert state.get_card_value(('7', 'oros')) == 11

        assert state.get_card_value(('3', 'copas')) == 10
        assert state.get_card_value(('2', 'oros')) == 9
        assert state.get_card_value(('1', 'oros')) == 8
        assert state.get_card_value(('12', 'copas')) == 7
        assert state.get_card_value(('11', 'bastos')) == 6
        assert state.get_card_value(('10', 'espadas')) == 5
        assert state.get_card_value(('7', 'copas')) == 4
        assert state.get_card_value(('6', 'oros')) == 3
        assert state.get_card_value(('5', 'bastos')) == 2
        assert state.get_card_value(('4', 'espadas')) == 1

    def test_clone(self):
        state = SimplifiedTrucoState()
        state.deal_cards()
        state.scores = [10, 15]
        state.current_player = 1

        cloned = state.clone()

        assert cloned.scores == state.scores
        assert cloned.current_player == state.current_player
        assert cloned.hands == state.hands

        cloned.scores[0] = 20
        assert state.scores[0] == 10

        cloned.hands[0].append(('test', 'card'))
        assert ('test', 'card') not in state.hands[0]

    def test_get_legal_actions_normal(self):
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas'), ('3', 'copas'), ('7', 'oros')]

        actions = state.get_legal_actions(0)

        assert ('PLAY_CARD', 0) in actions
        assert ('PLAY_CARD', 1) in actions
        assert ('PLAY_CARD', 2) in actions
        assert ('TRUCO', None) in actions

    def test_get_legal_actions_pending_truco(self):
        state = SimplifiedTrucoState()
        state.pending_truco = True
        state.truco_state = 'TRUCO'

        actions = state.get_legal_actions(0)

        assert ('ACCEPT_TRUCO', None) in actions
        assert ('REJECT_TRUCO', None) in actions
        assert ('RETRUCO', None) in actions

    def test_get_legal_actions_game_over(self):
        state = SimplifiedTrucoState()
        state.game_over = True

        actions = state.get_legal_actions(0)
        assert actions == []

    def test_apply_action_play_card(self):
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas'), ('3', 'copas'), ('7', 'oros')]
        state.hands[1] = [('2', 'bastos'), ('4', 'copas'), ('10', 'oros')]

        state.apply_action(('PLAY_CARD', 0))

        assert len(state.hands[0]) == 2
        assert len(state.played_cards[0]) == 1
        assert state.played_cards[0][0][0] == 0
        assert state.played_cards[0][0][1] == ('1', 'espadas')
        assert state.current_player == 1

    def test_apply_action_truco(self):
        state = SimplifiedTrucoState()

        state.apply_action(('TRUCO', None))

        assert state.pending_truco is True
        assert state.truco_state == 'TRUCO'
        assert state.current_bet == 3
        assert state.current_player == 1

    def test_apply_action_accept_truco(self):
        state = SimplifiedTrucoState()
        state.pending_truco = True
        state.current_bet = 3

        state.apply_action(('ACCEPT_TRUCO', None))

        assert state.pending_truco is False
        assert state.current_bet == 3

    def test_apply_action_reject_truco(self):
        state = SimplifiedTrucoState()
        state.pending_truco = True
        state.current_player = 1

        state.apply_action(('REJECT_TRUCO', None))

        assert state.game_over is True
        assert state.scores[0] == 1

    def test_evaluate_round(self):
        state = SimplifiedTrucoState()

        state.played_cards[0] = [
            (0, ('1', 'espadas')),
            (1, ('3', 'copas'))
        ]

        state._evaluate_round()

        assert state.round_winners[0] == 0
        assert state.current_round == 1

    def test_evaluate_round_tie(self):
        state = SimplifiedTrucoState()

        state.played_cards[0] = [
            (0, ('3', 'espadas')),
            (1, ('3', 'copas'))
        ]

        state._evaluate_round()

        assert state.round_winners[0] == -1

    def test_check_game_end(self):
        state = SimplifiedTrucoState()

        state.round_winners = [0, 1, -1]
        assert state._check_game_end() is False

        state.round_winners = [0, 0, -1]
        assert state._check_game_end() is True

        state.round_winners = [1, -1, 1]
        assert state._check_game_end() is True

        state.game_over = True
        assert state._check_game_end() is True

    def test_get_game_winner(self):
        state = SimplifiedTrucoState()

        state.round_winners = [0, 0, 1]
        assert state._get_game_winner() == 0

        state.round_winners = [1, 0, 1]
        assert state._get_game_winner() == 1

        state.round_winners = [0, 1, -1]
        assert state._get_game_winner() == -1

    def test_is_terminal(self):
        state = SimplifiedTrucoState()
        state.deal_cards()
        assert state.is_terminal() is False

        state.game_over = True
        assert state.is_terminal() is True

        state = SimplifiedTrucoState()
        state.deal_cards()
        state.scores = [30, 25]
        assert state.is_terminal() is True

        state = SimplifiedTrucoState()
        state.deal_cards()
        state.current_round = 3
        assert state.is_terminal() is True

        state = SimplifiedTrucoState()
        state.hands = [[], []]
        assert state.is_terminal() is True

    def test_get_reward(self):
        state = SimplifiedTrucoState()

        assert state.get_reward(0) == 0.5
        assert state.get_reward(1) == 0.5

        state.game_over = True
        state.scores = [15, 10]
        assert state.get_reward(0) == 1.0
        assert state.get_reward(1) == 0.0

        state.scores = [10, 15]
        assert state.get_reward(0) == 0.0
        assert state.get_reward(1) == 1.0


class TestMCTSNode:

    def test_initialization(self):
        state = SimplifiedTrucoState()
        node = MCTSNode(state, parent=None, action=None, player=0)

        assert node.state == state
        assert node.parent is None
        assert node.action is None
        assert node.player == 0
        assert node.children == []
        assert node.visits == 0
        assert node.total_reward == 0.0
        assert node.untried_actions is None
        assert node.is_terminal is False

    def test_add_child(self):
        parent_state = SimplifiedTrucoState()
        parent = MCTSNode(parent_state)

        child_state = SimplifiedTrucoState()
        child_state.current_player = 1

        child = parent.add_child(child_state, ('PLAY_CARD', 0), 1)

        assert child.parent == parent
        assert child.state == child_state
        assert child.action == ('PLAY_CARD', 0)
        assert child.player == 1
        assert child in parent.children

    def test_update(self):
        node = MCTSNode(SimplifiedTrucoState())

        node.update(0.8)
        assert node.visits == 1
        assert node.total_reward == 0.8

        node.update(0.6)
        assert node.visits == 2
        assert node.total_reward == 1.4

    def test_uct_value_unvisited(self):
        node = MCTSNode(SimplifiedTrucoState())

        assert node.uct_value() == float('inf')

    def test_uct_value_no_parent(self):
        node = MCTSNode(SimplifiedTrucoState())
        node.visits = 5
        node.total_reward = 3.0

        assert node.uct_value() == float('inf')

    def test_uct_value_normal(self):
        parent = MCTSNode(SimplifiedTrucoState())
        parent.visits = 100

        child = MCTSNode(SimplifiedTrucoState(), parent=parent)
        child.visits = 10
        child.total_reward = 6.0

        c = 1.414
        exploitation = 0.6
        exploration = c * math.sqrt(2 * math.log(100) / 10)
        expected = exploitation + exploration

        assert abs(child.uct_value(c) - expected) < 0.0001

    def test_best_child(self):
        parent = MCTSNode(SimplifiedTrucoState())
        parent.visits = 100

        child1 = parent.add_child(SimplifiedTrucoState(), ('PLAY_CARD', 0), 0)
        child1.visits = 10
        child1.total_reward = 5.0

        child2 = parent.add_child(SimplifiedTrucoState(), ('PLAY_CARD', 1), 0)
        child2.visits = 20
        child2.total_reward = 15.0

        child3 = parent.add_child(SimplifiedTrucoState(), ('PLAY_CARD', 2), 0)
        child3.visits = 5
        child3.total_reward = 4.5

        best = parent.best_child()
        assert best in [child1, child2, child3]

    def test_is_fully_expanded(self):
        node = MCTSNode(SimplifiedTrucoState())

        assert node.is_fully_expanded() is False

        node.untried_actions = [('PLAY_CARD', 0), ('PLAY_CARD', 1)]
        assert node.is_fully_expanded() is False

        node.untried_actions = []
        assert node.is_fully_expanded() is True


class TestDeterminizedMCTS:

    def test_initialization(self):
        mcts = DeterminizedMCTS(total_simulations=100, exploration_constant=1.5)

        assert mcts.total_simulations_budget == 100
        assert mcts.exploration_constant == 1.5
        assert isinstance(mcts.action_statistics, defaultdict)
        assert mcts.total_simulations == 0

    def test_determinize(self):
        state = SimplifiedTrucoState()
        state.deal_cards()

        mcts = DeterminizedMCTS()

        det_state = mcts._determinize(state, 0)

        assert det_state.hands[0] == state.hands[0]

        assert len(det_state.hands[1]) == len(state.hands[1])

    def test_expand(self):
        mcts = DeterminizedMCTS()
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas'), ('3', 'copas')]

        root = MCTSNode(state)
        root.untried_actions = state.get_legal_actions(0)

        child = mcts._expand(root)

        assert child.parent == root
        assert child in root.children
        assert child.action in state.get_legal_actions(0)
        assert len(root.untried_actions) == len(state.get_legal_actions(0)) - 1

    def test_simulate(self):
        mcts = DeterminizedMCTS()
        state = SimplifiedTrucoState()
        state.deal_cards()

        node = MCTSNode(state, player=0)

        reward = mcts._simulate(node)

        assert 0 <= reward <= 1

    def test_backpropagate(self):
        mcts = DeterminizedMCTS()

        root = MCTSNode(SimplifiedTrucoState(), player=0)
        child = root.add_child(SimplifiedTrucoState(), ('PLAY_CARD', 0), player=1)
        grandchild = child.add_child(SimplifiedTrucoState(), ('PLAY_CARD', 1), player=0)

        mcts._backpropagate(grandchild, 1.0)

        assert grandchild.visits == 1
        assert grandchild.total_reward == 1.0

        assert child.visits == 1
        assert child.total_reward == 0.0

        assert root.visits == 1
        assert root.total_reward == 1.0

    def test_get_action_simple(self):
        mcts = DeterminizedMCTS(total_simulations=10)
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas')]
        state.hands[1] = [('3', 'copas')]

        action = mcts.get_action(state, 0)

        assert action in state.get_legal_actions(0)

    @patch('random.choice')
    def test_get_action_no_simulations(self, mock_random):
        mcts = DeterminizedMCTS(total_simulations=0)
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas'), ('3', 'copas')]

        mock_random.return_value = ('PLAY_CARD', 0)

        action = mcts.get_action(state, 0)

        mock_random.assert_called()


class TestISMCTSNode:

    def test_initialization(self):
        info_set = "test_info_set"
        node = ISMCTSNode(info_set)

        assert node.information_set == info_set
        assert node.parent is None
        assert node.action is None
        assert node.player == 0
        assert node.children == {}
        assert node.visits == 0
        assert node.total_reward == 0.0
        assert isinstance(node.available_actions, set)

    def test_add_child_new(self):
        parent = ISMCTSNode("parent_info")
        action = ('PLAY_CARD', 0)
        child_info = "child_info"

        child = parent.add_child(action, child_info, player=1)

        assert child.parent == parent
        assert child.action == action
        assert child.information_set == child_info
        assert child.player == 1
        assert parent.children[action] == child

    def test_add_child_existing(self):
        parent = ISMCTSNode("parent_info")
        action = ('PLAY_CARD', 0)

        child1 = parent.add_child(action, "child_info", player=1)

        child2 = parent.add_child(action, "child_info_2", player=1)

        assert child1 == child2
        assert len(parent.children) == 1


class TestISMCTS:

    def test_initialization(self):
        ismcts = ISMCTS(total_simulations=200, exploration_constant=2.0)

        assert ismcts.total_simulations_budget == 200
        assert ismcts.exploration_constant == 2.0
        assert isinstance(ismcts.action_statistics, defaultdict)
        assert ismcts.total_determinizations == 0
        assert ismcts.total_simulations == 0

    def test_get_information_set(self):
        ismcts = ISMCTS()
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas'), ('3', 'copas')]
        state.played_cards = [[]]
        state.scores = [10, 15]
        state.truco_state = 'TRUCO'
        state.round_winners = [0, -1, -1]

        info_set = ismcts._get_information_set(state, 0)

        assert isinstance(info_set, str)

        assert "('1', 'espadas')" in info_set
        assert "('3', 'copas')" in info_set
        assert "[10, 15]" in info_set
        assert "'TRUCO'" in info_set

    def test_determinize(self):
        state = SimplifiedTrucoState()
        state.deal_cards()

        ismcts = ISMCTS()
        det_state = ismcts._determinize(state, 0)

        assert det_state.hands[0] == state.hands[0]
        assert len(det_state.hands[1]) == len(state.hands[1])

    def test_get_action_simple(self):
        ismcts = ISMCTS(total_simulations=10)
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas')]
        state.hands[1] = [('3', 'copas')]

        action = ismcts.get_action(state, 0)

        assert action in state.get_legal_actions(0)

    @patch('random.choice')
    def test_get_action_no_legal_actions(self, mock_random):
        ismcts = ISMCTS()
        state = SimplifiedTrucoState()
        state.game_over = True

        mock_random.return_value = ('PASS', None)

        with patch.object(state, 'get_legal_actions', return_value=[]):
            action = ismcts.get_action(state, 0)

        mock_random.assert_called()


class TestTrucoStateAdapter:

    def test_initialization(self):
        env_state = {
            'current_player': 1,
            'game_over': False,
            'scores': [10, 15]
        }

        adapter = TrucoStateAdapter(env_state)

        assert adapter.current_player == 1
        assert adapter.game_over is False
        assert adapter._full_state == env_state

    def test_clone(self):
        env_state = {
            'current_player': 0,
            'game_over': False,
            'scores': [5, 10],
            'hands': [['card1'], ['card2']]
        }

        adapter = TrucoStateAdapter(env_state)
        cloned = adapter.clone()

        assert cloned.current_player == adapter.current_player
        assert cloned.game_over == adapter.game_over
        assert cloned._full_state == adapter._full_state

        cloned._full_state['scores'][0] = 20
        assert adapter._full_state['scores'][0] == 5

    def test_get_legal_actions_wrong_player(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})

        actions = adapter.get_legal_actions(1)
        assert actions == []

    def test_get_legal_actions_with_valid_actions(self):
        env_state = {
            'current_player': 0,
            'game_over': False,
            'valid_actions': [
                {'type': 'play_card', 'card_index': 0},
                {'type': 'play_card', 'card_index': 1},
                {'type': 'call_truco', 'which': 'Truco'}
            ]
        }

        adapter = TrucoStateAdapter(env_state)
        actions = adapter.get_legal_actions(0)

        assert ('PLAY_CARD', 0) in actions
        assert ('PLAY_CARD', 1) in actions
        assert ('TRUCO', None) in actions

    def test_get_legal_actions_game_over(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': True})

        actions = adapter.get_legal_actions(0)
        assert actions == []

    def test_apply_action(self):
        env_state = {
            'current_player': 0,
            'game_over': False,
            'scores': [28, 29]
        }

        adapter = TrucoStateAdapter(env_state)
        adapter.apply_action(('PLAY_CARD', 0))

        assert adapter.current_player == 1
        assert adapter._full_state['current_player'] == 1

    def test_apply_action_triggers_game_over(self):
        env_state = {
            'current_player': 0,
            'game_over': False,
            'scores': [30, 25]
        }

        adapter = TrucoStateAdapter(env_state)
        adapter.apply_action(('PLAY_CARD', 0))

        assert adapter.game_over is True
        assert adapter._full_state['game_over'] is True

    def test_is_terminal(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})
        assert adapter.is_terminal() is False

        adapter.game_over = True
        assert adapter.is_terminal() is True

    def test_get_reward_not_terminal(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})

        assert adapter.get_reward(0) == 0.5
        assert adapter.get_reward(1) == 0.5

    def test_get_reward_terminal(self):
        env_state = {
            'current_player': 0,
            'game_over': True,
            'scores': [30, 25]
        }

        adapter = TrucoStateAdapter(env_state)

        assert adapter.get_reward(0) == 1.0
        assert adapter.get_reward(1) == 0.0

    def test_deck_property(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})
        deck = adapter.deck

        assert len(deck) == 40
        assert isinstance(deck, list)
        assert all(isinstance(card, tuple) for card in deck)

    def test_hands_property(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False, 'hands': [['card1'], ['card2']]})
        assert adapter.hands == [['card1'], ['card2']]

        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})
        assert adapter.hands == [[], []]

    def test_played_cards_property(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False, 'round_cards_played': [['c1'], ['c2'], ['c3']]})
        assert adapter.played_cards == [['c1'], ['c2'], ['c3']]

        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})
        assert adapter.played_cards == [[], [], []]

    def test_scores_property(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False, 'scores': [15, 20]})
        assert adapter.scores == [15, 20]

        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})
        assert adapter.scores == [0, 0]

    def test_truco_state_property(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False, 'truco_stage': 'Truco'})
        assert adapter.truco_state == 'Truco'

        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False, 'truco_stage': None})
        assert adapter.truco_state == 'NONE'

        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})
        assert adapter.truco_state == 'NONE'

    def test_round_winners_property(self):
        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False, 'round_wins': [0, 1, 0]})
        assert adapter.round_winners == [0, 1, 0]

        adapter = TrucoStateAdapter({'current_player': 0, 'game_over': False})
        assert adapter.round_winners == [-1, -1, -1]


class TestIntegration:

    def test_determinized_mcts_full_game(self):
        state = SimplifiedTrucoState()
        state.deal_cards()

        mcts = DeterminizedMCTS(total_simulations=50)

        moves_played = 0
        max_moves = 10

        while not state.is_terminal() and moves_played < max_moves:
            player = state.current_player
            action = mcts.get_action(state, player)

            assert action in state.get_legal_actions(player)

            state.apply_action(action)
            moves_played += 1

        assert moves_played > 0

    def test_ismcts_full_game(self):
        state = SimplifiedTrucoState()
        state.deal_cards()

        ismcts = ISMCTS(total_simulations=50)

        moves_played = 0
        max_moves = 10

        while not state.is_terminal() and moves_played < max_moves:
            player = state.current_player
            action = ismcts.get_action(state, player)

            assert action in state.get_legal_actions(player)

            state.apply_action(action)
            moves_played += 1

        assert moves_played > 0

    def test_adapter_with_mcts(self):
        env_state = {
            'current_player': 0,
            'game_over': False,
            'scores': [0, 0],
            'valid_actions': [
                {'type': 'play_card', 'card_index': 0},
                {'type': 'play_card', 'card_index': 1}
            ]
        }

        adapter = TrucoStateAdapter(env_state)

        actions = adapter.get_legal_actions(0)
        assert len(actions) > 0

        adapter.apply_action(actions[0])
        assert adapter.current_player != 0


class TestHelperFunctions:

    def test_imports_handling(self):
        assert True

    def test_action_statistics_tracking(self):
        mcts = DeterminizedMCTS(total_simulations=20)
        state = SimplifiedTrucoState()
        state.hands[0] = [('1', 'espadas'), ('3', 'copas')]
        state.hands[1] = [('2', 'bastos'), ('4', 'copas')]

        action = mcts.get_action(state, 0)

        assert len(mcts.action_statistics) > 0
        assert mcts.total_simulations > 0

    def test_empty_hands_edge_case(self):
        state = SimplifiedTrucoState()
        state.hands = [[], []]

        mcts = DeterminizedMCTS(total_simulations=10)

        actions = state.get_legal_actions(0)
        if actions:
            action = mcts.get_action(state, 0)
            assert action in actions