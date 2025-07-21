import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import math
import copy
import time
from typing import List, Tuple, Dict, Any, Optional
import json

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')

try:
    from truco_setup.truco_environment import TrucoEnvironment
    from truco_setup.basic_agents import Agent

    USE_REAL_ENV = True
except ImportError:
    print("Warning: Could not import TrucoEnvironment, using simplified version")
    USE_REAL_ENV = False



class SimplifiedTrucoState:
    def __init__(self):
        self.deck = self._create_deck()
        self.hands = [[], []]
        self.played_cards = [[], [], []]
        self.current_round = 0
        self.round_winners = [-1, -1, -1]
        self.scores = [0, 0]
        self.truco_state = 'NONE'
        self.current_bet = 1
        self.current_player = 0
        self.game_over = False
        self.pending_truco = False

    def _create_deck(self):
        suits = ['espadas', 'bastos', 'oros', 'copas']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '10', '11', '12']
        return [(n, s) for s in suits for n in numbers]

    def deal_cards(self):
        shuffled = self.deck.copy()
        random.shuffle(shuffled)
        self.hands[0] = shuffled[:3]
        self.hands[1] = shuffled[3:6]

    def get_card_value(self, card):
        num, suit = card
        special_values = {
            ('1', 'espadas'): 14,
            ('1', 'bastos'): 13,
            ('7', 'espadas'): 12,
            ('7', 'oros'): 11
        }
        if (num, suit) in special_values:
            return special_values[(num, suit)]

        regular_values = {
            '3': 10, '2': 9, '1': 8, '12': 7, '11': 6,
            '10': 5, '7': 4, '6': 3, '5': 2, '4': 1
        }
        return regular_values.get(num, 0)

    def clone(self):
        return copy.deepcopy(self)

    def get_legal_actions(self, player):
        actions = []

        if self.game_over:
            return []

        if self.pending_truco:
            actions.extend([('ACCEPT_TRUCO', None), ('REJECT_TRUCO', None)])
            if self.truco_state == 'TRUCO':
                actions.append(('RETRUCO', None))
        else:
            if player < len(self.hands) and self.current_round < 3:
                for i in range(len(self.hands[player])):
                    actions.append(('PLAY_CARD', i))

            if self.truco_state == 'NONE' and len(actions) > 0:
                actions.append(('TRUCO', None))

        if not actions and not self.game_over:
            actions.append(('PASS', None))

        return actions

    def apply_action(self, action):
        action_type, action_data = action

        if action_type == 'PLAY_CARD':
            if action_data < len(self.hands[self.current_player]) and self.current_round < 3:
                card = self.hands[self.current_player].pop(action_data)
                self.played_cards[self.current_round].append((self.current_player, card))

                if len(self.played_cards[self.current_round]) == 2:
                    self._evaluate_round()
                else:
                    self.current_player = 1 - self.current_player

        elif action_type == 'TRUCO':
            self.pending_truco = True
            self.truco_state = 'TRUCO'
            self.current_bet = 3
            self.current_player = 1 - self.current_player

        elif action_type == 'ACCEPT_TRUCO':
            self.pending_truco = False
            self.current_player = 1 - self.current_player

        elif action_type == 'REJECT_TRUCO':
            self.game_over = True
            self.scores[1 - self.current_player] += 1

        elif action_type == 'PASS':
            self.current_player = 1 - self.current_player

    def _evaluate_round(self):
        if len(self.played_cards[self.current_round]) < 2:
            return

        p1_card = self.played_cards[self.current_round][0][1]
        p2_card = self.played_cards[self.current_round][1][1]

        p1_value = self.get_card_value(p1_card)
        p2_value = self.get_card_value(p2_card)

        if p1_value > p2_value:
            self.round_winners[self.current_round] = 0
        elif p2_value > p1_value:
            self.round_winners[self.current_round] = 1
        else:
            self.round_winners[self.current_round] = -1

        if self._check_game_end():
            self.game_over = True
            winner = self._get_game_winner()
            if winner >= 0:
                self.scores[winner] += self.current_bet
        else:
            self.current_round += 1
            self.current_player = 0

    def _check_game_end(self):
        if self.game_over:
            return True
        p1_wins = self.round_winners.count(0)
        p2_wins = self.round_winners.count(1)
        return p1_wins >= 2 or p2_wins >= 2 or self.current_round >= 2

    def _get_game_winner(self):
        p1_wins = self.round_winners.count(0)
        p2_wins = self.round_winners.count(1)
        if p1_wins > p2_wins:
            return 0
        elif p2_wins > p1_wins:
            return 1
        return -1

    def is_terminal(self):
        return self.game_over or any(s >= 30 for s in self.scores) or self.current_round > 2 or all(
            len(hand) == 0 for hand in self.hands)

    def get_reward(self, player):
        if self.game_over:
            if self.scores[player] > self.scores[1 - player]:
                return 1.0
            elif self.scores[player] < self.scores[1 - player]:
                return 0.0
        return 0.5


class MCTSNode:
    def __init__(self, state, parent=None, action=None, player=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.player = player
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None
        self.is_terminal = False

    def add_child(self, child_state, action, player):
        child = MCTSNode(child_state, parent=self, action=action, player=player)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward

    def uct_value(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return float('inf')

        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c=1.414):
        return max(self.children, key=lambda child: child.uct_value(c))

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0


class DeterminizedMCTS:
    def __init__(self, total_simulations=500, exploration_constant=1.414):
        self.total_simulations_budget = total_simulations
        self.exploration_constant = exploration_constant
        self.action_statistics = defaultdict(lambda: {'visits': 0, 'total_reward': 0})
        self.total_simulations = 0

    def get_action(self, state, player_perspective):
        self.action_statistics.clear()
        simulations_done = 0

        while simulations_done < self.total_simulations_budget:
            det_state = self._determinize(state, player_perspective)

            root = MCTSNode(det_state)
            root.untried_actions = det_state.get_legal_actions(det_state.current_player).copy()

            batch_size = min(20, self.total_simulations_budget - simulations_done)
            for _ in range(batch_size):
                node = self._select(root)
                reward = self._simulate(node)
                self._backpropagate(node, reward)
                simulations_done += 1
                self.total_simulations += 1

            if root.children:
                best_child = max(root.children, key=lambda child: child.visits)
                action = best_child.action
                self.action_statistics[str(action)]['visits'] += 1
                self.action_statistics[str(action)][
                    'total_reward'] += best_child.total_reward / best_child.visits if best_child.visits > 0 else 0

        if self.action_statistics:
            best_action_str = max(self.action_statistics.keys(),
                                  key=lambda a: self.action_statistics[a]['visits'])
            return eval(best_action_str)
        else:
            return random.choice(state.get_legal_actions(state.current_player))

    def _determinize(self, state, player_perspective):
        determinized = state.clone()
        opponent = 1 - player_perspective

        all_cards = determinized.deck.copy()

        visible_cards = determinized.hands[player_perspective].copy()
        for round_cards in determinized.played_cards:
            for player, card in round_cards:
                visible_cards.append(card)

        remaining_cards = [c for c in all_cards if c not in visible_cards]
        random.shuffle(remaining_cards)

        opponent_hand_size = len(determinized.hands[opponent])
        determinized.hands[opponent] = remaining_cards[:opponent_hand_size]

        return determinized

    def _select(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                if node.children:
                    node = node.best_child(self.exploration_constant)
                else:
                    return node
        return node

    def _expand(self, node):
        if node.untried_actions:
            action = node.untried_actions.pop()
            new_state = node.state.clone()
            new_state.apply_action(action)
            child = node.add_child(new_state, action, new_state.current_player)

            child.untried_actions = new_state.get_legal_actions(new_state.current_player)
            child.is_terminal = new_state.is_terminal()

            return child
        return node

    def _simulate(self, node):
        state = node.state.clone()

        while not state.is_terminal():
            legal_actions = state.get_legal_actions(state.current_player)
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            state.apply_action(action)

        return state.get_reward(node.player)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            reward = 1 - reward
            node = node.parent


class ISMCTSNode:
    def __init__(self, information_set, parent=None, action=None, player=0):
        self.information_set = information_set
        self.parent = parent
        self.action = action
        self.player = player
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.available_actions = set()

    def add_child(self, action, information_set, player):
        if action not in self.children:
            child = ISMCTSNode(information_set, parent=self, action=action, player=player)
            self.children[action] = child
            return child
        return self.children[action]

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward

    def uct_value(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return float('inf')

        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class ISMCTS:
    def __init__(self, total_simulations=500, exploration_constant=1.414):
        self.total_simulations_budget = total_simulations
        self.exploration_constant = exploration_constant
        self.action_statistics = defaultdict(lambda: {'visits': 0, 'total_reward': 0})
        self.total_determinizations = 0
        self.total_simulations = 0

    def get_action(self, state, player_perspective):
        simulations_done = 0
        self.action_statistics.clear()

        root = ISMCTSNode(self._get_information_set(state, player_perspective))

        while simulations_done < self.total_simulations_budget:
            determinized_state = self._determinize(state, player_perspective)

            self._iterate(root, determinized_state, player_perspective)
            simulations_done += 1
            self.total_simulations += 1

        for action, child in root.children.items():
            self.action_statistics[action]['visits'] = child.visits
            self.action_statistics[action]['total_reward'] = child.total_reward

        if not self.action_statistics:
            return random.choice(state.get_legal_actions(player_perspective))

        best_action = max(self.action_statistics.keys(),
                          key=lambda a: self.action_statistics[a]['visits'])

        return best_action

    def _determinize(self, state, player_perspective):
        determinized = state.clone()

        opponent = 1 - player_perspective
        all_cards = determinized.deck.copy()

        visible_cards = determinized.hands[player_perspective].copy()
        for round_cards in determinized.played_cards:
            for player, card in round_cards:
                visible_cards.append(card)

        remaining_cards = [c for c in all_cards if c not in visible_cards]
        random.shuffle(remaining_cards)

        opponent_hand_size = len(determinized.hands[opponent])
        determinized.hands[opponent] = remaining_cards[:opponent_hand_size]

        return determinized

    def _get_information_set(self, state, player):
        info = {
            'hand': sorted(state.hands[player]),
            'played': state.played_cards,
            'scores': state.scores,
            'truco_state': state.truco_state,
            'round_winners': state.round_winners
        }
        return str(info)

    def _iterate(self, root, state, player):
        node = root
        current_state = state.clone()
        history = []

        if root.visits == 0:
            root.update(0.5)

        while not current_state.is_terminal():
            legal_actions = current_state.get_legal_actions(current_state.current_player)
            if not legal_actions:
                break

            node.available_actions.update(legal_actions)

            untried_actions = [a for a in legal_actions if a not in node.children]

            if untried_actions:
                action = random.choice(untried_actions)
            else:
                valid_actions = [a for a in legal_actions if a in node.children]
                if valid_actions:
                    action = max(valid_actions,
                                 key=lambda a: node.children[a].uct_value(self.exploration_constant))
                else:
                    action = random.choice(legal_actions)

            current_state.apply_action(action)
            info_set = self._get_information_set(current_state, current_state.current_player)
            node = node.add_child(action, info_set, current_state.current_player)

            history.append((node, action))

        reward = current_state.get_reward(player)

        for node, action in reversed(history):
            node.update(reward)
            if node.parent is None:
                self.action_statistics[action]['visits'] += 1
                self.action_statistics[action]['total_reward'] += reward
            reward = 1 - reward


class TrucoStateAdapter:
    def __init__(self, env_state):
        self.env = env_state
        self.current_player = env_state['current_player']
        self.game_over = env_state['game_over']

        self._full_state = env_state

    def clone(self):
        new_adapter = TrucoStateAdapter(copy.deepcopy(self._full_state))
        return new_adapter

    def get_legal_actions(self, player):
        if player != self.current_player:
            return []

        actions = []

        if 'valid_actions' in self._full_state:
            for action in self._full_state['valid_actions']:
                if action['type'] == 'play_card':
                    actions.append(('PLAY_CARD', action['card_index']))
                elif action['type'] == 'call_truco':
                    actions.append(('TRUCO', None))

        if not actions and not self.game_over:
            actions.append(('PASS', None))

        return actions

    def apply_action(self, action):
        action_type, action_data = action

        self.current_player = 1 - self.current_player
        self._full_state['current_player'] = self.current_player

        if 'scores' in self._full_state:
            if any(s >= 30 for s in self._full_state['scores']):
                self.game_over = True
                self._full_state['game_over'] = True

    def is_terminal(self):
        return self.game_over

    def get_reward(self, player):
        if not self.game_over:
            return 0.5

        if 'scores' in self._full_state:
            scores = self._full_state['scores']
            if scores[player] > scores[1 - player]:
                return 1.0
            elif scores[player] < scores[1 - player]:
                return 0.0
        return 0.5

    @property
    def deck(self):
        suits = ['espadas', 'bastos', 'oros', 'copas']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '10', '11', '12']
        return [(n, s) for s in suits for n in numbers]

    @property
    def hands(self):
        if 'hands' in self._full_state:
            return self._full_state['hands']
        return [[], []]

    @property
    def played_cards(self):
        if 'round_cards_played' in self._full_state:
            return self._full_state['round_cards_played']
        return [[], [], []]

    @property
    def scores(self):
        if 'scores' in self._full_state:
            return self._full_state['scores']
        return [0, 0]

    @property
    def truco_state(self):
        if 'truco_stage' in self._full_state:
            return self._full_state['truco_stage'] or 'NONE'
        return 'NONE'

    @property
    def round_winners(self):
        if 'round_wins' in self._full_state:
            return self._full_state['round_wins']
        return [-1, -1, -1]