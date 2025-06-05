from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

from truco_setup.cards import Deck
from truco_setup.utils import compute_envido_score, determine_card_round_winner
from truco_setup.basic_agents import RandomAgent, SimpleRuleBasedAgent, IntermediateAgent
from truco_setup.rules import (
    max_points,
    valid_envido_order,
    get_envido_points_on_accept,
    get_envido_points_on_reject,
    valid_truco_order,
    get_truco_points_on_accept,
    get_truco_points_on_reject
)


class TrucoEnvironment:

    def __init__(self,
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 use_encoded_cards: bool = False,
                 difficulty_level: int = 0,
                 enable_curriculum: bool = False,
                 reward_scale: float = 1.0):
        self.master_seed = seed
        self._rng = np.random.default_rng(self.master_seed)
        self.verbose = verbose
        self.use_encoded_cards = use_encoded_cards
        self.reward_scale = reward_scale

        self.difficulty_level = difficulty_level
        self.enable_curriculum = enable_curriculum
        self.curriculum_metrics = {
            "recent_win_rate": 0.5,
            "recent_avg_reward": 0.0,
            "games_at_current_level": 0
        }

        self._setup_card_encoding()

        self._init_game_state()

        self._init_performance_metrics()

        self.reward_components = defaultdict(lambda: [0.0, 0.0])

    def _setup_card_encoding(self):
        self.card_to_id = {}
        self.id_to_card = {}
        suits = ["Espada", "Basto", "Oro", "Copa"]
        for suit_idx, suit in enumerate(suits):
            for val in range(1, 13):
                if val in [8, 9]:
                    continue
                card_str = f"{val} de {suit}"
                card_id = val + (suit_idx * 12)
                self.card_to_id[str(card_str)] = card_id
                self.id_to_card[card_id] = card_str

        self.hidden_card_id = 0
        self.num_cards = len(self.card_to_id) + 1

        self.card_strength = {}
        special_cards = [
            ("1 de Espada", 14),
            ("1 de Basto", 13),
            ("7 de Espada", 12),
            ("7 de Oro", 11)
        ]

        for card, value in special_cards:
            self.card_strength[card] = value

        for suit in suits:
            self.card_strength[f"3 de {suit}"] = 10
            self.card_strength[f"2 de {suit}"] = 9
            if suit == "Copa" or suit == "Oro":
                self.card_strength[f"1 de {suit}"] = 8

            self.card_strength[f"12 de {suit}"] = 7
            self.card_strength[f"11 de {suit}"] = 6
            self.card_strength[f"10 de {suit}"] = 5

            if suit == "Copa" or suit == "Basto":
                self.card_strength[f"7 de {suit}"] = 4
            self.card_strength[f"6 de {suit}"] = 3
            self.card_strength[f"5 de {suit}"] = 2
            self.card_strength[f"4 de {suit}"] = 1

    def _init_game_state(self):
        self.scores = [0, 0]
        self.dealer = 0
        self.mano = 1
        self.max_points = max_points
        self.game_over = False
        self.deck = Deck(seed=self.master_seed)
        self.current_hand_active = False
        self.hands_played = 0
        self.start_of_hand_scores = [0, 0]

        self.partial_rewards = [0.0, 0.0]
        self.hand_partial_rewards = [0.0, 0.0]
        self.total_game_rewards = [0.0, 0.0]
        self.episode_rewards = [0.0, 0.0]

        self.action_history = []
        self.envido_history = []
        self.truco_history = []
        self.round_winner_history = []
        self.last_action = None
        self.hand_action_history = []

        self.rejection_counts = [
            {"envido": 0, "truco": 0},
            {"envido": 0, "truco": 0}
        ]
        self.total_envido_calls = [0, 0]
        self.total_truco_calls = [0, 0]
        self.envido_scores_history = [[], []]
        self.avg_envido_scores = [0.0, 0.0]

        self.opponent_hand_strength_model = [
            {"high_card_rate": 0.33, "samples": 0},
            {"high_card_rate": 0.33, "samples": 0}
        ]

        self.bluff_attempts = [0, 0]
        self.successful_bluffs = [0, 0]

        self.aggression_score = [0.5, 0.5]
        self.aggression_samples = [1, 1]

        self.envido_attempts = [0, 0]
        self.envido_wins = [0, 0]
        self.truco_attempts = [0, 0]
        self.truco_wins = [0, 0]
        self.total_actions = 0
        self.cards_played_optimally = [0, 0]
        self.total_cards_played = [0, 0]

        self.previous_position_score = [0, 0]

        self.envido_initiator = None
        self.truco_initiator = None

        self.recent_outcomes = deque(maxlen=20)

    def _init_performance_metrics(self):
        self.metrics = {
            "games_played": 0,
            "games_won": [0, 0],
            "win_rate": [0.0, 0.0],
            "avg_points_scored": [0.0, 0.0],
            "avg_reward_per_game": [0.0, 0.0],
            "avg_hands_per_game": 0.0,
            "envido_success_rate": [0.0, 0.0],
            "truco_success_rate": [0.0, 0.0],
            "bluff_success_rate": [0.0, 0.0],
            "optimal_card_play_rate": [0.0, 0.0],
        }

    def _add_partial_reward(self, player: int, reward: float, reason: str = ""):
        scaled_reward = reward * self.reward_scale
        self.partial_rewards[player] += scaled_reward
        self.hand_partial_rewards[player] += scaled_reward
        self.total_game_rewards[player] += scaled_reward

        self.reward_components[reason][player] += scaled_reward

        if self.verbose:
            logger.debug(f"PARTIAL REWARD: Player {player} gets {scaled_reward:.2f} for {reason}")
            logger.debug(f"Current hand partial rewards: P0={self.hand_partial_rewards[0]:.2f}, "
                         f"P1={self.hand_partial_rewards[1]:.2f}")
            logger.debug(f"Total game rewards: P0={self.total_game_rewards[0]:.2f}, "
                         f"P1={self.total_game_rewards[1]:.2f}")

    def _get_game_progress(self) -> float:
        return max(self.scores[0], self.scores[1]) / self.max_points

    def _encode_card(self, card: str) -> int:
        return self.card_to_id[str(card)]

    def _decode_card(self, card_id: int) -> str:
        if card_id == self.hidden_card_id:
            return "Hidden Card"
        return self.id_to_card[card_id]

    def _add_points(self, player: int, points: int):
        old_score = self.scores[player]
        self.scores[player] = min(self.scores[player] + points, self.max_points)
        if self.verbose:
            logger.debug(f"Awarding {points} points to player {player}. "
                         f"Score goes from {old_score} to {self.scores[player]}")

    def reset(self, episode_idx=None) -> Dict[str, Any]:
        self.previous_position_score = self.scores[:]

        self.scores = [0, 0]
        self.partial_rewards = [0.0, 0.0]
        self.hand_partial_rewards = [0.0, 0.0]
        self.total_game_rewards = [0.0, 0.0]
        self.episode_rewards = [0.0, 0.0]
        self.reward_components.clear()

        self.game_over = False
        self.hands_played = 0

        self.dealer = 0
        self.mano = 1

        self.action_history = []
        self.envido_history = []
        self.truco_history = []
        self.round_winner_history = []
        self.last_action = None

        self.envido_initiator = None
        self.truco_initiator = None

        sub_seed = int(self._rng.integers(2 ** 32))
        if episode_idx is not None:
            self.deck = Deck(seed=sub_seed + episode_idx)
        else:
            self.deck = Deck(seed=sub_seed)

        self._start_new_hand()

        return self._get_observation()

    def _start_new_hand(self):
        sub_seed = int(self._rng.integers(2 ** 32))
        self.deck = Deck(seed=sub_seed)

        self.current_hand_active = True
        self.hands_played += 1

        self.hand_partial_rewards = [0.0, 0.0]
        self.hand_action_history = []

        self.envido_initiator = None
        self.truco_initiator = None

        self.start_of_hand_scores = self.scores[:]

        self.deck.reset()
        self.hands = [
            self.deck.draw(3),
            self.deck.draw(3)
        ]

        self.original_hands = [
            self.hands[0][:],
            self.hands[1][:]
        ]

        self.current_round = 1
        self.round_wins = [0, 0]

        self.round_cards_played = [[], []]

        self.truco_stage: Optional[str] = None
        self.truco_pending: bool = False
        self.truco_caller: Optional[int] = None

        self.envido_stage: Optional[str] = None
        self.envido_pending: bool = False
        self.envido_caller: Optional[int] = None
        self.envido_finished: bool = False

        self.current_player = self.mano
        pie_player = 1 - self.mano

        if self.verbose:
            logger.info(f"\n######## Match {self.hands_played} ########")
            logger.info(f"Mano: Player {self.mano}")
            logger.info(f"Pie: Player {pie_player}")

            cards_p0 = ", ".join(str(c) for c in self.hands[0])
            cards_p1 = ", ".join(str(c) for c in self.hands[1])
            logger.info(f"Cards of player 0: {cards_p0}")
            logger.info(f"Cards of player 1: {cards_p1}")

            p0_envido = compute_envido_score(self.original_hands[0])
            p1_envido = compute_envido_score(self.original_hands[1])
            logger.info(f"Envido points of player 0: {p0_envido}")
            logger.info(f"Envido points of player 1: {p1_envido}")

            logger.info(f"Game scores: ({self.scores[0]}, {self.scores[1]})")
            logger.info("---------------")

    def _is_action_in_valid_set(self, action: Dict[str, Any],
                                valid_actions: List[Dict[str, Any]]) -> bool:
        for valid_action in valid_actions:
            if action["type"] == valid_action["type"]:
                if action["type"] == "play_card":
                    if action["card_index"] == valid_action["card_index"]:
                        return True
                elif action["type"] in ["call_envido", "call_truco", "raise_envido", "raise_truco"]:
                    if action.get("which") == valid_action.get("which"):
                        return True
                else:
                    return True
        return False

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.game_over:
            return self._get_observation(full_info=False), 0.0, True, {"error": "Game already over."}

        valid_actions = self.get_valid_actions()
        if not self._is_action_in_valid_set(action, valid_actions):
            return self._get_observation(full_info=False), -5.0, False, {"error": "Invalid action attempted"}

        current_p = self.current_player
        action_type = action["type"]
        if self.verbose:
            logger.debug(f"Player {current_p} action: {action}")

        self.action_history.append({"player": current_p, "action": action})
        self.hand_action_history.append({"player": current_p, "action": action})
        self.last_action = action
        self.total_actions += 1

        reward = self.partial_rewards[current_p]

        self.episode_rewards[current_p] += self.partial_rewards[current_p]
        self.partial_rewards[current_p] = 0.0

        if self.envido_pending and current_p != self.envido_caller:
            if action_type in ["accept_envido", "reject_envido", "raise_envido"]:
                self._handle_envido_action(action)
                obs = self._get_observation(full_info=False)
                done = self.game_over
                return obs, reward, done, {}
            else:
                return self._get_observation(full_info=False), -5.0, False, {"error": "Must respond to pending envido"}

        if self.truco_pending and current_p != self.truco_caller:
            if action_type in ["accept_truco", "reject_truco", "raise_truco"]:
                self._handle_truco_action(action)
                obs = self._get_observation(full_info=False)
                done = self.game_over
                return obs, reward, done, {}
            else:
                return self._get_observation(full_info=False), -5.0, False, {"error": "Must respond to pending truco"}

        if action_type == "play_card":
            return self._handle_play_card_action(action, current_p, reward)
        elif action_type == "call_envido":
            return self._handle_call_envido_action(action, current_p, reward)
        elif action_type == "call_truco":
            return self._handle_call_truco_action(action, current_p, reward)
        else:
            return self._get_observation(full_info=False), -5.0, False, {"error": f"Unknown action type: {action_type}"}

    def _handle_play_card_action(self, action: Dict[str, Any], current_p: int, reward: float) -> Tuple[
        Dict[str, Any], float, bool, Dict[str, Any]]:
        card_index = action["card_index"]
        if card_index < 0 or card_index >= len(self.hands[current_p]):
            return self._get_observation(full_info=False), -5.0, False, {"error": "Invalid card index"}

        played_card = self.hands[current_p].pop(card_index)
        self.round_cards_played[current_p].append(played_card)
        self.total_cards_played[current_p] += 1

        optimal_card = self._get_optimal_card(current_p)
        if str(played_card) == str(optimal_card):
            self.cards_played_optimally[current_p] += 1
            self._add_partial_reward(current_p, 0.05, "playing optimal card")

        if self.verbose:
            logger.debug(f"Player {current_p} plays {played_card} (index={card_index})")

        if (len(self.round_cards_played[0]) == len(self.round_cards_played[1])
                and len(self.round_cards_played[0]) > 0):
            c0 = self.round_cards_played[0][-1]
            c1 = self.round_cards_played[1][-1]
            winner = determine_card_round_winner(c0, c1)

            if winner is not None:
                self._add_partial_reward(winner, 0.3, f"winning round {self.current_round}")
                self._add_partial_reward(1 - winner, -0.2, f"losing round {self.current_round}")
                self.round_wins[winner] += 1
                self.round_winner_history.append(winner)
                if self.verbose:
                    logger.debug(f"Round result: Player {winner} wins round {self.current_round}")
            else:
                if self.verbose:
                    logger.debug(f"Round result: Tied round, no one wins.")
                self._add_partial_reward(0, -0.05, f"tied round {self.current_round}")
                self._add_partial_reward(1, -0.05, f"tied round {self.current_round}")
                self.round_winner_history.append(-1)

            if max(self.round_wins) >= 2:
                self._finish_hand(force_end=False)
                obs = self._get_observation(full_info=False)
                done = self.game_over
                return obs, reward, done, {}
            else:
                self.current_round += 1

        self._maybe_force_fallback_end()

        self.current_player = 1 - current_p

        obs = self._get_observation(full_info=False)
        done = self.game_over

        future_actions = self.get_valid_actions()
        if not done and not future_actions:
            if (len(self.hands[0]) == 0 and len(self.hands[1]) == 0) or max(self.round_wins) >= 2:
                if self.verbose:
                    logger.debug("No valid actions remain => forcibly finishing hand.")
                self._force_end_hand_with_winner()
                obs = self._get_observation(full_info=False)
                done = self.game_over
                return obs, reward, done, {}
            else:
                if self.verbose:
                    logger.debug(f"Unexpected state: No valid actions but hand not complete.")
                    logger.debug(
                        f"Round wins: {self.round_wins}, Cards left: P0={len(self.hands[0])}, P1={len(self.hands[1])}")
                other_player = 1 - self.current_player
                if len(self.hands[other_player]) > 0:
                    self.current_player = other_player
                    return self._get_observation(full_info=False), reward, False, {}
                else:
                    self._force_end_hand_with_winner()
                    obs = self._get_observation(full_info=False)
                    done = self.game_over
                    return obs, reward, done, {}

        position_reward = self._calculate_position_improvement_reward(current_p)
        if position_reward != 0:
            self._add_partial_reward(current_p, position_reward, "position improvement")

        return obs, reward, done, {}

    def _handle_call_envido_action(self, action: Dict[str, Any], current_p: int, reward: float) -> Tuple[
        Dict[str, Any], float, bool, Dict[str, Any]]:
        which = action["which"]
        if self.verbose:
            logger.debug(f"Player {current_p} calls {which}.")

        if which not in valid_envido_order or self.envido_stage is not None:
            return self._get_observation(full_info=False), -5.0, False, {"error": "Invalid envido call"}

        if self.envido_initiator is None:
            self.envido_initiator = current_p

        self.envido_stage = which
        self.envido_pending = True
        self.envido_caller = current_p
        self.current_player = 1 - current_p

        self.envido_history.append({"player": current_p, "stage": which})
        self.envido_attempts[current_p] += 1
        self.total_envido_calls[current_p] += 1

        aggression_value = valid_envido_order.index(which) / (len(valid_envido_order) - 1) * 0.7 + 0.3
        self._update_aggression_score(current_p, aggression_value)

        obs = self._get_observation(full_info=False)
        done = self.game_over
        return obs, reward, done, {}

    def _handle_call_truco_action(self, action: Dict[str, Any], current_p: int, reward: float) -> Tuple[
        Dict[str, Any], float, bool, Dict[str, Any]]:
        which = action["which"]
        if self.verbose:
            logger.debug(f"Player {current_p} calls {which}.")

        if which not in valid_truco_order or self.truco_stage is not None:
            return self._get_observation(full_info=False), -5.0, False, {"error": "Invalid truco call"}

        if self.truco_initiator is None:
            self.truco_initiator = current_p

        self.truco_stage = which
        self.truco_pending = True
        self.truco_caller = current_p
        self.current_player = 1 - current_p

        self.truco_history.append({"player": current_p, "stage": which})
        self.truco_attempts[current_p] += 1
        self.total_truco_calls[current_p] += 1

        self._add_partial_reward(current_p, 0.1, f"calling {which}")

        aggression_value = valid_truco_order.index(which) / (len(valid_truco_order) - 1) * 0.7 + 0.3
        self._update_aggression_score(current_p, aggression_value)

        obs = self._get_observation(full_info=False)
        done = self.game_over

        return obs, reward, done, {}

    def _update_aggression_score(self, player: int, new_value: float):
        alpha = 0.2
        self.aggression_score[player] = (1 - alpha) * self.aggression_score[player] + alpha * new_value
        self.aggression_samples[player] += 1

    def _handle_envido_action(self, action: Dict[str, Any]):
        p = self.current_player
        a_type = action["type"]

        if a_type == "accept_envido":
            if self.verbose:
                logger.debug(f"Player {p} accepts {self.envido_stage}.")

            p0_score = compute_envido_score(self.original_hands[0])
            p1_score = compute_envido_score(self.original_hands[1])

            self.envido_scores_history[0].append(p0_score)
            self.envido_scores_history[1].append(p1_score)
            self.avg_envido_scores[0] = np.mean(self.envido_scores_history[0])
            self.avg_envido_scores[1] = np.mean(self.envido_scores_history[1])

            if self.verbose:
                logger.debug(f"Envido scores: \n Player 0: {p0_score}, Player 1: {p1_score}")

            if p0_score == p1_score:
                winner = self.mano
                if self.verbose:
                    logger.debug(f"Envido is tied, mano (Player {self.mano}) wins.")
            else:
                winner = 0 if p0_score > p1_score else 1
                if self.verbose:
                    logger.debug(f"Player {winner} wins the envido.")

            self.envido_wins[winner] += 1

            initiator = self.envido_initiator
            if initiator is not None and initiator == winner and (p0_score if winner == 0 else p1_score) < 20:
                self.successful_bluffs[winner] += 1
                self._add_partial_reward(winner, 0.3, "successful envido bluff")

            pts = get_envido_points_on_accept(self.envido_stage)
            if pts == "game":
                if self.verbose:
                    logger.debug(f"Game-winning envido! Player {winner} wins.")
                self.scores[winner] = self.max_points
            else:
                if self.verbose:
                    logger.debug(f"Player {winner} gets {pts} points from {self.envido_stage}.")
                self._add_points(winner, pts)

            loser = 1 - winner
            score_diff = abs(p0_score - p1_score)

            bonus = 0.2 if score_diff > 5 else 0.0

            self._calculate_envido_rewards(winner, loser, bonus)

            self.envido_stage = None
            self.envido_pending = False
            self.envido_finished = True
            self.current_player = self.envido_caller
            self.envido_caller = None

            if self.scores[winner] >= self.max_points:
                self.game_over = True

        elif a_type == "reject_envido":
            caller = self.envido_caller
            if self.verbose:
                logger.debug(f"Player {p} rejects {self.envido_stage}.")

            self.rejection_counts[p]["envido"] += 1

            pts = get_envido_points_on_reject(self.envido_stage)
            if self.verbose:
                logger.debug(f"Player {caller} gets {pts} points from rejection.")
            self._add_points(caller, pts)

            self._calculate_envido_rejection_rewards(caller, p)

            self.envido_stage = None
            self.envido_pending = False
            self.envido_finished = True
            self.envido_caller = None

            if self.scores[caller] >= self.max_points:
                self.game_over = True

            self.current_player = caller

        elif a_type == "raise_envido":
            which = action["which"]
            if self.verbose:
                logger.debug(f"Player {p} raises envido to {which}.")

            if which not in valid_envido_order:
                return

            old_idx = valid_envido_order.index(self.envido_stage)
            new_idx = valid_envido_order.index(which)
            if new_idx <= old_idx:
                return

            if which == "FaltaEnvido" and self.scores[0] < 15 and self.scores[1] < 15:
                if self.verbose:
                    logger.debug(f"Player {p} called FaltaEnvido too early, applying penalty.")
                self._add_partial_reward(p, -2.0, "calling FaltaEnvido too early")

            self.envido_history.append({"player": p, "stage": which, "raised_from": self.envido_stage})
            aggression_value = (new_idx - old_idx) / (len(valid_envido_order) - 1) * 0.7 + 0.3
            self._update_aggression_score(p, aggression_value)

            self.envido_stage = which
            self.current_player = 1 - p
            self.envido_caller = p

        else:
            logger.error(f"Invalid envido action: {a_type}")

    def _calculate_envido_rewards(self, winner: int, loser: int, bonus: float):
        if self.envido_stage == "Envido":
            if winner == self.envido_initiator:
                self._add_partial_reward(winner, 0.8 + bonus, f"called and won {self.envido_stage}")
                self._add_partial_reward(loser, -1.0, f"accepted and lost {self.envido_stage}")
            else:
                self._add_partial_reward(winner, 0.6 + bonus, f"accepted and won {self.envido_stage}")
                self._add_partial_reward(loser, -0.4, f"called and lost {self.envido_stage}")

        elif self.envido_stage == "RealEnvido":
            if winner == self.envido_initiator:
                self._add_partial_reward(winner, 1.2 + bonus, f"called and won {self.envido_stage}")
                self._add_partial_reward(loser, -1.5, f"accepted and lost {self.envido_stage}")
            else:
                self._add_partial_reward(winner, 0.9 + bonus, f"accepted and won {self.envido_stage}")
                self._add_partial_reward(loser, -0.7, f"called and lost {self.envido_stage}")

        elif self.envido_stage == "FaltaEnvido":
            if winner == self.envido_initiator:
                self._add_partial_reward(winner, 2.0 + bonus, f"called and won {self.envido_stage}")
                self._add_partial_reward(loser, -3.0, f"accepted and lost {self.envido_stage}")
            else:
                self._add_partial_reward(winner, 1.5 + bonus, f"accepted and won {self.envido_stage}")
                self._add_partial_reward(loser, -1.0, f"called and lost {self.envido_stage}")

    def _calculate_envido_rejection_rewards(self, caller: int, rejector: int):
        if self.envido_stage == "Envido":
            self._add_partial_reward(caller, 0.5, f"successful {self.envido_stage} call (opponent rejected)")
            self._add_partial_reward(rejector, -0.2, f"rejecting {self.envido_stage}")
        elif self.envido_stage == "RealEnvido":
            self._add_partial_reward(caller, 0.7, f"successful {self.envido_stage} call (opponent rejected)")
            self._add_partial_reward(rejector, -0.3, f"rejecting {self.envido_stage}")
        elif self.envido_stage == "FaltaEnvido":
            self._add_partial_reward(caller, 1.0, f"successful {self.envido_stage} call (opponent rejected)")
            self._add_partial_reward(rejector, -0.4, f"rejecting {self.envido_stage}")

    def _handle_truco_action(self, action: Dict[str, Any]):
        p = self.current_player
        a_type = action["type"]

        if a_type == "accept_truco":
            if self.verbose:
                logger.debug(f"Player {p} accepts {self.truco_stage}.")

            self.truco_pending = False
            self.current_player = self.truco_caller

            if self.truco_stage == "Truco":
                self._add_partial_reward(p, 0.05, f"accepting {self.truco_stage}")
            elif self.truco_stage == "ReTruco":
                self._add_partial_reward(p, 0.1, f"accepting {self.truco_stage}")
            elif self.truco_stage == "ValeCuatro":
                self._add_partial_reward(p, 0.15, f"accepting {self.truco_stage}")

        elif a_type == "reject_truco":
            caller = self.truco_caller
            if self.verbose:
                logger.debug(f"Player {p} rejects {self.truco_stage}.")

            self.rejection_counts[p]["truco"] += 1

            pts = get_truco_points_on_reject(self.truco_stage)
            if self.verbose:
                logger.debug(f"Player {caller} gets {pts} points from rejection.")
            self._add_points(caller, pts)

            self._finish_hand_truco_reject(winner=caller)

        elif a_type == "raise_truco":
            which = action["which"]
            if self.verbose:
                logger.debug(f"Player {p} raises truco to {which}.")

            if which not in valid_truco_order:
                return

            old_idx = valid_truco_order.index(self.truco_stage)
            new_idx = valid_truco_order.index(which)
            if new_idx <= old_idx:
                return

            self.truco_history.append({"player": p, "stage": which, "raised_from": self.truco_stage})

            if which == "ReTruco":
                self._add_partial_reward(p, 0.15, f"raising to {which}")
            elif which == "ValeCuatro":
                self._add_partial_reward(p, 0.2, f"raising to {which}")

            aggression_value = (new_idx - old_idx) / (len(valid_truco_order) - 1) * 0.7 + 0.3
            self._update_aggression_score(p, aggression_value)

            self.truco_stage = which
            self.truco_pending = True
            self.truco_caller = p
            self.current_player = 1 - p

        else:
            logger.error(f"Invalid truco action: {a_type}")

    def _finish_hand_truco_reject(self, winner: int):
        if self.verbose:
            logger.info("\n--- Hand ended (Truco was rejected). ---")
            logger.info(f"Player {winner} gets partial points, now has {self.scores[winner]}")
            logger.info(f"Current scores: {self.scores}")

        loser = 1 - winner
        game_progress = self._get_game_progress()
        progress_multiplier = 1 + (game_progress * 0.5)

        if self.truco_stage == "Truco":
            base_win = 3.0
            base_lose = -3.5
        elif self.truco_stage == "ReTruco":
            base_win = 4.0
            base_lose = -5.0
        elif self.truco_stage == "ValeCuatro":
            base_win = 5.0
            base_lose = -7.0
        else:
            base_win = 1.0
            base_lose = -2.0

        final_win = base_win * progress_multiplier
        final_lose = base_lose * progress_multiplier

        self._add_partial_reward(winner, final_win, f"winning hand by {self.truco_stage} rejection")
        self._add_partial_reward(loser, final_lose, f"losing hand by rejecting {self.truco_stage}")

        self.recent_outcomes.append((winner, self.hand_partial_rewards[0], self.hand_partial_rewards[1]))

        self.truco_wins[winner] += 1

        if self.scores[winner] >= self.max_points:
            self.game_over = True
            self._handle_game_over(winner)
        else:
            self._print_end_of_hand_summary()
            self._rotate_dealer_and_start_new()

    def _maybe_force_fallback_end(self):
        if len(self.hands[0]) == 0 and len(self.hands[1]) == 0:
            self._force_end_hand_with_winner()

    def _force_end_hand_with_winner(self):
        if not self.current_hand_active:
            return

        if self.verbose:
            logger.debug(f"\nFinal round wins - Player 0: {self.round_wins[0]}, Player 1: {self.round_wins[1]}")
            logger.debug(
                f"Cards played in the final round - Player 0: {self.round_cards_played[0][-1] if self.round_cards_played[0] else 'None'}, Player 1: {self.round_cards_played[1][-1] if self.round_cards_played[1] else 'None'}")

        if self.round_wins[0] > self.round_wins[1]:
            winner = 0
            if self.verbose:
                logger.debug(f"Player 0 wins with {self.round_wins[0]} rounds vs {self.round_wins[1]}.")
        elif self.round_wins[1] > self.round_wins[0]:
            winner = 1
            if self.verbose:
                logger.debug(f"Player 1 wins with {self.round_wins[1]} rounds vs {self.round_wins[0]}.")
        else:
            if (self.round_cards_played[0] and self.round_cards_played[1]) and len(self.round_cards_played[0]) == len(
                    self.round_cards_played[1]):
                winner = None
                for i in range(3):
                    if i < len(self.round_cards_played[0]) and i < len(self.round_cards_played[1]):
                        c0 = self.round_cards_played[0][i]
                        c1 = self.round_cards_played[1][i]
                        result = determine_card_round_winner(c0, c1)
                        if result is not None:
                            winner = result
                            if self.verbose:
                                logger.debug(
                                    f"Tied round happened but Player {winner} wins due to having won the first (non-tied) round.")
                            break

                if winner is None:
                    winner = self.mano
                if self.verbose:
                    logger.debug(f"Tied in all 3 rounds thus Player {winner} wins for being mano.")
            else:
                logger.error(
                    f"Mismatch in number of cards played. "
                    f"Player 0: {len(self.round_cards_played[0])}, "
                    f"Player 1: {len(self.round_cards_played[1])}"
                    f"Both players MUST have played the same number of cards to evaluate a winner."
                )
                winner = self.mano

        if self.truco_stage is not None:
            self.truco_wins[winner] += 1

        if self.truco_stage is None:
            self._add_points(winner, 1)
        else:
            pts = get_truco_points_on_accept(self.truco_stage)
            self._add_points(winner, pts)

        if self.verbose:
            logger.info("\n--- Hand ended with normal best-of-3.. ---")
            logger.info(f"Player {winner} is declared the winner.")
            logger.info(f"Current scores: {self.scores}")

        self.current_hand_active = False

        self._calculate_normal_win_rewards(winner)

        self._check_missed_opportunities(winner, 1 - winner)

        self.recent_outcomes.append((winner, self.hand_partial_rewards[0], self.hand_partial_rewards[1]))

        if self.scores[winner] >= self.max_points:
            self.game_over = True
            self._handle_game_over(winner)
        else:
            self._print_end_of_hand_summary()
            self._rotate_dealer_and_start_new()

    def _calculate_normal_win_rewards(self, winner: int):
        loser = 1 - winner
        game_progress = self._get_game_progress()
        progress_multiplier = 1 + (game_progress * 0.5)

        score_before = self.start_of_hand_scores[:]
        comeback_bonus = 0.5 if score_before[winner] < score_before[loser] else 0.0
        closing_bonus = 0.3 if score_before[winner] >= 25 else 0.0

        if self.truco_stage is None:
            base_win = 2.0 + comeback_bonus + closing_bonus
            base_lose = -1.5
        else:
            truco_rewards = {
                "Truco": (3.0, -2.5),
                "ReTruco": (4.5, -3.5),
                "ValeCuatro": (6.0, -5.0)
            }
            base_win, base_lose = truco_rewards[self.truco_stage]
            base_win += comeback_bonus + closing_bonus

        final_win = base_win * progress_multiplier
        final_lose = base_lose * progress_multiplier

        if self.truco_stage and winner == self.truco_initiator:
            self._add_partial_reward(winner, final_win, f"initiated and won hand with {self.truco_stage}")
            self._add_partial_reward(loser, final_lose, f"accepted and lost hand with {self.truco_stage}")
        elif self.truco_stage:
            self._add_partial_reward(winner, final_win * 0.9, f"accepted and won hand with {self.truco_stage}")
            self._add_partial_reward(loser, final_lose * 1.1, f"initiated and lost hand with {self.truco_stage}")
        else:
            self._add_partial_reward(winner, final_win, f"winning hand (normal)")
            self._add_partial_reward(loser, final_lose, f"losing hand (normal)")

    def _finish_hand(self, force_end: bool = False):
        if not self.current_hand_active:
            return

        winner = None
        pts = 0

        if not force_end:
            if self.round_wins[0] > self.round_wins[1]:
                winner = 0
            else:
                winner = 1
            if self.truco_stage is not None:
                self.truco_wins[winner] += 1

            if self.truco_stage is None:
                pts = 1
                self._add_points(winner, 1)
            else:
                pts = get_truco_points_on_accept(self.truco_stage)
                self._add_points(winner, pts)

            if self.verbose:
                logger.info(f"\n--- Hand finished with normal best-of-3. ---")
                logger.info(f"Player {winner} won the hand, gaining {pts} point(s).")
        else:
            if self.verbose:
                logger.info("\n--- Hand ended forcibly. A winner was assigned by fallback. ---")

        if self.verbose:
            logger.info(f"Current scores: {self.scores}")

        self.current_hand_active = False
        self._calculate_normal_win_rewards(winner)
        self._check_missed_opportunities(winner, 1 - winner)
        self.recent_outcomes.append((winner, self.hand_partial_rewards[0], self.hand_partial_rewards[1]))

        if self.scores[0] >= self.max_points or self.scores[1] >= self.max_points:
            winner = 0 if self.scores[0] >= self.max_points else 1
            self.game_over = True
            self._handle_game_over(winner)
        else:
            self._print_end_of_hand_summary()
            self._rotate_dealer_and_start_new()

    def _handle_game_over(self, winner: int):
        loser = 1 - winner

        self._add_partial_reward(winner, 10.0, "winning the game")
        self._add_partial_reward(loser, -8.0, "losing the game")

        self.metrics["games_played"] += 1
        self.metrics["games_won"][winner] += 1
        self.metrics["win_rate"] = [
            self.metrics["games_won"][0] / self.metrics["games_played"],
            self.metrics["games_won"][1] / self.metrics["games_played"]
        ]
        self.metrics["avg_points_scored"] = [
            (self.metrics["avg_points_scored"][0] * (self.metrics["games_played"] - 1) + self.scores[0]) / self.metrics[
                "games_played"],
            (self.metrics["avg_points_scored"][1] * (self.metrics["games_played"] - 1) + self.scores[1]) / self.metrics[
                "games_played"]
        ]
        self.metrics["avg_reward_per_game"] = [
            (self.metrics["avg_reward_per_game"][0] * (self.metrics["games_played"] - 1) + self.total_game_rewards[0]) /
            self.metrics["games_played"],
            (self.metrics["avg_reward_per_game"][1] * (self.metrics["games_played"] - 1) + self.total_game_rewards[1]) /
            self.metrics["games_played"]
        ]
        self.metrics["avg_hands_per_game"] = ((self.metrics["avg_hands_per_game"] * (
                    self.metrics["games_played"] - 1)) + self.hands_played) / self.metrics["games_played"]

        self.metrics["envido_success_rate"] = [
            self.envido_wins[0] / max(1, self.envido_attempts[0]),
            self.envido_wins[1] / max(1, self.envido_attempts[1])
        ]
        self.metrics["truco_success_rate"] = [
            self.truco_wins[0] / max(1, self.truco_attempts[0]),
            self.truco_wins[1] / max(1, self.truco_attempts[1])
        ]
        self.metrics["bluff_success_rate"] = [
            self.successful_bluffs[0] / max(1, self.bluff_attempts[0]),
            self.successful_bluffs[1] / max(1, self.bluff_attempts[1])
        ]
        self.metrics["optimal_card_play_rate"] = [
            self.cards_played_optimally[0] / max(1, self.total_cards_played[0]),
            self.cards_played_optimally[1] / max(1, self.total_cards_played[1])
        ]

        self.recent_outcomes.append((winner, self.total_game_rewards[0], self.total_game_rewards[1]))

        if self.enable_curriculum:
            self._adjust_curriculum_difficulty()

        if self.verbose:
            logger.info(f"\n######## GAME OVER ########")
            logger.info(f"Final scores: {self.scores}")
            logger.info(f"Winner: Player {winner}")
            logger.info(
                f"Total partial rewards - Player 0: {self.total_game_rewards[0]:.2f}, Player 1: {self.total_game_rewards[1]:.2f}")
            logger.info("#########################")

    def _rotate_dealer_and_start_new(self):
        self.current_hand_active = False

        if self.verbose:
            logger.debug(f"round_wins: {self.round_wins}, "
                         f"cards left P0={len(self.hands[0])}, P1={len(self.hands[1])}")

        old_dealer = self.dealer
        old_mano = self.mano
        self.dealer = old_mano
        self.mano = old_dealer

        if not self.game_over:
            self._start_new_hand()

    def _print_end_of_hand_summary(self):
        p0_gained = self.scores[0] - self.start_of_hand_scores[0]
        p1_gained = self.scores[1] - self.start_of_hand_scores[1]
        if self.verbose:
            logger.info("---------------")
            logger.info(f"Match {self.hands_played} summary:")
            logger.info(f"Points gained - Player 0: {p0_gained}, Player 1: {p1_gained}")
            logger.info(f"Updated scores: ({self.scores[0]}, {self.scores[1]})")

            logger.info(
                f"Partial rewards this hand - Player 0: {self.hand_partial_rewards[0]:.2f}, "
                f"Player 1: {self.hand_partial_rewards[1]:.2f}")
            logger.info(
                f"Total game rewards so far - Player 0: {self.total_game_rewards[0]:.2f}, "
                f"Player 1: {self.total_game_rewards[1]:.2f}")
            logger.info("---------------")

    def get_valid_actions(self) -> List[Dict[str, Any]]:
        p = self.current_player
        actions = []

        if self.envido_pending and p != self.envido_caller:
            actions.append({"type": "accept_envido"})
            actions.append({"type": "reject_envido"})
            old_idx = valid_envido_order.index(self.envido_stage)
            if old_idx < len(valid_envido_order) - 1:
                nxt = valid_envido_order[old_idx + 1]
                actions.append({"type": "raise_envido", "which": nxt})
            return actions

        if self.truco_pending and p != self.truco_caller:
            actions.append({"type": "accept_truco"})
            actions.append({"type": "reject_truco"})
            old_idx = valid_truco_order.index(self.truco_stage)
            if old_idx < len(valid_truco_order) - 1:
                nxt = valid_truco_order[old_idx + 1]
                actions.append({"type": "raise_truco", "which": nxt})
            return actions

        if self.current_round == 1 and not self.envido_finished:
            if self.envido_stage is None and len(self.round_cards_played[p]) == 0:
                actions.append({"type": "call_envido", "which": "Envido"})

        if self.truco_stage is None and \
                (len(self.round_cards_played[p]) == len(self.round_cards_played[1 - p])):
            actions.append({"type": "call_truco", "which": "Truco"})

        for idx, _ in enumerate(self.hands[p]):
            actions.append({"type": "play_card", "card_index": idx})

        return actions

    def _get_observation(self, full_info: bool = False) -> Dict[str, Any]:

        def process_cards(cards, is_opponent=False):
            if is_opponent and not full_info:
                return ([self.hidden_card_id] * len(cards)) if self.use_encoded_cards else ["Hidden Card"] * len(cards)
            else:
                if self.use_encoded_cards:
                    return [self._encode_card(str(c)) for c in cards]
                else:
                    return [str(c) for c in cards]

        if full_info:
            hands = [process_cards(self.hands[0]), process_cards(self.hands[1])]

            round_cards = [
                process_cards(self.round_cards_played[0]),
                process_cards(self.round_cards_played[1])
            ]

            obs = {
                "scores": list(self.scores),
                "dealer": self.dealer,
                "mano": self.mano,
                "current_player": self.current_player,
                "hands": hands,
                "current_round": self.current_round,
                "round_wins": list(self.round_wins),
                "round_cards_played": round_cards,
                "truco_stage": self.truco_stage,
                "truco_pending": self.truco_pending,
                "truco_caller": self.truco_caller,
                "envido_stage": self.envido_stage,
                "envido_pending": self.envido_pending,
                "envido_caller": self.envido_caller,
                "game_over": self.game_over,
                "hand_number": self.hands_played,
                "cards_played_total": len(self.round_cards_played[0]) + len(self.round_cards_played[1]),
                "envido_history": self.envido_history.copy(),
                "truco_history": self.truco_history.copy(),
                "last_action": self.last_action,
                "action_history": self.hand_action_history.copy(),
                "opponent_rejected_envido_rate": [
                    self.rejection_counts[i]["envido"] / max(1, self.total_envido_calls[i])
                    for i in range(2)
                ],
                "opponent_rejected_truco_rate": [
                    self.rejection_counts[i]["truco"] / max(1, self.total_truco_calls[i])
                    for i in range(2)
                ],
                "opponent_avg_envido_score": self.avg_envido_scores.copy(),
                "aggression_score": self.aggression_score.copy(),
                "opponent_hand_strength_model": [
                    model["high_card_rate"] for model in self.opponent_hand_strength_model
                ],
            }
            return obs
        else:
            p = self.current_player
            opponent = 1 - p

            hands = [
                process_cards(self.hands[0], is_opponent=(0 != p)),
                process_cards(self.hands[1], is_opponent=(1 != p))
            ]

            round_cards = [
                process_cards(self.round_cards_played[0]),
                process_cards(self.round_cards_played[1])
            ]

            normalized_scores = [s / self.max_points for s in self.scores]
            normalized_round = (self.current_round - 1) / 2.0
            normalized_round_wins = [w / 3.0 for w in self.round_wins]
            normalized_hand_number = self.hands_played / 30.0

            obs = {
                "scores": list(self.scores),
                "normalized_scores": normalized_scores,
                "dealer": self.dealer,
                "mano": self.mano,
                "current_player": self.current_player,
                "hands": [hands[p], hands[opponent]],
                "current_round": self.current_round,
                "normalized_round": normalized_round,
                "round_wins": list(self.round_wins),
                "normalized_round_wins": normalized_round_wins,
                "round_cards_played": round_cards,
                "truco_stage": self.truco_stage,
                "truco_pending": self.truco_pending,
                "truco_caller": self.truco_caller,
                "envido_stage": self.envido_stage,
                "envido_pending": self.envido_pending,
                "envido_caller": self.envido_caller,
                "game_over": self.game_over,
                "is_player": p,
                "hand_number": self.hands_played,
                "normalized_hand_number": normalized_hand_number,
                "cards_played_total": len(self.round_cards_played[0]) + len(self.round_cards_played[1]),
                "envido_history": self.envido_history.copy(),
                "truco_history": self.truco_history.copy(),
                "last_action": self.last_action,
                "action_history": self.hand_action_history.copy(),
                "opponent_rejected_envido_rate": self.rejection_counts[opponent]["envido"] / max(1,
                                                                                                 self.total_envido_calls[
                                                                                                     opponent]),
                "opponent_rejected_truco_rate": self.rejection_counts[opponent]["truco"] / max(1,
                                                                                               self.total_truco_calls[
                                                                                                   opponent]),
                "opponent_avg_envido_score": self.avg_envido_scores[opponent],
                "opponent_aggression": self.aggression_score[opponent],
                "opponent_hand_strength_estimate": self._estimate_opponent_card_strength(opponent),
                "opponent_bluff_tendency": self.successful_bluffs[opponent] / max(1, self.bluff_attempts[opponent]),
                "is_mano": float(p == self.mano),
                "game_progress": self._get_game_progress(),
            }
            return obs

    def play_full_game(self, agent_0, agent_1) -> Dict[str, Any]:
        self.reset()
        hand_history = []

        while not self.game_over:
            done = False
            hand_log = {"points_start": self.scores[:], "actions": [], "points_gained": [0, 0]}

            while not done:
                current_p = self.current_player
                agent = agent_0 if current_p == 0 else agent_1

                obs = self._get_observation(full_info=False)
                valid = self.get_valid_actions()
                action = agent.choose_action(obs, valid)
                obs, reward, done, _ = self.step(action)

                agent.record_reward(reward)

                hand_log["actions"].append({
                    "player": current_p,
                    "action": action,
                    "reward": reward
                })

            agent_0.finish_episode_and_update()
            agent_1.finish_episode_and_update()

            hand_log["points_gained"] = [
                self.scores[0] - hand_log["points_start"][0],
                self.scores[1] - hand_log["points_start"][1]
            ]
            hand_history.append(hand_log)

            if not self.game_over:
                self._start_new_hand()
        winner = 0 if self.scores[0] >= self.max_points else 1

        return {
            "winner": winner,
            "scores": self.scores[:],
            "hand_history": hand_history,
            "total_rewards": self.total_game_rewards[:],
            "statistics": self.get_episode_statistics(),
            "reward_components": dict(self.reward_components)
        }

    def _adjust_curriculum_difficulty(self):
        if len(self.recent_outcomes) < 10:
            return

        win_rate_p0 = sum(1 for w, _, _ in self.recent_outcomes if w == 0) / len(self.recent_outcomes)
        win_rate_p1 = 1.0 - win_rate_p0
        avg_reward_p0 = sum(r0 for _, r0, _ in self.recent_outcomes) / len(self.recent_outcomes)
        avg_reward_p1 = sum(r1 for _, _, r1 in self.recent_outcomes) / len(self.recent_outcomes)

        self.curriculum_metrics["recent_win_rate"] = win_rate_p0 if self.metrics[
                                                                        "games_played"] % 2 == 0 else win_rate_p1
        self.curriculum_metrics["recent_avg_reward"] = avg_reward_p0 if self.metrics[
                                                                            "games_played"] % 2 == 0 else avg_reward_p1
        self.curriculum_metrics["games_at_current_level"] += 1

        current_level = self.difficulty_level
        win_rate = self.curriculum_metrics["recent_win_rate"]
        avg_reward = self.curriculum_metrics["recent_avg_reward"]
        games_at_level = self.curriculum_metrics["games_at_current_level"]

        if (win_rate > 0.65 and avg_reward > 3.0 and games_at_level >= 20) or games_at_level >= 50:
            new_level = min(5, current_level + 1)
            if new_level != current_level:
                if self.verbose:
                    logger.info(f"Curriculum advancing to level {new_level} (from {current_level})")
                self.set_curriculum_difficulty(new_level)
                self.curriculum_metrics["games_at_current_level"] = 0
        elif win_rate < 0.2 and avg_reward < -5.0 and games_at_level >= 10:
            new_level = max(0, current_level - 1)
            if new_level != current_level:
                if self.verbose:
                    logger.info(f"Curriculum regressing to level {new_level} (from {current_level})")
                self.set_curriculum_difficulty(new_level)
                self.curriculum_metrics["games_at_current_level"] = 0

    def _is_high_card(self, card) -> bool:
        card_str = str(card)
        high_cards = ["1 de Espada", "1 de Basto", "7 de Espada", "7 de Oro"]
        return card_str in high_cards

    def _count_high_cards(self, hand) -> int:
        return sum(1 for card in hand if self._is_high_card(card))

    def _check_missed_opportunities(self, winner: int, loser: int):
        winner_high_cards = self._count_high_cards(self.original_hands[winner])

        if winner_high_cards >= 2 and self.truco_stage is None:
            game_progress = self._get_game_progress()
            penalty_multiplier = max(0.3, 1.0 - game_progress)

            penalty = -0.4 * penalty_multiplier
            self._add_partial_reward(winner, penalty, f"wasting hand with {winner_high_cards} high cards without Truco")

        if self.current_round == 1 and not self.envido_finished:
            p0_envido = compute_envido_score(self.original_hands[0])
            p1_envido = compute_envido_score(self.original_hands[1])

            for player, envido_score in [(0, p0_envido), (1, p1_envido)]:
                if envido_score >= 30 and self.envido_initiator != player:
                    game_progress = self._get_game_progress()
                    penalty_multiplier = max(0.3, 1.0 - game_progress)

                    penalty = -0.3 * penalty_multiplier
                    self._add_partial_reward(player, penalty, f"wasting {envido_score} envido points without calling")

    def _get_optimal_card(self, player: int) -> Optional[Any]:
        hand = self.hands[player]
        if not hand:
            return None

        if len(self.round_cards_played[0]) == len(self.round_cards_played[1]):
            card_values = [(card, self._get_card_strength(card)) for card in hand]
            card_values.sort(key=lambda x: x[1])

            return card_values[len(card_values) // 2][0]
        else:
            opponent = 1 - player
            if len(self.round_cards_played[opponent]) > 0:
                opponent_card = self.round_cards_played[opponent][-1]

                winning_cards = []
                for card in hand:
                    if determine_card_round_winner(card, opponent_card) == player:
                        winning_cards.append((card, self._get_card_strength(card)))

                if winning_cards:
                    winning_cards.sort(key=lambda x: x[1])
                    return winning_cards[0][0]
                else:
                    card_values = [(card, self._get_card_strength(card)) for card in hand]
                    card_values.sort(key=lambda x: x[1])
                    return card_values[0][0]
            else:
                card_values = [(card, self._get_card_strength(card)) for card in hand]
                card_values.sort(key=lambda x: x[1])
                return card_values[len(card_values) // 2][0]

    def _get_card_strength(self, card) -> int:
        card_str = str(card)
        return self.card_strength.get(card_str, 0)

    def _calculate_position_improvement_reward(self, player: int) -> float:
        current_position = self.scores[player] - self.scores[1 - player]
        previous_position = self.previous_position_score[player] - self.previous_position_score[1 - player]

        improvement = current_position - previous_position

        if improvement > 0:
            return 0.1 * improvement
        elif improvement < 0:
            return 0.05 * improvement
        return 0.0

    def _estimate_opponent_card_strength(self, player: int) -> float:
        base_strength = 0.5

        if self.truco_caller == player:
            base_strength += 0.2

        avg_envido = self.avg_envido_scores[player]
        if avg_envido > 25:
            base_strength += 0.1
        elif avg_envido < 15:
            base_strength -= 0.1

        played_cards = self.round_cards_played[player]
        high_cards_played = sum(1 for card in played_cards if self._is_high_card(card))

        if high_cards_played > 0:
            base_strength -= 0.15 * high_cards_played

        return max(0.0, min(1.0, base_strength))

    def get_state_encoding(self) -> np.ndarray:
        p = self.current_player
        opponent = 1 - p
        encoding = []

        encoding.extend([self.scores[p] / self.max_points,
                         self.scores[opponent] / self.max_points])
        encoding.append((self.current_round - 1) / 2.0)
        encoding.extend([self.round_wins[p] / 3.0,
                         self.round_wins[opponent] / 3.0])

        hand_encoding = np.zeros(self.num_cards * 3)
        for i, card in enumerate(self.hands[p]):
            card_id = self._encode_card(str(card))
            hand_encoding[i * self.num_cards + card_id] = 1.0
        encoding.extend(hand_encoding)

        played_encoding = np.zeros(self.num_cards * 6)
        idx = 0
        for player_cards in self.round_cards_played:
            for card in player_cards:
                card_id = self._encode_card(str(card))
                played_encoding[idx * self.num_cards + card_id] = 1.0
                idx += 1
        encoding.extend(played_encoding)

        encoding.extend([
            float(self.truco_pending),
            float(self.truco_stage == "Truco"),
            float(self.truco_stage == "ReTruco"),
            float(self.truco_stage == "ValeCuatro"),
            float(self.envido_pending),
            float(self.envido_finished),
            float(self.dealer == p),
            float(self.mano == p),
        ])

        encoding.extend([
            self.rejection_counts[opponent]["envido"] / max(1, self.total_envido_calls[opponent]),
            self.rejection_counts[opponent]["truco"] / max(1, self.total_truco_calls[opponent]),
            self.avg_envido_scores[opponent] / 33.0,
            self.aggression_score[opponent],
            self._estimate_opponent_card_strength(opponent),
        ])

        return np.array(encoding, dtype=np.float32)

    def get_action_mask(self) -> np.ndarray:
        action_space_size = 20

        mask = np.zeros(action_space_size, dtype=bool)
        valid_actions = self.get_valid_actions()

        action_to_idx = {
            "play_card_0": 0,
            "play_card_1": 1,
            "play_card_2": 2,

            "call_envido": 3,
            "accept_envido": 4,
            "reject_envido": 5,
            "raise_envido_real": 6,
            "raise_envido_falta": 7,

            "call_truco": 8,
            "accept_truco": 9,
            "reject_truco": 10,
            "raise_truco_retruco": 11,
            "raise_truco_vale4": 12,
        }

        for action in valid_actions:
            if action["type"] == "play_card":
                idx = action["card_index"]
                if 0 <= idx <= 2:
                    mask[idx] = True
            elif action["type"] == "call_envido":
                mask[3] = True
            elif action["type"] == "accept_envido":
                mask[4] = True
            elif action["type"] == "reject_envido":
                mask[5] = True
            elif action["type"] == "raise_envido":
                if action["which"] == "RealEnvido":
                    mask[6] = True
                elif action["which"] == "FaltaEnvido":
                    mask[7] = True
            elif action["type"] == "call_truco":
                mask[8] = True
            elif action["type"] == "accept_truco":
                mask[9] = True
            elif action["type"] == "reject_truco":
                mask[10] = True
            elif action["type"] == "raise_truco":
                if action["which"] == "ReTruco":
                    mask[11] = True
                elif action["which"] == "ValeCuatro":
                    mask[12] = True

        return mask

    def get_episode_statistics(self) -> Dict[str, Any]:
        stats = {}

        for p in [0, 1]:
            prefix = f"player_{p}_"
            stats.update({
                prefix + "hands_played": self.hands_played,
                prefix + "envido_success_rate": self.envido_wins[p] / max(1, self.envido_attempts[p]),
                prefix + "truco_success_rate": self.truco_wins[p] / max(1, self.truco_attempts[p]),
                prefix + "bluff_success_rate": self.successful_bluffs[p] / max(1, self.bluff_attempts[p]),
                prefix + "avg_hand_length": self.total_actions / max(1, self.hands_played),
                prefix + "optimal_card_play_rate": self.cards_played_optimally[p] / max(1, self.total_cards_played[p]),
                prefix + "envido_rejection_rate": self.rejection_counts[p]["envido"] / max(1,
                                                                                           self.total_envido_calls[p]),
                prefix + "truco_rejection_rate": self.rejection_counts[p]["truco"] / max(1, self.total_truco_calls[p]),
                prefix + "avg_envido_score": self.avg_envido_scores[p],
                prefix + "total_rewards": self.total_game_rewards[p],
                prefix + "aggression_score": self.aggression_score[p],
            })

        reward_breakdown = {}
        for reason, values in self.reward_components.items():
            reward_breakdown[f"reward_p0_{reason}"] = values[0]
            reward_breakdown[f"reward_p1_{reason}"] = values[1]

        stats.update(reward_breakdown)
        stats.update(self.metrics)

        return stats

    def render(self, mode='human'):
        if mode == 'human':
            print("\n=== Current Game State ===")
            print(f"Scores: Player 0: {self.scores[0]}, Player 1: {self.scores[1]}")
            print(f"Current Player: {self.current_player}")
            print(f"Round: {self.current_round}, Wins: {self.round_wins}")
            print(f"Hands:")
            for i in range(2):
                print(f"  Player {i}: {[str(c) for c in self.hands[i]]}")
            print(f"Played Cards:")
            for i in range(2):
                print(f"  Player {i}: {[str(c) for c in self.round_cards_played[i]]}")
            print(f"Truco: {self.truco_stage}, Pending: {self.truco_pending}")
            print(f"Envido: {self.envido_stage}, Pending: {self.envido_pending}")
            print("========================\n")
        elif mode == 'rgb_array':
            raise NotImplementedError("RGB array rendering not implemented")

    def get_curriculum_difficulty(self) -> int:
        return self.difficulty_level

    def set_curriculum_difficulty(self, level: int):
        self.difficulty_level = max(0, min(level, 5))
        self.curriculum_metrics["games_at_current_level"] = 0

        if self.verbose:
            logger.info(f"Curriculum difficulty set to {self.difficulty_level}")

    def get_curriculum_opponent(self):
        if self.difficulty_level == 0:
            return RandomAgent()
        elif self.difficulty_level == 1:
            return SimpleRuleBasedAgent()
        elif self.difficulty_level == 2:
            return IntermediateAgent(aggression_level=0.3)
        elif self.difficulty_level == 3:
            return IntermediateAgent(aggression_level=0.5)
        elif self.difficulty_level == 4:
            return IntermediateAgent(aggression_level=0.7)
        elif self.difficulty_level == 5:
            return IntermediateAgent(aggression_level=0.9)

    def save_performance_metrics(self, filepath):
        import json
        stats = self.get_episode_statistics()
        stats["reward_components"] = {k: list(v) for k, v in self.reward_components.items()}

        if self.enable_curriculum:
            stats["curriculum"] = {
                "difficulty_level": self.difficulty_level,
                "metrics": self.curriculum_metrics
            }

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        if self.verbose:
            logger.info(f"Performance metrics saved to {filepath}")

    def create_baseline_agents(self, agent_type: str = "random"):
        if agent_type == "random":
            return RandomAgent(), RandomAgent()
        elif agent_type == "rule_based":
            return SimpleRuleBasedAgent(), SimpleRuleBasedAgent()
        elif agent_type == "intermediate":
            return IntermediateAgent(), IntermediateAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")