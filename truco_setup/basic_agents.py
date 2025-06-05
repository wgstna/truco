import random
import numpy as np
from typing import Dict, Any, List, Optional


class RandomAgent:
    def choose_action(self, obs, valid_actions):
        return random.choice(valid_actions)

    def record_reward(self, reward):
        pass

    def finish_episode_and_update(self):
        pass


class SimpleRuleBasedAgent:

    def __init__(self):
        self.card_values = self._create_card_value_map()

    def _create_card_value_map(self) -> Dict[str, int]:
        values = {}

        values["1 de Espada"] = 14
        values["1 de Basto"] = 13
        values["7 de Espada"] = 12
        values["7 de Oro"] = 11

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"3 de {suit}"] = 10

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"2 de {suit}"] = 9

        values["1 de Oro"] = 8
        values["1 de Copa"] = 8

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"12 de {suit}"] = 7

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"11 de {suit}"] = 6

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"10 de {suit}"] = 5

        values["7 de Basto"] = 4
        values["7 de Copa"] = 4

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"6 de {suit}"] = 3

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"5 de {suit}"] = 2

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"4 de {suit}"] = 1

        return values

    def _get_card_value(self, card: str) -> int:
        return self.card_values.get(card, 0)

    def _get_envido_score(self, hand: List[str]) -> int:
        suits = {}
        for card in hand:
            if isinstance(card, str) and card != "Hidden Card":
                parts = card.split(" de ")
                if len(parts) == 2:
                    value, suit = parts
                    if suit not in suits:
                        suits[suit] = []
                    val = int(value)
                    if val >= 10:
                        val = 0
                    suits[suit].append(val)

        max_score = 0
        for suit, values in suits.items():
            if len(values) >= 2:
                values.sort(reverse=True)
                score = 20 + values[0] + values[1]
                max_score = max(max_score, score)
            elif len(values) == 1:
                max_score = max(max_score, values[0])

        return max_score

    def choose_action(self, obs: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        player = obs["current_player"]
        my_score = obs["scores"][player]
        opponent_score = obs["scores"][1 - player]

        if obs.get("envido_pending") and player != obs.get("envido_caller"):
            return self._handle_envido_response(obs, valid_actions)

        if obs.get("truco_pending") and player != obs.get("truco_caller"):
            return self._handle_truco_response(obs, valid_actions)

        if obs["current_round"] == 1 and not obs.get("envido_finished"):
            envido_action = self._consider_envido(obs, valid_actions)
            if envido_action:
                return envido_action

        truco_action = self._consider_truco(obs, valid_actions)
        if truco_action:
            return truco_action

        return self._choose_card_to_play(obs, valid_actions)

    def _handle_envido_response(self, obs: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        player = obs["current_player"]
        hand = obs["hands"][player]
        envido_score = self._get_envido_score(hand)

        if envido_score >= 27:
            for action in valid_actions:
                if action["type"] == "raise_envido":
                    return action
            for action in valid_actions:
                if action["type"] == "accept_envido":
                    return action
        elif envido_score >= 23:
            for action in valid_actions:
                if action["type"] == "accept_envido":
                    return action

        for action in valid_actions:
            if action["type"] == "reject_envido":
                return action

        return valid_actions[0]

    def _handle_truco_response(self, obs: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        player = obs["current_player"]
        hand = obs["hands"][player]

        hand_values = []
        for card in hand:
            if isinstance(card, str) and card != "Hidden Card":
                hand_values.append(self._get_card_value(card))

        avg_value = sum(hand_values) / len(hand_values) if hand_values else 0

        if avg_value >= 8:
            for action in valid_actions:
                if action["type"] == "raise_truco":
                    return action
            for action in valid_actions:
                if action["type"] == "accept_truco":
                    return action
        elif avg_value >= 5:
            for action in valid_actions:
                if action["type"] == "accept_truco":
                    return action

        for action in valid_actions:
            if action["type"] == "reject_truco":
                return action

        return valid_actions[0]

    def _consider_envido(self, obs: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        player = obs["current_player"]
        hand = obs["hands"][player]
        envido_score = self._get_envido_score(hand)

        if envido_score >= 25:
            for action in valid_actions:
                if action["type"] == "call_envido":
                    return action

        return None

    def _consider_truco(self, obs: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        player = obs["current_player"]
        hand = obs["hands"][player]

        high_cards = 0
        for card in hand:
            if isinstance(card, str) and card != "Hidden Card":
                if self._get_card_value(card) >= 10:
                    high_cards += 1

        if high_cards >= 2:
            for action in valid_actions:
                if action["type"] == "call_truco":
                    return action

        return None

    def _choose_card_to_play(self, obs: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        player = obs["current_player"]

        card_actions = [a for a in valid_actions if a["type"] == "play_card"]

        if not card_actions:
            return valid_actions[0]

        card_values = []
        for action in card_actions:
            card_idx = action["card_index"]
            card = obs["hands"][player][card_idx]
            if isinstance(card, str) and card != "Hidden Card":
                value = self._get_card_value(card)
                card_values.append((action, value))

        card_values.sort(key=lambda x: x[1])

        if len(obs["round_cards_played"][0]) == len(obs["round_cards_played"][1]):
            if len(card_values) > 1:
                return card_values[len(card_values) // 2][0]
            return card_values[0][0]
        else:
            opponent = 1 - player
            opponent_card = obs["round_cards_played"][opponent][-1]

            for action, value in card_values:
                card_idx = action["card_index"]
                card = obs["hands"][player][card_idx]
                if self._would_win_against(card, opponent_card):
                    return action

            return card_values[0][0]

    def _would_win_against(self, my_card: str, opponent_card: str) -> bool:
        if not isinstance(my_card, str) or not isinstance(opponent_card, str):
            return False

        my_value = self._get_card_value(my_card)
        opp_value = self._get_card_value(opponent_card)

        return my_value > opp_value

    def record_reward(self, reward: float):
        pass

    def finish_episode_and_update(self):
        pass


class IntermediateAgent:

    def __init__(self, aggression_level=0.5):
        self.card_values = self._create_card_value_map()
        self.bluff_probability = 0.15
        self.aggression_level = aggression_level
        self.opponent_stats = {
            "envido_accepts": 0,
            "envido_rejects": 0,
            "truco_accepts": 0,
            "truco_rejects": 0,
            "total_games": 0
        }

    def _create_card_value_map(self):
        values = {}
        values["1 de Espada"] = 14
        values["1 de Basto"] = 13
        values["7 de Espada"] = 12
        values["7 de Oro"] = 11

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"3 de {suit}"] = 10
            values[f"2 de {suit}"] = 9

        values["1 de Oro"] = 8
        values["1 de Copa"] = 8

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"12 de {suit}"] = 7
            values[f"11 de {suit}"] = 6
            values[f"10 de {suit}"] = 5

        values["7 de Basto"] = 4
        values["7 de Copa"] = 4

        for suit in ["Espada", "Basto", "Oro", "Copa"]:
            values[f"6 de {suit}"] = 3
            values[f"5 de {suit}"] = 2
            values[f"4 de {suit}"] = 1

        return values

    def _get_card_value(self, card):
        return self.card_values.get(str(card), 0)

    def _get_envido_score(self, hand):
        suits = {}
        for card in hand:
            if isinstance(card, str) and card != "Hidden Card":
                parts = card.split(" de ")
                if len(parts) == 2:
                    value, suit = parts
                    if suit not in suits:
                        suits[suit] = []
                    try:
                        val = int(value)
                        if val >= 10:
                            val = 0
                        suits[suit].append(val)
                    except ValueError:
                        continue

        max_score = 0
        for suit, values in suits.items():
            if len(values) >= 2:
                values.sort(reverse=True)
                score = 20 + values[0] + values[1]
                max_score = max(max_score, score)
            elif len(values) == 1:
                max_score = max(max_score, values[0])

        return max_score

    def choose_action(self, obs, valid_actions):
        if not valid_actions:
            return None

        player = obs.get("current_player", 0)
        my_score = obs.get("scores", [0, 0])[player] if "scores" in obs else 0
        opponent_score = obs.get("scores", [0, 0])[1 - player] if "scores" in obs else 0

        if obs.get("envido_pending") and player != obs.get("envido_caller"):
            return self._handle_envido_response(obs, valid_actions)

        if obs.get("truco_pending") and player != obs.get("truco_caller"):
            return self._handle_truco_response(obs, valid_actions)

        if obs.get("current_round") == 1 and not obs.get("envido_finished", False):
            envido_action = self._consider_envido(obs, valid_actions)
            if envido_action:
                return envido_action

        truco_action = self._consider_truco(obs, valid_actions)
        if truco_action:
            return truco_action

        return self._choose_card_to_play(obs, valid_actions)

    def _handle_envido_response(self, obs, valid_actions):
        if not valid_actions:
            return None

        player = obs.get("current_player", 0)
        hand = obs.get("hands", [[], []])[player]
        envido_score = self._get_envido_score(hand)

        accept_action = next((a for a in valid_actions if a["type"] == "accept_envido"), None)
        reject_action = next((a for a in valid_actions if a["type"] == "reject_envido"), None)
        raise_action = next((a for a in valid_actions if a["type"] == "raise_envido"), None)

        if envido_score >= 27 and raise_action:
            return raise_action
        elif envido_score >= 23 and accept_action:
            return accept_action
        elif reject_action:
            return reject_action
        return valid_actions[0]

    def _handle_truco_response(self, obs, valid_actions):
        if not valid_actions:
            return None

        player = obs.get("current_player", 0)
        hand = obs.get("hands", [[], []])[player]

        hand_values = []
        for card in hand:
            if isinstance(card, str) and card != "Hidden Card":
                value = self._get_card_value(card)
                hand_values.append(value)

        avg_value = sum(hand_values) / len(hand_values) if hand_values else 0

        accept_action = next((a for a in valid_actions if a["type"] == "accept_truco"), None)
        reject_action = next((a for a in valid_actions if a["type"] == "reject_truco"), None)
        raise_action = next((a for a in valid_actions if a["type"] == "raise_truco"), None)

        if avg_value >= 8 and raise_action:
            return raise_action
        elif avg_value >= 5 and accept_action:
            return accept_action
        elif reject_action:
            return reject_action
        return valid_actions[0]

    def _consider_envido(self, obs, valid_actions):
        player = obs.get("current_player", 0)
        hand = obs.get("hands", [[], []])[player]
        envido_score = self._get_envido_score(hand)

        call_action = next((a for a in valid_actions if a["type"] == "call_envido"), None)

        if call_action:
            if envido_score >= 25:
                return call_action
            elif envido_score < 15 and random.random() < self.bluff_probability:
                return call_action

        return None

    def _consider_truco(self, obs, valid_actions):
        player = obs.get("current_player", 0)
        hand = obs.get("hands", [[], []])[player]

        high_cards = 0
        hand_values = []
        for card in hand:
            if isinstance(card, str) and card != "Hidden Card":
                value = self._get_card_value(card)
                hand_values.append(value)
                if value >= 10:
                    high_cards += 1

        avg_value = sum(hand_values) / len(hand_values) if hand_values else 0
        call_action = next((a for a in valid_actions if a["type"] == "call_truco"), None)

        if call_action:
            if high_cards >= 2:
                return call_action
            elif random.random() < self.bluff_probability * self.aggression_level:
                return call_action

        return None

    def _choose_card_to_play(self, obs, valid_actions):
        if not valid_actions:
            return None

        player = obs.get("current_player", 0)

        card_actions = [a for a in valid_actions if a["type"] == "play_card"]

        if not card_actions:
            return valid_actions[0]

        card_values = []
        for action in card_actions:
            try:
                card_idx = action["card_index"]
                if player < len(obs.get("hands", [])) and card_idx < len(obs.get("hands", [[]])[player]):
                    card = obs["hands"][player][card_idx]
                    if isinstance(card, str) and card != "Hidden Card":
                        value = self._get_card_value(card)
                        card_values.append((action, value, card))
            except (IndexError, KeyError, TypeError):
                continue

        if not card_values:
            return card_actions[0]

        card_values.sort(key=lambda x: x[1])

        if len(obs.get("round_cards_played", [[], []])[0]) == len(obs.get("round_cards_played", [[], []])[1]):
            if len(card_values) > 0:
                mid_idx = len(card_values) // 2
                return card_values[min(mid_idx, len(card_values) - 1)][0]
        else:
            opponent = 1 - player
            if (opponent < len(obs.get("round_cards_played", [])) and
                    len(obs.get("round_cards_played", [[]])[opponent]) > 0):

                opponent_card = obs["round_cards_played"][opponent][-1]

                winning_cards = []
                for action, value, card in card_values:
                    if self._would_win_against(card, opponent_card):
                        winning_cards.append((action, value, card))

                if winning_cards:
                    return winning_cards[0][0]

            if card_values:
                return card_values[0][0]

        return card_actions[0]

    def _would_win_against(self, my_card, opponent_card):
        if not isinstance(my_card, str) or not isinstance(opponent_card, str):
            return False

        my_value = self._get_card_value(my_card)
        opp_value = self._get_card_value(opponent_card)

        return my_value > opp_value

    def record_reward(self, reward):
        pass

    def finish_episode_and_update(self):
        self.opponent_stats["total_games"] += 1
