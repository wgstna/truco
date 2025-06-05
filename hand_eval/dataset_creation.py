from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from itertools import product
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd

p_root = os.path.dirname(os.path.abspath(__file__))
if p_root not in sys.path:
    sys.path.append(p_root)

from truco_setup.truco_environment import TrucoEnvironment
from truco_setup.basic_agents import RandomAgent, SimpleRuleBasedAgent, IntermediateAgent

env = TrucoEnvironment(seed=0, use_encoded_cards=True)
card_rank: Dict[str, int] = env.card_strength
id_to_card_: Dict[int, str] = env.id_to_card
total_cards: int = len(id_to_card_)
_strength_to_cards: Dict[int, List[str]] = {}
for card_str, strength in card_rank.items():
    _strength_to_cards.setdefault(strength, []).append(card_str)

tie_probability: Dict[str, float] = {}
stronger_count: Dict[str, int] = {}
for strength, group in _strength_to_cards.items():
    stronger_cards = [c for s, g in _strength_to_cards.items() if s > strength for c in g]
    for c in group:
        tie_probability[c] = (len(group) - 1) / (total_cards - 1)
        stronger_count[c] = len(stronger_cards)

def _envido_value(card_str: str) -> int:
    first_token = card_str.split()[0]
    try:
        rank_int = int(first_token)
    except ValueError:
        return 0
    return rank_int if rank_int <= 7 else 0


def _envido_points(cards: Tuple[str, str, str]) -> int:
    suit_groups: Dict[str, List[int]] = {}
    for c in cards:
        suit_groups.setdefault(c[-1], []).append(_envido_value(c))

    best_same_suit = -1
    for vals in suit_groups.values():
        if len(vals) >= 2:
            best_same_suit = max(best_same_suit, 20 + sum(sorted(vals)[-2:]))

    if best_same_suit >= 0:
        return best_same_suit
    return max(_envido_value(c) for c in cards)

def _play_game_and_collect(agent0_cls: Type, agent1_cls: Type, seed: int) -> List[dict]:
    env = TrucoEnvironment(seed=seed, use_encoded_cards=True)
    agent0, agent1 = agent0_cls(), agent1_cls()
    env.reset()

    rows: List[dict] = []
    start_scores = env.scores[:]
    recorded_hand_number: int | None = None
    tmp_rows: List[dict] = []

    while not env.game_over:
        if (
            env.current_round == 1
            and not env.round_cards_played[0]
            and not env.round_cards_played[1]
            and recorded_hand_number != env.hands_played
        ):
            recorded_hand_number = env.hands_played
            start_scores = env.scores[:]
            tmp_rows.clear()

            obs_full = env._get_observation(full_info=True)
            dealer_id = obs_full["dealer"]
            mano_id = obs_full["mano"]

            for pid in (0, 1):
                hand_encoded = obs_full["hands"][pid]
                card_strs = [id_to_card_[cid] for cid in hand_encoded]
                strengths = [card_rank[s] for s in card_strs]
                suits = [s.split()[-1] for s in card_strs]
                most_common = Counter(suits).most_common(1)[0][1]

                row: dict = {
                    # Identifiers
                    "game_seed": seed,
                    "hand_number": env.hands_played,
                    "player_id": pid,
                    "agent_player": agent0_cls.__name__ if pid == 0 else agent1_cls.__name__,
                    "agent_opponent": agent1_cls.__name__ if pid == 0 else agent0_cls.__name__,
                    "dealer_id": dealer_id,
                    "mano_id": mano_id,
                    "is_dealer": int(dealer_id == pid),
                    "is_mano": int(mano_id == pid),
                    "card1_id": hand_encoded[0],
                    "card2_id": hand_encoded[1],
                    "card3_id": hand_encoded[2],
                    "card1_str": card_strs[0],
                    "card2_str": card_strs[1],
                    "card3_str": card_strs[2],
                    "strength1": strengths[0],
                    "strength2": strengths[1],
                    "strength3": strengths[2],
                    "strength_sum": int(sum(strengths)),
                    "strength_max": int(max(strengths)),
                    "starting_score_player": start_scores[pid],
                    "starting_score_opponent": start_scores[1 - pid],
                }

                row["num_same_suit"] = most_common
                row["has_two_same_suit"] = int(most_common >= 2)

                row["envido_points"] = _envido_points(tuple(card_strs))

                for i, cstr in enumerate(card_strs, start=1):
                    row[f"tie_probability_card{i}"] = tie_probability[cstr]
                    row[f"stronger_count_card{i}"] = stronger_count[cstr]
                tmp_rows.append(row)

        current_pid = env.current_player
        acting_agent = agent0 if current_pid == 0 else agent1
        obs = env._get_observation()
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        try:
            action = acting_agent.choose_action(obs, valid_actions)
        except Exception:
            action = random.choice(valid_actions)
        _, _, done, _ = env.step(action)

        if not env.current_hand_active or done:
            for row in tmp_rows:
                pid = row["player_id"]
                gained = env.scores[pid] - start_scores[pid]
                row["points_gained"] = gained
                row["won_hand"] = int(gained > 0)
                rows.append(row.copy())
            tmp_rows.clear()

        if done:
            break

    return rows

def generate_dataset(num_games_per_matchup: int, output_csv: str, seed: int = 0) -> None:
    rng = random.Random(seed)
    all_rows: List[dict] = []

    agent_classes = [RandomAgent, SimpleRuleBasedAgent, IntermediateAgent]

    for cls_a, cls_b in product(agent_classes, repeat=2):
        for _ in range(num_games_per_matchup):
            game_seed = rng.randrange(2**32)
            all_rows.extend(_play_game_and_collect(cls_a, cls_b, game_seed))

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df):,} rows to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Truco hand‑level dataset for ML training")
    p.add_argument("--num_games", type=int, default=50000, help="Full games *per* agent matchup (default 500)")
    p.add_argument("--output", type=str, default="truco_training_data.csv", help="CSV output path")
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    args = p.parse_args()

    generate_dataset(args.num_games, args.output, args.seed)