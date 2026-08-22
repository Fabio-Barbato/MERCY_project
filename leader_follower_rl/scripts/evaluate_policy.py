from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from rl_core.config import SACConfig
from rl_core.agent_factory import make_agent
from envs.leader_follower_env import LeaderFollowerEnv


# ================================================================
# EPISODE EVALUATION
# ================================================================

def evaluate_episode(agent, env, seed: int):
    """
    Esegue un singolo episodio in modalità deterministica
    e raccoglie metriche comportamentali.
    """

    obs, _ = env.reset(seed=seed)

    total_return = 0.0
    steps = 0

    distances = []
    distance_errors = []

    follower_speeds = []
    follower_omegas = []

    leader_speeds = []
    leader_omegas = []

    failed = False
    failure_step = None

    target_distance = env.follow_distance
    max_distance = env.max_lead_dist

    while True:

        # --------------------------------------------------------
        # POLICY DETERMINISTICA
        # --------------------------------------------------------

        action = agent.select_action(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        # --------------------------------------------------------
        # ENVIRONMENT
        # --------------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_return += float(reward)
        steps += 1

        # --------------------------------------------------------
        # METRICHE
        # --------------------------------------------------------

        distance = float(info["dist"])

        distances.append(distance)
        distance_errors.append(
            abs(distance - target_distance)
        )

        follower_speeds.append(
            float(env.follower_v)
        )

        follower_omegas.append(
            float(env.follower_w)
        )

        leader_speeds.append(
            float(env.leader_v)
        )

        leader_omegas.append(
            float(env.leader_w)
        )

        # --------------------------------------------------------
        # FAILURE
        # --------------------------------------------------------

        if terminated:

            failed = True
            failure_step = steps

        if terminated or truncated:
            break

    # ============================================================
    # METRICHE FINALI
    # ============================================================

    distances = np.asarray(distances)
    distance_errors = np.asarray(distance_errors)

    follower_speeds = np.asarray(follower_speeds)
    follower_omegas = np.asarray(follower_omegas)

    leader_speeds = np.asarray(leader_speeds)
    leader_omegas = np.asarray(leader_omegas)

    # ------------------------------------------------------------
    # ZONA DI FOLLOWING
    # ------------------------------------------------------------

    # Consideriamo "buona" una distanza entro ±20 cm
    # dalla distanza target.

    good_distance = (
        np.abs(distances - target_distance)
        <= 0.20
    )

    # ------------------------------------------------------------
    # ZONA DI SICUREZZA
    # ------------------------------------------------------------

    safe_distance = distances <= max_distance

    # ============================================================
    # RETURN
    # ============================================================

    return {
        "seed": seed,

        "return": total_return,

        "steps": steps,

        "failed": failed,

        "failure_step": failure_step,

        "distance_mean": float(
            np.mean(distances)
        ),

        "distance_std": float(
            np.std(distances)
        ),

        "distance_min": float(
            np.min(distances)
        ),

        "distance_max": float(
            np.max(distances)
        ),

        "distance_error_mean": float(
            np.mean(distance_errors)
        ),

        "distance_error_max": float(
            np.max(distance_errors)
        ),

        "good_distance_fraction": float(
            np.mean(good_distance)
        ),

        "safe_distance_fraction": float(
            np.mean(safe_distance)
        ),

        "follower_speed_mean": float(
            np.mean(follower_speeds)
        ),

        "follower_speed_max": float(
            np.max(follower_speeds)
        ),

        "follower_omega_mean_abs": float(
            np.mean(np.abs(follower_omegas))
        ),

        "follower_omega_max_abs": float(
            np.max(np.abs(follower_omegas))
        ),

        "leader_speed_mean": float(
            np.mean(leader_speeds)
        ),

        "leader_omega_mean_abs": float(
            np.mean(np.abs(leader_omegas))
        ),
    }


# ================================================================
# POLICY LOADING
# ================================================================

def load_agent(
    checkpoint: str,
    obs_dim: int,
    act_dim: int,
):
    """
    Ricostruisce SAC e carica il checkpoint.
    """

    cfg = SACConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    agent = make_agent(
        "sac",
        config=cfg,
    )

    agent.load(checkpoint)

    return agent


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    # ============================================================
    # ENVIRONMENT
    # ============================================================

    env = LeaderFollowerEnv(
        seed=args.seed,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # ============================================================
    # AGENT
    # ============================================================

    print()
    print("==========================================")
    print(" SAC POLICY EVALUATION")
    print("==========================================")
    print()

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    print(
        f"Observation dimension : {obs_dim}"
    )

    print(
        f"Action dimension      : {act_dim}"
    )

    print(
        f"Episodes              : {args.episodes}"
    )

    print()

    agent = load_agent(
        args.checkpoint,
        obs_dim,
        act_dim,
    )

    # ============================================================
    # EPISODES
    # ============================================================

    results = []

    for i in range(args.episodes):

        episode_seed = args.seed + i

        result = evaluate_episode(
            agent,
            env,
            episode_seed,
        )

        results.append(result)

        status = (
            "FAIL"
            if result["failed"]
            else "OK"
        )

        print(
            f"Episode {i + 1:03d} | "
            f"seed={episode_seed} | "
            f"{status:4s} | "
            f"return={result['return']:8.2f} | "
            f"distance="
            f"{result['distance_mean']:.3f} m | "
            f"good="
            f"{result['good_distance_fraction'] * 100:5.1f}%"
        )

    # ============================================================
    # AGGREGATE
    # ============================================================

    def mean(key):
        return float(
            np.mean(
                [
                    r[key]
                    for r in results
                ]
            )
        )

    def std(key):
        return float(
            np.std(
                [
                    r[key]
                    for r in results
                ]
            )
        )

    failures = sum(
        r["failed"]
        for r in results
    )

    print()
    print("==========================================")
    print(" AGGREGATE RESULTS")
    print("==========================================")
    print()

    print(
        f"Mean return              : "
        f"{mean('return'):.3f}"
    )

    print(
        f"Return std               : "
        f"{std('return'):.3f}"
    )

    print(
        f"Failure rate             : "
        f"{100.0 * failures / len(results):.1f}%"
    )

    print()

    print(
        f"Mean distance            : "
        f"{mean('distance_mean'):.3f} m"
    )

    print(
        f"Distance std             : "
        f"{mean('distance_std'):.3f} m"
    )

    print(
        f"Mean distance error      : "
        f"{mean('distance_error_mean'):.3f} m"
    )

    print(
        f"Maximum distance         : "
        f"{mean('distance_max'):.3f} m"
    )

    print(
        f"Good-distance fraction   : "
        f"{100.0 * mean('good_distance_fraction'):.1f}%"
    )

    print(
        f"Safe-distance fraction   : "
        f"{100.0 * mean('safe_distance_fraction'):.1f}%"
    )

    print()

    print(
        f"Follower mean speed      : "
        f"{mean('follower_speed_mean'):.3f} m/s"
    )

    print(
        f"Follower max speed       : "
        f"{mean('follower_speed_max'):.3f} m/s"
    )

    print(
        f"Follower mean |omega|    : "
        f"{mean('follower_omega_mean_abs'):.3f} rad/s"
    )

    print(
        f"Follower max |omega|     : "
        f"{mean('follower_omega_max_abs'):.3f} rad/s"
    )

    print()

    print(
        f"Leader mean speed        : "
        f"{mean('leader_speed_mean'):.3f} m/s"
    )

    print(
        f"Leader mean |omega|      : "
        f"{mean('leader_omega_mean_abs'):.3f} rad/s"
    )

    print()

    # ============================================================
    # SIMPLE CLASSIFICATION
    # ============================================================

    failure_rate = failures / len(results)

    good_fraction = mean(
        "good_distance_fraction"
    )

    distance_error = mean(
        "distance_error_mean"
    )

    print("==========================================")
    print(" BEHAVIORAL INTERPRETATION")
    print("==========================================")
    print()

    if failure_rate > 0.5:

        print(
            "CRITICAL: la policy perde il leader "
            "nella maggioranza degli episodi."
        )

    elif failure_rate > 0.0:

        print(
            "WARNING: la policy presenta "
            "alcuni episodi di perdita del leader."
        )

    else:

        print(
            "OK: nessun episodio ha superato "
            "la distanza massima."
        )

    if good_fraction > 0.8:

        print(
            "Tracking della distanza: BUONO."
        )

    elif good_fraction > 0.5:

        print(
            "Tracking della distanza: DISCRETO."
        )

    else:

        print(
            "Tracking della distanza: SCARSO."
        )

    print(
        f"Errore medio distanza: "
        f"{distance_error:.3f} m"
    )

    print()
    print("Evaluation completata.")


if __name__ == "__main__":
    main()