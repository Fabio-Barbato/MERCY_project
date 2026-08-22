from __future__ import annotations

import argparse
import os
import sys

import numpy as np

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
# UTILS
# ================================================================

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def load_agent(checkpoint, obs_dim, act_dim):

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
# LEADER SCENARIOS
# ================================================================

def leader_command(
    env,
    scenario,
    step,
):
    """
    Restituisce (v, w) desiderati dal leader.

    Il follower rimane sempre controllato dalla policy SAC.

    NOTA:
    questa funzione NON modifica il training environment.
    Serve esclusivamente per stressare la policy durante evaluation.
    """

    t = step * env.dt

    # ------------------------------------------------------------
    # 1. STRAIGHT
    # ------------------------------------------------------------

    if scenario == "straight":

        v = 0.45
        w = 0.0

    # ------------------------------------------------------------
    # 2. CURVE
    # ------------------------------------------------------------

    elif scenario == "curves":

        v = 0.45
        w = 0.45 * np.sin(0.45 * t)

    # ------------------------------------------------------------
    # 3. SPEED CHANGES
    # ------------------------------------------------------------

    elif scenario == "speed_changes":

        period = 8.0

        phase = t % period

        if phase < 2.0:
            v = 0.20

        elif phase < 4.0:
            v = 0.55

        elif phase < 6.0:
            v = 0.35

        else:
            v = 0.60

        w = 0.15 * np.sin(0.5 * t)

    # ------------------------------------------------------------
    # 4. AGGRESSIVE TURNS
    # ------------------------------------------------------------

    elif scenario == "aggressive_turns":

        v = 0.40

        period = 5.0

        phase = t % period

        if phase < 1.5:
            w = 0.0

        elif phase < 2.0:
            w = 1.2

        elif phase < 3.5:
            w = 0.0

        elif phase < 4.0:
            w = -1.2

        else:
            w = 0.0

    # ------------------------------------------------------------
    # 5. MIXED
    # ------------------------------------------------------------

    elif scenario == "mixed":

        # Combinazione di variazioni di velocità,
        # curve e cambi di direzione.

        v = (
            0.40
            + 0.15 * np.sin(0.35 * t)
            + 0.08 * np.sin(0.9 * t)
        )

        w = (
            0.55 * np.sin(0.55 * t)
            + 0.25 * np.sin(1.1 * t)
        )

    else:

        raise ValueError(
            f"Scenario sconosciuto: {scenario}"
        )

    v = float(
        np.clip(
            v,
            0.0,
            env.leader_max_speed,
        )
    )

    w = float(
        np.clip(
            w,
            -1.5,
            1.5,
        )
    )

    return v, w


# ================================================================
# INITIAL CONDITIONS
# ================================================================

def initialize_episode(
    env,
    scenario,
):
    """
    Imposta le condizioni iniziali del follower.

    Per gli scenari normali partiamo dalla configurazione standard.

    Per recovery partiamo deliberatamente lontani.
    """

    if scenario == "recovery":

        env.leader_pose = np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        env.follower_pose = np.array(
            [-3.0, -1.0, 0.0],
            dtype=np.float64,
        )

    else:

        env.leader_pose = np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        env.follower_pose = np.array(
            [-env.follow_distance, 0.0, 0.0],
            dtype=np.float64,
        )

    env.leader_v = 0.0
    env.leader_w = 0.0

    env.follower_v = 0.0
    env.follower_w = 0.0

    env.step_count = 0


# ================================================================
# EPISODE
# ================================================================

def run_episode(
    agent,
    env,
    scenario,
    seed,
):
    """
    Esegue un episodio completo.
    """

    env.reset(seed=seed)

    initialize_episode(
        env,
        scenario,
    )

    obs = env._get_obs()

    total_return = 0.0

    distances = []
    distance_errors = []

    follower_speeds = []
    follower_omegas = []

    leader_speeds = []
    leader_omegas = []

    failure_step = None

    for step in range(env.max_steps):

        # ========================================================
        # LEADER
        # ========================================================

        if scenario == "recovery":

            # Nel recovery usiamo un comportamento moderato
            # ma continuo.

            leader_v = 0.45
            leader_w = 0.20 * np.sin(0.5 * step * env.dt)

        else:

            leader_v, leader_w = leader_command(
                env,
                scenario,
                step,
            )

        env.leader_v = leader_v
        env.leader_w = leader_w

        # --------------------------------------------------------
        # LEADER MOTION
        # --------------------------------------------------------

        env.leader_pose[2] = wrap_angle(
            env.leader_pose[2]
            + leader_w * env.dt
        )

        env.leader_pose[0] += (
            leader_v
            * np.cos(env.leader_pose[2])
            * env.dt
        )

        env.leader_pose[1] += (
            leader_v
            * np.sin(env.leader_pose[2])
            * env.dt
        )

        # ========================================================
        # SAC
        # ========================================================

        action = agent.select_action(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        # ========================================================
        # FOLLOWER
        # ========================================================

        env._drive_follower(action)

        env.step_count += 1

        # ========================================================
        # OBSERVATION
        # ========================================================

        obs = env._get_obs()

        dist = float(obs[3])

        reward = (
            1.0
            - abs(dist - env.follow_distance)
            - 0.05 * abs(env.follower_w)
        )

        # ========================================================
        # FAILURE
        # ========================================================

        if dist > env.max_lead_dist:

            reward -= 10.0

            failure_step = step + 1

            total_return += reward

            distances.append(dist)
            distance_errors.append(
                abs(dist - env.follow_distance)
            )

            follower_speeds.append(env.follower_v)
            follower_omegas.append(env.follower_w)

            leader_speeds.append(env.leader_v)
            leader_omegas.append(env.leader_w)

            break

        # ========================================================
        # METRICS
        # ========================================================

        total_return += reward

        distances.append(dist)

        distance_errors.append(
            abs(dist - env.follow_distance)
        )

        follower_speeds.append(
            env.follower_v
        )

        follower_omegas.append(
            env.follower_w
        )

        leader_speeds.append(
            env.leader_v
        )

        leader_omegas.append(
            env.leader_w
        )

    # ============================================================
    # ARRAYS
    # ============================================================

    distances = np.asarray(distances)

    distance_errors = np.asarray(
        distance_errors
    )

    follower_speeds = np.asarray(
        follower_speeds
    )

    follower_omegas = np.asarray(
        follower_omegas
    )

    leader_speeds = np.asarray(
        leader_speeds
    )

    leader_omegas = np.asarray(
        leader_omegas
    )

    # ============================================================
    # METRICS
    # ============================================================

    good_distance = (
        np.abs(
            distances - env.follow_distance
        )
        <= 0.20
    )

    return {
        "return": float(total_return),

        "steps": len(distances),

        "failed": failure_step is not None,

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

        "good_fraction": float(
            np.mean(good_distance)
        ),

        "follower_speed_mean": float(
            np.mean(follower_speeds)
        ),

        "follower_speed_max": float(
            np.max(follower_speeds)
        ),

        "follower_omega_mean_abs": float(
            np.mean(
                np.abs(follower_omegas)
            )
        ),

        "follower_omega_max_abs": float(
            np.max(
                np.abs(follower_omegas)
            )
        ),

        "leader_speed_mean": float(
            np.mean(leader_speeds)
        ),

        "leader_omega_mean_abs": float(
            np.mean(
                np.abs(leader_omegas)
            )
        ),
    }


# ================================================================
# SCENARIO EVALUATION
# ================================================================

def evaluate_scenario(
    agent,
    env,
    scenario,
    episodes,
    seed,
):

    results = []

    print()
    print("------------------------------------------")
    print(f" SCENARIO: {scenario.upper()}")
    print("------------------------------------------")

    for i in range(episodes):

        result = run_episode(
            agent,
            env,
            scenario,
            seed + i,
        )

        results.append(result)

        status = (
            "FAIL"
            if result["failed"]
            else "OK"
        )

        print(
            f"Episode {i + 1:02d} | "
            f"{status:4s} | "
            f"return={result['return']:8.2f} | "
            f"dist={result['distance_mean']:.3f} | "
            f"good={result['good_fraction'] * 100:5.1f}%"
        )

    # ============================================================
    # AGGREGATION
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

    failures = sum(
        r["failed"]
        for r in results
    )

    print()

    print(
        f"Return mean           : "
        f"{mean('return'):.3f}"
    )

    print(
        f"Distance mean         : "
        f"{mean('distance_mean'):.3f} m"
    )

    print(
        f"Distance error        : "
        f"{mean('distance_error_mean'):.3f} m"
    )

    print(
        f"Distance max          : "
        f"{mean('distance_max'):.3f} m"
    )

    print(
        f"Good distance         : "
        f"{mean('good_fraction') * 100:.1f}%"
    )

    print(
        f"Failure rate          : "
        f"{100.0 * failures / episodes:.1f}%"
    )

    print(
        f"Follower mean speed   : "
        f"{mean('follower_speed_mean'):.3f} m/s"
    )

    print(
        f"Follower max omega    : "
        f"{mean('follower_omega_max_abs'):.3f} rad/s"
    )

    return results


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
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    env = LeaderFollowerEnv(
        seed=args.seed,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = load_agent(
        args.checkpoint,
        obs_dim,
        act_dim,
    )

    scenarios = [
        "straight",
        "curves",
        "speed_changes",
        "aggressive_turns",
        "recovery",
        "mixed",
    ]

    print()
    print("==========================================")
    print(" SAC GENERALIZATION EVALUATION")
    print("==========================================")
    print()

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    print(
        f"Episodes per scenario : "
        f"{args.episodes}"
    )

    print()

    all_results = {}

    for scenario in scenarios:

        all_results[scenario] = evaluate_scenario(
            agent,
            env,
            scenario,
            args.episodes,
            args.seed,
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()
    print("==========================================")
    print(" FINAL SUMMARY")
    print("==========================================")
    print()

    print(
        f"{'Scenario':20s}"
        f"{'Return':>10s}"
        f"{'Dist':>10s}"
        f"{'Error':>10s}"
        f"{'Good':>10s}"
        f"{'Fail':>10s}"
    )

    print("-" * 70)

    for scenario, results in all_results.items():

        def m(key):
            return np.mean(
                [
                    r[key]
                    for r in results
                ]
            )

        failure_rate = (
            np.mean(
                [
                    r["failed"]
                    for r in results
                ]
            )
            * 100.0
        )

        print(
            f"{scenario:20s}"
            f"{m('return'):10.2f}"
            f"{m('distance_mean'):10.3f}"
            f"{m('distance_error_mean'):10.3f}"
            f"{m('good_fraction') * 100:9.1f}%"
            f"{failure_rate:9.1f}%"
        )

    print()
    print("Evaluation completata.")


if __name__ == "__main__":
    main()