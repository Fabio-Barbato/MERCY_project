import numpy as np

from rl_core.config import SACConfig
from rl_core.agent_factory import make_agent
from envs.leader_follower_env import LeaderFollowerEnv


NUM_EPISODES = 20


def main():
    env = LeaderFollowerEnv(seed=1000)

    cfg = SACConfig(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
    )

    agent = make_agent("sac", config=cfg)
    agent.load("checkpoints/best.pt")

    returns = []
    distances = []
    max_distances = []
    failures = []
    action_values = []

    print("=== VALUTAZIONE POLICY ===")
    print(f"Episodi: {NUM_EPISODES}")
    print()

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset(seed=1000 + episode)

        episode_return = 0.0
        episode_distances = []
        episode_actions = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(
                obs,
                deterministic=True,
            )

            action = np.asarray(action, dtype=np.float32)

            episode_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            episode_distances.append(float(info["dist"]))

        returns.append(episode_return)
        distances.extend(episode_distances)
        max_distances.append(max(episode_distances))
        failures.append(terminated)
        action_values.extend(episode_actions)

        print(
            f"Episodio {episode + 1:2d}: "
            f"return={episode_return:8.2f} | "
            f"dist media={np.mean(episode_distances):.3f} m | "
            f"dist finale={episode_distances[-1]:.3f} m | "
            f"dist max={max(episode_distances):.3f} m | "
            f"{'FAIL' if terminated else 'OK'}"
        )

    actions = np.asarray(action_values)

    print()
    print("=== RISULTATI GLOBALI ===")

    print(f"Return medio       : {np.mean(returns):.2f}")
    print(f"Return std         : {np.std(returns):.2f}")

    print(f"Distanza media     : {np.mean(distances):.3f} m")
    print(f"Distanza std       : {np.std(distances):.3f} m")

    print(f"Distanza massima   : {np.max(distances):.3f} m")
    print(f"Max medio episodio : {np.mean(max_distances):.3f} m")

    print(
        f"Episodi terminati  : "
        f"{sum(failures)}/{NUM_EPISODES}"
    )

    print()
    print("=== AZIONI ===")

    print(
        f"a_v min/max : "
        f"{actions[:, 0].min():.3f} / "
        f"{actions[:, 0].max():.3f}"
    )

    print(
        f"a_w min/max : "
        f"{actions[:, 1].min():.3f} / "
        f"{actions[:, 1].max():.3f}"
    )

    # Conversione alle grandezze fisiche
    v = 0.5 * (actions[:, 0] + 1.0) * 0.6
    w = actions[:, 1] * 1.5

    print()
    print("=== COMANDI FISICI ===")

    print(
        f"v min/max : "
        f"{v.min():.3f} / {v.max():.3f} m/s"
    )

    print(
        f"w min/max : "
        f"{w.min():.3f} / {w.max():.3f} rad/s"
    )

    print()
    print("Test completato.")


if __name__ == "__main__":
    main()
