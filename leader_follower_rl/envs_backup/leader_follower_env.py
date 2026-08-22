from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LeaderFollowerEnv(gym.Env):
    """
    Simulazione 2D leggerissima (nessuna dipendenza da ROS2/Webots) usata
    per addestrare la policy del follower.

    Il leader viene mosso da un semplice generatore di waypoint casuali
    con pure-pursuit grezzo (stand-in per il comportamento di Nav2 sulla
    mappa reale). Il follower e' controllato dall'agente RL con azione
    [v, w]: la STESSA interfaccia di un cmd_vel differenziale, per non
    dover tradurre nulla al momento del deploy.

    L'osservazione contiene SOLO grandezze relative + propriocezione
    (mai coordinate assolute nella mappa), cosi' da poter essere
    ricostruita identica sia da odometria ROS2 sia da GPS/bussola Webots.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 500,
        follow_distance: float = 1.0,
        max_lead_dist: float = 5.0,
        leader_max_speed: float = 0.5,
        follower_max_speed: float = 0.6,
        follower_max_omega: float = 1.5,
        seed: int | None = None,
    ):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.follow_distance = follow_distance
        self.max_lead_dist = max_lead_dist
        self.leader_max_speed = leader_max_speed
        self.follower_max_speed = follower_max_speed
        self.follower_max_omega = follower_max_omega

        # osservazione: [dx, dy, dtheta, dist, bearing, v_follower, w_follower, v_leader, w_leader]
        obs_dim = 9
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -follower_max_omega], dtype=np.float32),
            high=np.array([follower_max_speed, follower_max_omega], dtype=np.float32),
        )

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def _sample_waypoint(self):
        r = self._rng.uniform(2.0, 5.0)
        ang = self._rng.uniform(-np.pi, np.pi)
        return self.leader_pose[:2] + r * np.array([np.cos(ang), np.sin(ang)])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.leader_pose = np.array([0.0, 0.0, 0.0])       # x, y, theta
        self.follower_pose = np.array([-self.follow_distance, 0.0, 0.0])
        self.leader_v, self.leader_w = 0.0, 0.0
        self.follower_v, self.follower_w = 0.0, 0.0
        self.waypoint = self._sample_waypoint()
        return self._get_obs(), {}

    def _drive_leader(self):
        # pure-pursuit grezzo verso il waypoint corrente (stand-in per Nav2)
        to_wp = self.waypoint - self.leader_pose[:2]
        dist = np.linalg.norm(to_wp)
        if dist < 0.3:
            self.waypoint = self._sample_waypoint()
            to_wp = self.waypoint - self.leader_pose[:2]
            dist = np.linalg.norm(to_wp)

        target_theta = np.arctan2(to_wp[1], to_wp[0])
        heading_err = wrap_angle(target_theta - self.leader_pose[2])
        self.leader_w = np.clip(2.0 * heading_err, -1.0, 1.0)
        self.leader_v = self.leader_max_speed * max(0.0, np.cos(heading_err))

        self.leader_pose[2] = wrap_angle(self.leader_pose[2] + self.leader_w * self.dt)
        self.leader_pose[0] += self.leader_v * np.cos(self.leader_pose[2]) * self.dt
        self.leader_pose[1] += self.leader_v * np.sin(self.leader_pose[2]) * self.dt

    def _drive_follower(self, action):
        v, w = float(action[0]), float(action[1])
        self.follower_v, self.follower_w = v, w
        self.follower_pose[2] = wrap_angle(self.follower_pose[2] + w * self.dt)
        self.follower_pose[0] += v * np.cos(self.follower_pose[2]) * self.dt
        self.follower_pose[1] += v * np.sin(self.follower_pose[2]) * self.dt

    def _get_obs(self):
        rel = self.leader_pose[:2] - self.follower_pose[:2]
        c, s = np.cos(-self.follower_pose[2]), np.sin(-self.follower_pose[2])
        dx = c * rel[0] - s * rel[1]
        dy = s * rel[0] + c * rel[1]
        dtheta = wrap_angle(self.leader_pose[2] - self.follower_pose[2])
        dist = np.linalg.norm(rel)
        bearing = np.arctan2(dy, dx)
        return np.array([dx, dy, dtheta, dist, bearing,
                          self.follower_v, self.follower_w,
                          self.leader_v, self.leader_w], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._drive_leader()
        self._drive_follower(action)
        self.step_count += 1

        obs = self._get_obs()
        dist = float(obs[3])

        # reward: stare vicino alla distanza target, penalizzare sterzate brusche
        dist_err = abs(dist - self.follow_distance)
        reward = 1.0 - dist_err - 0.05 * abs(self.follower_w)

        terminated = dist > self.max_lead_dist
        if terminated:
            reward -= 10.0
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {"dist": dist}
