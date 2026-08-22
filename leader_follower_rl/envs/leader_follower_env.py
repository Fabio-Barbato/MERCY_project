from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LeaderFollowerEnv(gym.Env):
    """
    Simulazione 2D per l'addestramento SAC del follower.

    Il leader viene mosso da un semplice generatore di waypoint.

    Il follower è controllato da SAC con azione:

        [a_v, a_w] ∈ [-1, 1]^2

    L'azione viene convertita in:

        v ∈ [0, follower_max_speed]
        w ∈ [-follower_max_omega, follower_max_omega]

    Observation:

        [dx,
         dy,
         dtheta,
         dist,
         bearing,
         v_follower,
         w_follower,
         v_leader,
         w_leader]

    Le osservazioni sono esclusivamente relative/propriocettive,
    così da poter essere ricostruite successivamente tramite
    sensori o comunicazione robot-robot.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 500,
        follow_distance: float = 1.0,
        max_lead_dist: float = 5.0,
        leader_max_speed: float = 0.5,
        follower_max_speed: float = 0.8,
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

        # =========================================================
        # OBSERVATION SPACE
        # =========================================================

        # [dx,
        #  dy,
        #  dtheta,
        #  dist,
        #  bearing,
        #  v_follower,
        #  w_follower,
        #  v_leader,
        #  w_leader]

        obs_dim = 9

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # =========================================================
        # ACTION SPACE
        # =========================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # =========================================================
        # RANDOM GENERATOR
        # =========================================================

        self._rng = np.random.default_rng(seed)

        # =========================================================
        # RESET
        # =========================================================

        self.reset(seed=seed)

    # =============================================================
    # WAYPOINT
    # =============================================================

    def _sample_waypoint(self):
        """
        Genera un waypoint casuale per il leader.
        """

        r = self._rng.uniform(
            2.0,
            5.0,
        )

        ang = self._rng.uniform(
            -np.pi,
            np.pi,
        )

        return (
            self.leader_pose[:2]
            +
            r
            *
            np.array(
                [
                    np.cos(ang),
                    np.sin(ang),
                ]
            )
        )

    # =============================================================
    # RESET
    # =============================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        Reset dell'ambiente.

        Il follower può partire:

        - vicino alla distanza desiderata;
        - a distanza intermedia;
        - in una situazione di recovery.

        Questo permette a SAC di vedere durante il training
        anche situazioni nelle quali deve recuperare il leader.
        """

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0

        # =========================================================
        # LEADER
        # =========================================================

        self.leader_pose = np.array(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        # =========================================================
        # FOLLOWER INITIAL DISTANCE
        # =========================================================

        # 50% degli episodi:
        # situazione normale.

        # 50% degli episodi:
        # situazione di recovery.

        if self._rng.random() < 0.5:

            initial_distance = self._rng.uniform(
                0.8,
                1.5,
            )

        else:

            initial_distance = self._rng.uniform(
                1.5,
                3.0,
            )

        # =========================================================
        # FOLLOWER INITIAL ANGLE
        # =========================================================

        initial_angle = self._rng.uniform(
            -np.pi,
            np.pi,
        )

        self.follower_pose = np.array(
            [
                -initial_distance * np.cos(initial_angle),
                -initial_distance * np.sin(initial_angle),
                0.0,
            ],
            dtype=np.float64,
        )

        # =========================================================
        # FOLLOWER INITIAL ORIENTATION
        # =========================================================

        dx = (
            self.leader_pose[0]
            -
            self.follower_pose[0]
        )

        dy = (
            self.leader_pose[1]
            -
            self.follower_pose[1]
        )

        self.follower_pose[2] = np.arctan2(
            dy,
            dx,
        )

        # =========================================================
        # INITIAL VELOCITIES
        # =========================================================

        # Il leader parte già in movimento.
        #
        # Questo rende il recovery più realistico:
        # il follower deve recuperare un leader che continua
        # ad avanzare.

        self.leader_v = self._rng.uniform(
            0.2,
            self.leader_max_speed,
        )

        self.leader_w = self._rng.uniform(
            -0.3,
            0.3,
        )

        self.follower_v = 0.0
        self.follower_w = 0.0

        # =========================================================
        # WAYPOINT
        # =========================================================

        self.waypoint = self._sample_waypoint()

        # =========================================================
        # INITIAL DISTANCE
        # =========================================================

        rel = (
            self.leader_pose[:2]
            -
            self.follower_pose[:2]
        )

        self.previous_dist = float(
            np.linalg.norm(rel)
        )

        # previous_dist viene mantenuto solamente come stato
        # diagnostico. NON viene utilizzato nel reward.

        # =========================================================
        # OBSERVATION
        # =========================================================

        return self._get_obs(), {}

    # =============================================================
    # LEADER
    # =============================================================

    def _drive_leader(self):
        """
        Movimento del leader verso il waypoint corrente.
        """

        to_wp = (
            self.waypoint
            -
            self.leader_pose[:2]
        )

        dist = np.linalg.norm(to_wp)

        # Nuovo waypoint.

        if dist < 0.3:

            self.waypoint = self._sample_waypoint()

            to_wp = (
                self.waypoint
                -
                self.leader_pose[:2]
            )

        # =========================================================
        # TARGET HEADING
        # =========================================================

        target_theta = np.arctan2(
            to_wp[1],
            to_wp[0],
        )

        heading_err = wrap_angle(
            target_theta
            -
            self.leader_pose[2]
        )

        # =========================================================
        # ANGULAR VELOCITY
        # =========================================================

        self.leader_w = np.clip(
            2.0 * heading_err,
            -1.0,
            1.0,
        )

        # =========================================================
        # LINEAR VELOCITY
        # =========================================================

        self.leader_v = (
            self.leader_max_speed
            *
            max(
                0.0,
                np.cos(heading_err),
            )
        )

        # =========================================================
        # UPDATE POSE
        # =========================================================

        self.leader_pose[2] = wrap_angle(
            self.leader_pose[2]
            +
            self.leader_w * self.dt
        )

        self.leader_pose[0] += (
            self.leader_v
            *
            np.cos(self.leader_pose[2])
            *
            self.dt
        )

        self.leader_pose[1] += (
            self.leader_v
            *
            np.sin(self.leader_pose[2])
            *
            self.dt
        )

    # =============================================================
    # FOLLOWER
    # =============================================================

    def _drive_follower(
        self,
        action,
    ):
        """
        Applica l'azione SAC al follower.
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        # =========================================================
        # ACTION CLIPPING
        # =========================================================

        a_v = float(
            np.clip(
                action[0],
                -1.0,
                1.0,
            )
        )

        a_w = float(
            np.clip(
                action[1],
                -1.0,
                1.0,
            )
        )

        # =========================================================
        # ACTION -> VELOCITY
        # =========================================================

        # a_v = -1 -> v = 0
        # a_v =  0 -> v = 0.4
        # a_v = +1 -> v = 0.8

        v = (
            0.5
            *
            (a_v + 1.0)
            *
            self.follower_max_speed
        )

        # a_w = -1 -> w = -1.5
        # a_w =  0 -> w = 0
        # a_w = +1 -> w = +1.5

        w = (
            a_w
            *
            self.follower_max_omega
        )

        self.follower_v = v
        self.follower_w = w

        # =========================================================
        # UPDATE ORIENTATION
        # =========================================================

        self.follower_pose[2] = wrap_angle(
            self.follower_pose[2]
            +
            w * self.dt
        )

        # =========================================================
        # UPDATE POSITION
        # =========================================================

        self.follower_pose[0] += (
            v
            *
            np.cos(self.follower_pose[2])
            *
            self.dt
        )

        self.follower_pose[1] += (
            v
            *
            np.sin(self.follower_pose[2])
            *
            self.dt
        )

    # =============================================================
    # OBSERVATION
    # =============================================================

    def _get_obs(self):
        """
        Costruisce l'osservazione nel frame del follower.
        """

        # =========================================================
        # RELATIVE POSITION
        # =========================================================

        rel = (
            self.leader_pose[:2]
            -
            self.follower_pose[:2]
        )

        # =========================================================
        # WORLD -> FOLLOWER FRAME
        # =========================================================

        c = np.cos(
            -self.follower_pose[2]
        )

        s = np.sin(
            -self.follower_pose[2]
        )

        dx = (
            c * rel[0]
            -
            s * rel[1]
        )

        dy = (
            s * rel[0]
            +
            c * rel[1]
        )

        # =========================================================
        # ANGULAR ERROR
        # =========================================================

        dtheta = wrap_angle(
            self.leader_pose[2]
            -
            self.follower_pose[2]
        )

        # =========================================================
        # DISTANCE
        # =========================================================

        dist = float(
            np.linalg.norm(rel)
        )

        # =========================================================
        # BEARING
        # =========================================================

        bearing = float(
            np.arctan2(
                dy,
                dx,
            )
        )

        # =========================================================
        # OBSERVATION
        # =========================================================

        return np.array(
            [
                dx,
                dy,
                dtheta,
                dist,
                bearing,
                self.follower_v,
                self.follower_w,
                self.leader_v,
                self.leader_w,
            ],
            dtype=np.float32,
        )

    # =============================================================
    # STEP
    # =============================================================

    def step(self, action):

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # =========================================================
        # 1. LEADER MOVEMENT
        # =========================================================

        self._drive_leader()

        # =========================================================
        # 2. FOLLOWER MOVEMENT
        # =========================================================

        self._drive_follower(
            action
        )

        self.step_count += 1

        # =========================================================
        # 3. OBSERVATION
        # =========================================================

        obs = self._get_obs()

        dist = float(
            obs[3]
        )

        # =========================================================
        # 4. DISTANCE ERROR
        # =========================================================

        dist_err = abs(
            dist
            -
            self.follow_distance
        )

        # =========================================================
        # 5. REWARD
        # =========================================================

        # Reward originale.
        #
        # Non utilizziamo il progress reward perché il test
        # precedente ha dimostrato che peggiora drasticamente
        # il comportamento di recovery.

        reward = (
            1.0
            -
            dist_err
            -
            0.05 * abs(
                self.follower_w
            )
        )

        # =========================================================
        # 6. TERMINATION
        # =========================================================

        terminated = (
            dist
            >
            self.max_lead_dist
        )

        if terminated:
            reward -= 10.0

        # =========================================================
        # 7. TIME LIMIT
        # =========================================================

        truncated = (
            self.step_count
            >=
            self.max_steps
        )

        # =========================================================
        # 8. DIAGNOSTIC INFO
        # =========================================================

        info = {
            "dist": dist,
            "dist_error": dist_err,
            "leader_v": self.leader_v,
            "leader_w": self.leader_w,
            "follower_v": self.follower_v,
            "follower_w": self.follower_w,
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )