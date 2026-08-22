from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped


# ~/shared viene aggiunto al Python path.
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from inference.policy_runtime import load_policy


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class FollowerPolicyNode(Node):

    def __init__(self, policy_path: str):
        super().__init__("follower_policy_node")

        # =========================================================
        # POLICY
        # =========================================================

        self.get_logger().info(
            f"Loading policy: {policy_path}"
        )

        self.policy = load_policy(policy_path)

        # =========================================================
        # POSE
        # =========================================================

        self.leader_pose = None
        self.follower_pose = None

        # =========================================================
        # VELOCITÀ FOLLOWER
        # =========================================================

        # Letta dall'odometria del diff_drive_controller.
        self.follower_v = 0.0
        self.follower_w = 0.0

        # =========================================================
        # VELOCITÀ LEADER
        # =========================================================

        # Il leader non viene letto tramite odometria.
        # La velocità viene stimata dalla variazione della
        # ground-truth pose.
        self.leader_v = 0.0
        self.leader_w = 0.0

        self._previous_leader_pose = None
        self._previous_leader_time = None

        # =========================================================
        # SUBSCRIBERS
        # =========================================================

        self.create_subscription(
            PoseStamped,
            "/leader/ground_truth_pose",
            self._leader_pose_cb,
            10,
        )

        self.create_subscription(
            PoseStamped,
            "/follower/ground_truth_pose",
            self._follower_pose_cb,
            10,
        )

        self.create_subscription(
            Odometry,
            "/follower/odom",
            self._follower_odom_cb,
            10,
        )

        # =========================================================
        # PUBLISHER
        # =========================================================

        self.cmd_pub = self.create_publisher(
            Twist,
            "/follower/cmd_vel",
            10,
        )

        # =========================================================
        # CONTROL LOOP
        # =========================================================

        # Il training usa dt = 0.1 s.
        # Quindi la policy viene eseguita a 10 Hz.
        self.timer = self.create_timer(
            0.1,
            self._control_loop,
        )

        self.get_logger().info(
            "Follower policy node started."
        )

        self.get_logger().info(
            "Waiting for leader/follower pose and follower odometry..."
        )

    # =============================================================
    # POSE
    # =============================================================

    @staticmethod
    def _pose_from_posestamped(
        msg: PoseStamped
    ) -> np.ndarray:

        q = msg.pose.orientation

        # Estrazione dello yaw dal quaternion.
        # Non richiede tf_transformations.
        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        theta = np.arctan2(
            siny_cosp,
            cosy_cosp,
        )

        return np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                theta,
            ],
            dtype=np.float64,
        )

    def _leader_pose_cb(
        self,
        msg: PoseStamped
    ):

        pose = self._pose_from_posestamped(msg)

        # =========================================================
        # TIMESTAMP
        # =========================================================

        stamp = msg.header.stamp

        current_time = (
            float(stamp.sec) +
            float(stamp.nanosec) * 1e-9
        )

        # =========================================================
        # PRIMA MISURA
        # =========================================================

        if (
            self._previous_leader_pose is None
            or self._previous_leader_time is None
        ):

            self._previous_leader_pose = pose.copy()
            self._previous_leader_time = current_time

            self.leader_v = 0.0
            self.leader_w = 0.0

            self.leader_pose = pose

            return

        # =========================================================
        # DT
        # =========================================================

        dt = (
            current_time -
            self._previous_leader_time
        )

        # Timestamp non valido.
        if dt <= 1e-6 or dt > 1.0:

            self.leader_v = 0.0
            self.leader_w = 0.0

            self._previous_leader_pose = pose.copy()
            self._previous_leader_time = current_time

            self.leader_pose = pose

            return

        previous = self._previous_leader_pose

        # =========================================================
        # SPOSTAMENTO LEADER
        # =========================================================

        dx_world = pose[0] - previous[0]
        dy_world = pose[1] - previous[1]

        # =========================================================
        # VELOCITÀ LINEARE LEADER
        # =========================================================

        theta_previous = previous[2]

        forward_displacement = (
            dx_world * np.cos(theta_previous)
            +
            dy_world * np.sin(theta_previous)
        )

        self.leader_v = (
            forward_displacement / dt
        )

        # =========================================================
        # VELOCITÀ ANGOLARE LEADER
        # =========================================================

        dtheta = wrap_angle(
            pose[2] - previous[2]
        )

        self.leader_w = dtheta / dt

        # =========================================================
        # SALVATAGGIO MISURA
        # =========================================================

        self._previous_leader_pose = pose.copy()
        self._previous_leader_time = current_time

        self.leader_pose = pose

    def _follower_pose_cb(
        self,
        msg: PoseStamped
    ):

        self.follower_pose = (
            self._pose_from_posestamped(msg)
        )

    # =============================================================
    # ODOMETRIA FOLLOWER
    # =============================================================

    def _follower_odom_cb(
        self,
        msg: Odometry
    ):

        self.follower_v = float(
            msg.twist.twist.linear.x
        )

        self.follower_w = float(
            msg.twist.twist.angular.z
        )

    # =============================================================
    # OBSERVATION
    # =============================================================

    def _build_observation(
        self
    ) -> np.ndarray:

        # Posizione relativa leader - follower
        # nel frame globale.
        rel = (
            self.leader_pose[:2]
            -
            self.follower_pose[:2]
        )

        # =========================================================
        # TRASFORMAZIONE NEL FRAME DEL FOLLOWER
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
        # DIFFERENZA ANGOLARE
        # =========================================================

        dtheta = wrap_angle(
            self.leader_pose[2]
            -
            self.follower_pose[2]
        )

        # =========================================================
        # DISTANZA
        # =========================================================

        dist = float(
            np.linalg.norm(rel)
        )

        # =========================================================
        # BEARING
        # =========================================================

        bearing = float(
            np.arctan2(dy, dx)
        )

        # =========================================================
        # OSSERVAZIONE
        # =========================================================

        # DEVE essere identica all'ambiente di training:
        #
        # [dx,
        #  dy,
        #  dtheta,
        #  dist,
        #  bearing,
        #  v_follower,
        #  w_follower,
        #  v_leader,
        #  w_leader]

        obs = np.array(
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

        return obs

    # =============================================================
    # CONTROL LOOP
    # =============================================================

    def _control_loop(self):

        # Non possiamo eseguire la policy finché non abbiamo
        # entrambe le pose.
        if (
            self.leader_pose is None
            or self.follower_pose is None
        ):
            return

        # =========================================================
        # OSSERVAZIONE
        # =========================================================

        obs = self._build_observation()

        # =========================================================
        # INFERENZA ONNX
        # =========================================================

        # La policy produce:
        #
        # action[0] = a_v in [-1, 1]
        # action[1] = a_w in [-1, 1]

        action = self.policy.act(obs)

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape[0] < 2:

            self.get_logger().error(
                f"Policy returned invalid action shape: "
                f"{action.shape}"
            )

            return

        # =========================================================
        # CLIPPING
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
        # MAPPING IDENTICO AL TRAINING
        # =========================================================

        # Training:
        #
        # v = 0.5 * (a_v + 1) * 0.6
        # w = a_w * 1.5
        #
        # Quindi:
        #
        # a_v = -1 -> v = 0.0
        # a_v =  0 -> v = 0.3
        # a_v = +1 -> v = 0.6
        #
        # a_w = -1 -> w = -1.5
        # a_w =  0 -> w =  0.0
        # a_w = +1 -> w = +1.5

        v = (
            0.5
            * (a_v + 1.0)
            * 0.6
        )

        w = (
            a_w
            * 1.5
        )

        # =========================================================
        # CMD_VEL
        # =========================================================

        cmd = Twist()

        cmd.linear.x = v
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = w

        self.cmd_pub.publish(cmd)

    # =============================================================
    # STOP
    # =============================================================

    def stop_robot(self):

        stop_cmd = Twist()

        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0

        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0

        self.cmd_pub.publish(stop_cmd)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy-path",
        required=True,
    )

    args, _ = parser.parse_known_args()

    # =============================================================
    # ROS2 INIT
    # =============================================================

    rclpy.init()

    node = FollowerPolicyNode(
        args.policy_path
    )

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        # =========================================================
        # STOP ROBOT
        # =========================================================

        node.stop_robot()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
