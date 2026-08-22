#!/usr/bin/env python3
"""
sac_follower_node.py

Nodo ROS2 che carica la policy SAC addestrata in MERCY
e controlla il TurtleBot3 follower in Webots.

Sottoscrizioni:
    /odom           (nav_msgs/Odometry) - posizione del leader
    /follower/odom  (nav_msgs/Odometry) - posizione del follower

Pubblicazioni:
    /follower/cmd_vel (geometry_msgs/Twist) - comandi al follower

Parametri ROS2:
    model_path          : percorso del file .pt dell'actor SAC
    leader_odom_topic   : default '/odom'
    follower_odom_topic : default '/follower/odom'
    cmd_vel_topic       : default '/follower/cmd_vel'
    catch_threshold     : distanza (m) per considerare il leader "raggiunto"
    max_linear_vel      : velocita lineare massima (m/s), default 0.22
    max_angular_vel     : velocita angolare massima (rad/s), default 2.84
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# ---------------------------------------------------------------------------
# Import PyTorch — Actor definito solo se disponibile
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class Actor(nn.Module):
        """
        Identica alla classe Actor in algorithms/sac_continuous.py.
        Deve essere mantenuta sincronizzata con quella di training.
        """
        def __init__(self, state_dim=3, action_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128),       nn.ReLU(),
            )
            self.mean_head    = nn.Linear(128, action_dim)
            self.log_std_head = nn.Linear(128, action_dim)

        def forward(self, x):
            h       = self.net(x)
            mean    = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(-20, 2)
            return mean, log_std

        def deterministic(self, state):
            mean, _ = self(state)
            return torch.tanh(mean)

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    Actor = None


# ---------------------------------------------------------------------------
# Nodo ROS2
# ---------------------------------------------------------------------------

class SACFollowerNode(Node):

    def __init__(self):
        super().__init__('sac_follower')

        # Parametri
        self.declare_parameter('model_path',          '/home/ubuntu/shared/sac_actor.pt')
        self.declare_parameter('leader_odom_topic',   '/odom')
        self.declare_parameter('follower_odom_topic', '/follower/odom')
        self.declare_parameter('cmd_vel_topic',       '/follower/cmd_vel')
        self.declare_parameter('catch_threshold',     0.3)
        self.declare_parameter('max_linear_vel',      0.22)
        self.declare_parameter('max_angular_vel',     2.84)

        model_path           = self.get_parameter('model_path').value
        leader_odom_topic    = self.get_parameter('leader_odom_topic').value
        follower_odom_topic  = self.get_parameter('follower_odom_topic').value
        cmd_vel_topic        = self.get_parameter('cmd_vel_topic').value
        self.catch_threshold = self.get_parameter('catch_threshold').value
        self.max_linear_vel  = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value

        # Stato interno
        self.leader_pos   = None
        self.follower_pos = None
        self.catches      = 0
        self.steps        = 0

        # Carica policy SAC
        self.actor = None
        if TORCH_AVAILABLE:
            self._load_model(model_path)
        else:
            self.get_logger().error(
                'PyTorch non disponibile. Installa con: pip3 install torch'
            )

        # Subscriber
        self.create_subscription(Odometry, leader_odom_topic,   self._leader_cb,   10)
        self.create_subscription(Odometry, follower_odom_topic, self._follower_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Timer di controllo: 10 Hz
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'SAC Follower Node avviato.\n'
            f'  Leader:   {leader_odom_topic}\n'
            f'  Follower: {follower_odom_topic}\n'
            f'  Cmd:      {cmd_vel_topic}\n'
            f'  Modello:  {model_path}\n'
            f'  Torch:    {"OK" if TORCH_AVAILABLE else "NON DISPONIBILE"}'
        )

    # ------------------------------------------------------------------

    def _load_model(self, model_path):
        try:
            self.actor = Actor(state_dim=3, action_dim=2)
            state_dict = torch.load(model_path, map_location='cpu')
            self.actor.load_state_dict(state_dict)
            self.actor.eval()
            self.get_logger().info(f'Modello SAC caricato da: {model_path}')
        except FileNotFoundError:
            self.get_logger().warn(
                f'File modello non trovato: {model_path}\n'
                f'Uso fallback geometrico. Salva i pesi con:\n'
                f'  torch.save(actor.state_dict(), "{model_path}")'
            )
            self.actor = None
        except Exception as e:
            self.get_logger().error(f'Errore caricamento modello: {e}')
            self.actor = None

    # ------------------------------------------------------------------

    def _leader_cb(self, msg):
        self.leader_pos = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    def _follower_cb(self, msg):
        self.follower_pos = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    # ------------------------------------------------------------------

    def _compute_obs(self):
        fx, fy = self.follower_pos
        lx, ly = self.leader_pos

        dx_raw = lx - fx
        dy_raw = ly - fy
        dist   = math.sqrt(dx_raw**2 + dy_raw**2)

        # Normalizzazione rispetto alla diagonale dell'arena (5x10m ~ 11.2m)
        ARENA_DIAG = 11.2
        dx        = dx_raw / ARENA_DIAG
        dy        = dy_raw / ARENA_DIAG
        dist_norm = dist   / ARENA_DIAG

        return np.array([dx, dy, dist_norm], dtype=np.float32), dist

    def _action_to_twist(self, vx, vy):
        """
        Converte azione SAC [vx, vy] in Twist per robot differenziale.
        vx -> velocita lineare (avanti/indietro)
        vy -> velocita angolare (sinistra/destra)
        """
        twist = Twist()
        twist.linear.x  = float(vx) * self.max_linear_vel
        twist.angular.z = -float(vy) * self.max_angular_vel
        return twist

    # ------------------------------------------------------------------

    def _control_loop(self):
        if self.leader_pos is None or self.follower_pos is None:
            return

        obs, dist = self._compute_obs()
        self.steps += 1

        if dist < self.catch_threshold:
            self.catches += 1
            self.get_logger().info(
                f'CATTURA #{self.catches} | dist={dist:.3f}m | step={self.steps}'
            )

        twist = Twist()

        if self.actor is not None:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.actor.deterministic(obs_t).squeeze(0).numpy()
            twist = self._action_to_twist(float(action[0]), float(action[1]))
        else:
            # Fallback geometrico: punta verso il leader e avanza
            fx, fy = self.follower_pos
            lx, ly = self.leader_pos
            twist.linear.x  = min(dist * 0.5, self.max_linear_vel)
            angle            = math.atan2(ly - fy, lx - fx)
            twist.angular.z  = max(-self.max_angular_vel,
                                   min(self.max_angular_vel, angle))
            if self.steps % 50 == 0:
                self.get_logger().warn(
                    f'Fallback geometrico attivo. Dist: {dist:.2f}m'
                )

        self.cmd_pub.publish(twist)

        if self.steps % 100 == 0:
            self.get_logger().info(
                f'Step {self.steps} | Dist: {dist:.2f}m | '
                f'Catture: {self.catches} | '
                f'v={twist.linear.x:.2f} w={twist.angular.z:.2f}'
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SACFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Nodo terminato. Catture: {node.catches} / {node.steps} step'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
