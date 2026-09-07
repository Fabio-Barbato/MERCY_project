# MERCY — Leader-Follower Reinforcement Learning for Mobile Robots

MERCY is a research project that trains a **Soft Actor-Critic (SAC)** reinforcement learning agent to make a follower robot track a leader robot, then deploys the trained policy onto a **ROS2 + Webots** simulated (and eventually real) TurtleBot3 setup.

The project is deliberately split into two independent halves:

1. **`leader_follower_rl/`** — a self-contained RL package (training, evaluation, policy export). It has no dependency on ROS2 or Webots and can be developed and iterated on quickly, on any machine.
2. **`ros2_package/`** — the deployment side: ROS2 launch files, a pose-supervisor package, and the runtime node that loads an exported policy and drives the follower robot inside Webots.

This separation means the RL algorithm can be trained and evaluated in a lightweight 2D simulation in minutes, while the robotics integration (Nav2, Webots, ROS2 topics) is handled independently and only consumes the final exported policy (`.pt` / `.onnx`).

## How it works

```
TRAINING            2D Gymnasium env → SAC agent → policy checkpoint
EXPORT              checkpoint → TorchScript (.pt) / ONNX (.onnx)
DEPLOY              policy → inference runtime → ROS2 node → Webots / real robot
```

- The **leader** follows random waypoints using pure-pursuit logic (a stand-in for Nav2 behavior).
- The **follower** is a unicycle-kinematics robot controlled by the RL policy, which outputs a `[v, w]` action equivalent to a `cmd_vel` command.
- Once trained, the policy is exported and used by a ROS2 node that translates real robot observations into the format the policy expects — the policy itself never depends on ROS2 or Webots.

## Project structure

```
MERCY_project/
├── leader_follower_rl/        # RL training/eval/export (no ROS2/Webots dependency)
│   ├── rl_core/                # Algorithm-agnostic RL core (SAC implementation, base agent interface, factory)
│   ├── envs/                   # Self-contained 2D Gymnasium leader-follower environment
│   ├── scripts/                # train.py, export_policy.py, evaluate_policy.py, evaluate_scenarios.py
│   ├── inference/               # Minimal runtime to run an exported policy
│   ├── integration/             # ROS2 follower node + pose-supervisor ROS2 package
│   ├── requirements-train.txt   # Dependencies for training (torch, gymnasium, tensorboard, onnx)
│   └── requirements-deploy.txt  # Dependencies for inference-only deployment (numpy, onnxruntime)
└── ros2_package/                # ROS2/Webots deployment assets
    ├── ros2_launch/              # Patched robot launch file
    ├── ros2_ws/                  # ROS2 workspace (colcon)
    ├── shared/                   # Exported policy + inference/integration code shared with the VM
    ├── webots/                   # Webots world file
    ├── nav2/                     # Example Nav2 map for TurtleBot3 Burger
    └── README_WORKING_SETUP.md   # Known-good ROS2 Humble + Webots configuration notes
```

## Requirements

- Python 3.10+ (tested with 3.14)
- For **training**: [PyTorch](https://pytorch.org/) (`torch>=2.1`), plus `gymnasium`, `numpy`, `tensorboard`, `onnx`
- For **deployment/inference only**: `numpy`, `onnxruntime`
- For **robot deployment**: ROS2 Humble and Webots with `webots_ros2` (see [ros2_package/README_WORKING_SETUP.md](ros2_package/README_WORKING_SETUP.md))

## Installation

Clone the repository and install the dependencies for the part of the project you need.

### Training environment

```bash
cd leader_follower_rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

### Deployment / inference-only environment (e.g. on the robot or the ROS2 machine)

```bash
cd leader_follower_rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-deploy.txt
```

## Usage

### 1. Train an agent

```bash
cd leader_follower_rl
python scripts/train.py --algo sac --total-steps 200000
```

Useful flags: `--seed`, `--batch-size`, `--buffer-capacity`, `--eval-every`, `--eval-episodes`, `--checkpoint-dir` (default `checkpoints`), `--log-dir` (default `runs`).

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs
```

### 2. Evaluate a checkpoint

```bash
python scripts/evaluate_policy.py --checkpoint checkpoints/best.pt --episodes 50
python scripts/evaluate_scenarios.py --checkpoint checkpoints/best.pt --episodes 10
```

### 3. Export the trained policy

```bash
python scripts/export_policy.py \
    --checkpoint checkpoints/best.pt \
    --obs-dim 9 \
    --act-dim 2 \
    --act-limit 0.6 \
    --out policy.pt \
    --onnx
```

This produces `policy.pt` (TorchScript) and, with `--onnx`, `policy.onnx` — the artifact consumed by the deployment side.

### 4. Deploy to ROS2/Webots

Copy `policy.onnx` to the machine running ROS2, then start the follower node:

```bash
python integration/ros2_follower_node.py --policy-path policy.onnx
```

See [ros2_package/README_WORKING_SETUP.md](ros2_package/README_WORKING_SETUP.md) for the exact known-good launch sequence (Webots + `robot_launch_patched.py` + Nav2 + the SAC follower node) and expected ROS2 topics/controllers.

For the full rationale behind the architecture (why `rl_core/` stays framework-agnostic, how the pose-supervisor ground truth works, how to add a second robot to a `.wbt` world, how to plug in a new RL algorithm, etc.), see [leader_follower_rl/README.md](leader_follower_rl/README.md).

## Adding a new algorithm

`rl_core/` is designed so that new algorithms (e.g. TD3, DDPG) can be added without touching the training script: implement the `BaseAgent` interface (`select_action`, `update`, `save`, `load`, `export_actor`) and register it in `rl_core/agent_factory.py`. `scripts/train.py` stays unchanged.

## License

MIT — see [LICENSE](LICENSE).
