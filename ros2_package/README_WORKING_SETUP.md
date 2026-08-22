# MERCY - ROS2/Webots Working Configuration

## Environment

ROS2 Humble

## IMPORTANT

Do NOT source:

source ~/ros2_ws/install/setup.bash

The working environment uses only:

source /opt/ros/humble/setup.bash

## Start simulation

source /opt/ros/humble/setup.bash

ros2 launch webots_ros2_turtlebot robot_launch_patched.py

## Start SAC follower

source /opt/ros/humble/setup.bash

python3 ~/shared/integration/ros2_follower_node.py \
    --policy-path ~/shared/policy.onnx

## Expected ROS2 state

Nav2:

/planner_server     active
/controller_server  active
/bt_navigator       active
/behavior_server    active

Follower:

/follower/odom
/follower/cmd_vel

Controllers:

diffdrive_controller      active
joint_state_broadcaster   active

Follower controller manager:

ros2 control list_controllers \
    --controller-manager /follower/controller_manager
