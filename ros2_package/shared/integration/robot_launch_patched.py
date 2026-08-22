#!/usr/bin/env python

# Basato su /opt/ros/humble/share/webots_ros2_turtlebot/launch/robot_launch.py
# (Copyright 1996-2023 Cyberbotics Ltd., Apache License 2.0), esteso con:
#   - un secondo TurtleBot3Burger (il follower, name "TurtleBot3Follower"),
#     namespace "follower"
#   - il plugin pose_supervisor_driver (pose ground-truth di leader/follower)
#   - il nodo della policy SAC (ros2_follower_node.py)
#
# World file: turtlebot3_burger_example.wbt (il DEFAULT del pacchetto, lo
# stesso gia' usato per la mappa e la navigazione Nav2 esistenti) esteso con
# il follower + il Supervisor — vedi integration/wbt_diff_instructions.txt.
# Nessun world:= diverso da passare: resta il world di default.
#
# Uso:
#   ros2 launch <questo_file>
# (oppure copia questo file dentro il pacchetto webots_ros2_turtlebot,
#  sovrascrivendo l'originale, come gia' fai per altri file di config)

import os
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node
import launch
from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import WaitForControllerConnection

FOLLOWER_ROBOT_NAME = 'TurtleBot3Follower'  # confermato dal world file turtlebot3_burger_example_two_robots.wbt


def generate_launch_description():
    package_dir = get_package_share_directory('webots_ros2_turtlebot')
    pose_supervisor_dir = get_package_share_directory('pose_supervisor_package')

    world = LaunchConfiguration('world')
    mode = LaunchConfiguration('mode')
    use_nav = LaunchConfiguration('nav', default=False)
    use_slam = LaunchConfiguration('slam', default=False)
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)

    webots = WebotsLauncher(
        world=PathJoinSubstitution([package_dir, 'worlds', world]),
        mode=mode,
        ros2_supervisor=True
    )

    # --- invariato rispetto all'originale (leader) ---
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': '<robot name=""><link name=""/></robot>'
        }],
    )

    footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
    )

    controller_manager_timeout = ['--controller-manager-timeout', '50']
    controller_manager_prefix = 'python.exe' if os.name == 'nt' else ''

    diffdrive_controller_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        prefix=controller_manager_prefix,
        arguments=['diffdrive_controller'] + controller_manager_timeout,
    )
    joint_state_broadcaster_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        prefix=controller_manager_prefix,
        arguments=['joint_state_broadcaster'] + controller_manager_timeout,
    )
    ros_control_spawners = [diffdrive_controller_spawner, joint_state_broadcaster_spawner]

    robot_description_path = os.path.join(package_dir, 'resource', 'turtlebot_webots.urdf')
    ros2_control_params = os.path.join(package_dir, 'resource', 'ros2control.yml')
    use_twist_stamped = 'ROS_DISTRO' in os.environ and (os.environ['ROS_DISTRO'] in ['rolling', 'jazzy'])
    if use_twist_stamped:
        mappings = [('/diffdrive_controller/cmd_vel', '/cmd_vel'), ('/diffdrive_controller/odom', '/odom')]
    else:
        mappings = [('/diffdrive_controller/cmd_vel_unstamped', '/cmd_vel'), ('/diffdrive_controller/odom', '/odom')]

    turtlebot_driver = WebotsController(
        robot_name='TurtleBot3Burger',
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time': use_sim_time,
             'set_robot_state_publisher': True},
            ros2_control_params
        ],
        remappings=mappings,
        respawn=True
    )

    # --- NUOVO: follower, stesso modello, namespace separato ---
    # remap con target relativi (senza "/"): col namespace 'follower' sotto,
    # diventano automaticamente /follower/cmd_vel e /follower/odom
    if use_twist_stamped:
        follower_mappings = [('/diffdrive_controller/cmd_vel', 'cmd_vel'), ('/diffdrive_controller/odom', 'odom')]
    else:
        follower_mappings = [('/diffdrive_controller/cmd_vel_unstamped', 'cmd_vel'), ('/diffdrive_controller/odom', 'odom')]

    follower_driver = WebotsController(
        robot_name=FOLLOWER_ROBOT_NAME,   # TODO 1
        namespace='follower',
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time': use_sim_time,
             'set_robot_state_publisher': True},
            ros2_control_params
        ],
        remappings=follower_mappings,
        respawn=True
    )
    # nota: se lo spawner non trova il controller_manager del follower,
    # aggiungi esplicitamente '--controller-manager', '/follower/controller_manager'
    # agli arguments qui sotto (verificare empiricamente, vedi README)
    follower_diffdrive_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        namespace='follower', prefix=controller_manager_prefix,
        arguments=['diffdrive_controller'] + controller_manager_timeout,
    )
    follower_joint_state_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        namespace='follower', prefix=controller_manager_prefix,
        arguments=['joint_state_broadcaster'] + controller_manager_timeout,
    )

    # --- NUOVO: pose_supervisor (pose ground-truth di leader e follower) ---
    pose_supervisor_urdf = os.path.join(pose_supervisor_dir, 'resource', 'pose_supervisor.urdf')
    pose_supervisor_driver = WebotsController(
        robot_name='pose_supervisor',   # deve combaciare col "name" del Robot supervisor nel .wbt
        parameters=[{'robot_description': pose_supervisor_urdf}],
        respawn=True,
    )

    # --- Nodo della policy SAC: NON incluso in questo launch file.
    # Lancialo a mano in un terzo terminale, dopo che questo launch e' partito:
    #   python3 ros2_follower_node.py --policy-path /home/ubuntu/shared/policy.onnx
    # (tenerlo separato rende piu' facile leggere i log di ciascun pezzo,
    # ed evita di dover impacchettare ros2_follower_node.py come eseguibile
    # ROS2 installato solo per poterlo referenziare qui dentro)

    # --- invariato: navigazione/SLAM opzionali (di default entrambi False,
    # dato che nel tuo flusso lanci navigation2.launch.py separatamente) ---
    navigation_nodes = []
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    nav2_map = os.path.join(package_dir, 'resource', 'turtlebot3_burger_example_map.yaml')
    nav2_params = os.path.join(package_dir, 'resource', 'nav2_params.yaml')
    if 'turtlebot3_navigation2' in get_packages_with_prefixes():
        turtlebot_navigation = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('turtlebot3_navigation2'), 'launch', 'navigation2.launch.py')),
            launch_arguments=[
                ('map', nav2_map),
                ('params_file', nav2_params),
                ('use_sim_time', use_sim_time),
            ],
            condition=launch.conditions.IfCondition(use_nav))
        navigation_nodes.append(turtlebot_navigation)

    if 'turtlebot3_cartographer' in get_packages_with_prefixes():
        turtlebot_slam = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('turtlebot3_cartographer'), 'launch', 'cartographer.launch.py')),
            launch_arguments=[('use_sim_time', use_sim_time)],
            condition=launch.conditions.IfCondition(use_slam))
        navigation_nodes.append(turtlebot_slam)

    waiting_nodes = WaitForControllerConnection(
        target_driver=turtlebot_driver,
        nodes_to_start=navigation_nodes + ros_control_spawners + [
            follower_driver, follower_diffdrive_spawner, follower_joint_state_spawner,
            pose_supervisor_driver,
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='turtlebot3_burger_example.wbt',  # invariato: e' il world gia' esteso col follower
            description='Choose one of the world files from `/webots_ros2_turtlebot/world` directory'
        ),
        DeclareLaunchArgument('mode', default_value='realtime', description='Webots startup mode'),
        webots,
        webots._supervisor,
        robot_state_publisher,
        footprint_publisher,
        turtlebot_driver,
        waiting_nodes,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ])
