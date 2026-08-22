#!/usr/bin/env python
"""
Launch file per la simulazione MERCY leader-follower.

Avvia:
  - Webots con la scena a due robot
  - Driver leader  (TurtleBot3Burger)   → topic /odom, /cmd_vel
  - Driver follower (TurtleBot3Follower) → topic /follower/odom, /follower/cmd_vel
  - Nav2 per il leader (opzionale, --nav)
  - Nodo SAC follower che legge /odom (leader) e /follower/odom (follower)
    e pubblica su /follower/cmd_vel

Uso:
    ros2 launch webots_ros2_turtlebot robot_launch_two_robots.py nav:=true
"""

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


def generate_launch_description():
    package_dir = get_package_share_directory('webots_ros2_turtlebot')
    mode        = LaunchConfiguration('mode')
    use_nav     = LaunchConfiguration('nav',          default=False)
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)

    # Percorso del modello con due robot (copiato nella cartella worlds del pacchetto)
    world_path = os.path.join(
        package_dir, 'worlds', 'turtlebot3_burger_example_two_robots.wbt'
    )

    # ── Webots ──────────────────────────────────────────────────────────────
    webots = WebotsLauncher(
        world=world_path,
        mode=mode,
        ros2_supervisor=True
    )

    # ── Robot state publisher (comune) ───────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': '<robot name=""><link name=""/></robot>'}],
    )

    footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
    )

    # ── Driver LEADER (TurtleBot3Burger) ─────────────────────────────────────
    robot_description_path = os.path.join(package_dir, 'resource', 'turtlebot_webots.urdf')
    ros2_control_params    = os.path.join(package_dir, 'resource', 'ros2control.yml')

    use_twist_stamped = 'ROS_DISTRO' in os.environ and \
                        os.environ['ROS_DISTRO'] in ['rolling', 'jazzy']

    if use_twist_stamped:
        leader_mappings = [
            ('/diffdrive_controller/cmd_vel', '/cmd_vel'),
            ('/diffdrive_controller/odom',    '/odom'),
        ]
    else:
        leader_mappings = [
            ('/diffdrive_controller/cmd_vel_unstamped', '/cmd_vel'),
            ('/diffdrive_controller/odom',              '/odom'),
        ]

    leader_driver = WebotsController(
        robot_name='TurtleBot3Burger',
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time':      use_sim_time,
             'set_robot_state_publisher': True},
            ros2_control_params,
        ],
        remappings=leader_mappings,
        respawn=True
    )

    # ROS control spawners per il leader
    controller_manager_prefix  = 'python.exe' if os.name == 'nt' else ''
    controller_manager_timeout = ['--controller-manager-timeout', '50']

    # ── ROS control spawners per il LEADER ───────────────────────────────────
    leader_diffdrive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=['diffdrive_controller', '-c', '/TurtleBot3Burger/controller_manager'] + controller_manager_timeout,
    )
    leader_joint_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=['joint_state_broadcaster', '-c', '/TurtleBot3Burger/controller_manager'] + controller_manager_timeout,
    )
    leader_ros_control_spawners = [leader_diffdrive_spawner, leader_joint_broadcaster]

    # ── ROS control spawners per il FOLLOWER ─────────────────────────────────
    follower_diffdrive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=['diffdrive_controller', '-c', '/TurtleBot3Follower/controller_manager'] + controller_manager_timeout,
    )
    follower_joint_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=['joint_state_broadcaster', '-c', '/TurtleBot3Follower/controller_manager'] + controller_manager_timeout,
    )
    follower_ros_control_spawners = [follower_diffdrive_spawner, follower_joint_broadcaster]
    # ── Driver FOLLOWER (TurtleBot3Follower) ─────────────────────────────────
    # Tutti i topic del follower vengono rimappati sotto /follower/
    # ── Driver FOLLOWER (TurtleBot3Follower) ─────────────────────────────────
    if use_twist_stamped:
        follower_mappings = [
            ('/diffdrive_controller/cmd_vel', '/follower/cmd_vel'),
            ('/diffdrive_controller/odom',    '/follower/odom'),
            ('/scan',                         '/follower/scan'),      # Isola il Lidar
        ]
    else:
        follower_mappings = [
            ('/diffdrive_controller/cmd_vel_unstamped', '/follower/cmd_vel'),
            ('/diffdrive_controller/odom',              '/follower/odom'),
            ('/scan',                                   '/follower/scan'),
        ]
    follower_driver = WebotsController(
        robot_name='TurtleBot3Follower',
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time':      use_sim_time,
             'set_robot_state_publisher': False},   # il leader gestisce già il RSP
            ros2_control_params,
        ],
        remappings=follower_mappings,
        respawn=True
    )

    # ── Nodo SAC Follower ────────────────────────────────────────────────────
    # Legge /odom (leader) e /follower/odom (follower),
    # pubblica Twist su /follower/cmd_vel
    # Modifica 'model_path' con il percorso del tuo actor addestrato (.pt)
    sac_follower_node = Node(
        package='my_mercy_pkg',
        executable='sac_follower_node',
        name='sac_follower',
        output='screen',
        parameters=[{
            'use_sim_time':       use_sim_time,
            'model_path':         '/home/ubuntu/shared/sac_actor.pt',
            'leader_odom_topic':  '/odom',
            'follower_odom_topic': '/follower/odom',
            'cmd_vel_topic':      '/follower/cmd_vel',
            'catch_threshold':    0.3,    # metri — distanza per considerare "catturato"
            'max_linear_vel':     0.22,   # m/s (limite TurtleBot3 Burger)
            'max_angular_vel':    2.84,   # rad/s
        }]
    )

    # ── Nav2 per il leader ───────────────────────────────────────────────────
    navigation_nodes = []
    os.environ['TURTLEBOT3_MODEL'] = 'burger'

    nav2_map    = os.path.join(package_dir, 'resource', 'turtlebot3_burger_example_map.yaml')
    nav2_params = os.path.join(package_dir, 'resource', 'nav2_params.yaml')

    if 'turtlebot3_navigation2' in get_packages_with_prefixes():
        turtlebot_navigation = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('turtlebot3_navigation2'),
                'launch', 'navigation2.launch.py'
            )),
            launch_arguments=[
                ('map',          nav2_map),
                ('params_file',  nav2_params),
                ('use_sim_time', use_sim_time),
            ],
            condition=launch.conditions.IfCondition(use_nav)
        )
        navigation_nodes.append(turtlebot_navigation)

    # ── Attendi che il leader sia pronto prima di avviare Nav2 ───────────────
    waiting_nodes = WaitForControllerConnection(
        target_driver=leader_driver,
        nodes_to_start=navigation_nodes + leader_ros_control_spawners
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='realtime',
            description='Webots startup mode'
        ),
        DeclareLaunchArgument(
            'nav',
            default_value='False',
            description='Start Nav2 for the leader robot'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
        ),

        webots,
        webots._supervisor,

        robot_state_publisher,
        footprint_publisher,

        leader_driver,
        follower_driver,
        waiting_nodes,
        sac_follower_node,

        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ])


# ── Attendi che il leader sia pronto prima di avviare Nav2 e i controller
    waiting_nodes = WaitForControllerConnection(
        target_driver=leader_driver,
        nodes_to_start=navigation_nodes + leader_ros_control_spawners + follower_ros_control_spawners
    )
