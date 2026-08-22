#!/usr/bin/env python

# Basato su:
# /opt/ros/humble/share/webots_ros2_turtlebot/launch/robot_launch.py
#
# Esteso con:
#   - un secondo TurtleBot3Burger (follower)
#   - namespace ROS2 "follower"
#   - pose_supervisor_driver
#   - nodo SAC separato, lanciato manualmente
#
# Il robot_state_publisher usa il vero URDF del pacchetto
# turtlebot3_description.
#
# Il WebotsController continua invece a usare
# turtlebot_webots.urdf, che contiene la configurazione specifica
# di Webots (Lidar, IMU, ros2_control, ecc.).

import os

from launch import LaunchDescription
import launch

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch_ros.actions import Node

from ament_index_python.packages import (
    get_package_share_directory,
    get_packages_with_prefixes,
)

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import (
    WaitForControllerConnection,
)


FOLLOWER_ROBOT_NAME = 'TurtleBot3Follower'


def generate_launch_description():

    # ============================================================
    # PACKAGE DIRECTORIES
    # ============================================================

    package_dir = get_package_share_directory(
        'webots_ros2_turtlebot'
    )

    pose_supervisor_dir = get_package_share_directory(
        'pose_supervisor_package'
    )

    rsp_robot_description_path = (
        '/home/ubuntu/ros2_ws/src/my_mercy_pkg/resource/'
        'robot_description/turtlebot3_burger_rsp.urdf'
    )
    
    with open(rsp_robot_description_path, 'r') as f:
        rsp_robot_description = f.read()
    
    leader_robot_description = rsp_robot_description
    follower_robot_description = rsp_robot_description

    # ============================================================
    # WEBOTS ROBOT DESCRIPTION
    # ============================================================
    #
    # Questo URDF NON viene usato dal robot_state_publisher.
    #
    # Serve al WebotsController perché contiene:
    #   - Webots devices
    #   - Lidar
    #   - IMU
    #   - ros2_control
    #
    # ============================================================

    webots_leader_description_path = (
        '/home/ubuntu/ros2_ws/src/my_mercy_pkg/resource/webots/'
        'turtlebot_webots_leader.urdf'
    )
    
    webots_follower_description_path = (
        '/home/ubuntu/ros2_ws/src/my_mercy_pkg/resource/webots/'
        'turtlebot_webots_follower.urdf'
    )

    # ============================================================
    # TRUE TURTLEBOT3 URDF
    # ============================================================
    #
    # Questo è il modello utilizzato da robot_state_publisher.
    #
    # Leader:
    #   base_link
    #   base_footprint
    #   wheel_left_link
    #   ...
    #
    # Follower:
    #   follower/base_link
    #   follower/base_footprint
    #   follower/wheel_left_link
    #   ...
    #
    # Il prefisso del follower viene applicato direttamente da xacro.
    #
    # ============================================================

    # ============================================================
    # LAUNCH ARGUMENTS
    # ============================================================

    world = LaunchConfiguration('world')

    mode = LaunchConfiguration('mode')

    use_nav = LaunchConfiguration(
        'nav',
        default=True
    )

    use_slam = LaunchConfiguration(
        'slam',
        default=False
    )

    use_sim_time = LaunchConfiguration(
        'use_sim_time',
        default=True
    )

    # ============================================================
    # WEBOTS
    # ============================================================

    webots = WebotsLauncher(
        world=PathJoinSubstitution([
            package_dir,
            'worlds',
            world
        ]),
        mode=mode,
        ros2_supervisor=True
    )

    # ============================================================
    # ROBOT STATE PUBLISHER - LEADER
    # ============================================================

    leader_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': leader_robot_description,
                'frame_prefix': '',
                'use_sim_time': use_sim_time,
            }
        ],
    )

    # ============================================================
    # ROBOT STATE PUBLISHER - FOLLOWER
    # ============================================================
    #
    # Il namespace ROS2 è /follower.
    #
    # Il frame_prefix NON viene impostato qui perché il prefisso
    # "follower/" è già presente nei frame generati da xacro.
    #
    # Quindi:
    #
    #   follower/base_link
    #   follower/base_footprint
    #   follower/wheel_left_link
    #
    # ============================================================

    follower_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='follower',
        output='screen',
        parameters=[
            {
                'robot_description': follower_robot_description,
                'frame_prefix': 'follower/',
                'use_sim_time': use_sim_time,
            }
        ],
    )

    # ============================================================
    # LEADER STATIC TF
    # ============================================================
    #
    # Questo collega:
    #
    #   base_link -> LDS-01
    #
    # perché LDS-01 è un frame specifico del modello Webots.
    #
    # ============================================================


    # ============================================================
    # ROS2 CONTROL
    # ============================================================

    controller_manager_timeout = [
        '--controller-manager-timeout',
        '50'
    ]

    controller_manager_prefix = (
        'python.exe'
        if os.name == 'nt'
        else ''
    )

    # ------------------------------------------------------------
    # LEADER CONTROLLERS
    # ------------------------------------------------------------

    diffdrive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=[
            'diffdrive_controller'
        ] + controller_manager_timeout,
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=[
            'joint_state_broadcaster'
        ] + controller_manager_timeout,
    )

    ros_control_spawners = [
        diffdrive_controller_spawner,
        joint_state_broadcaster_spawner,
    ]

    # ============================================================
    # ROS2 CONTROL CONFIGURATION
    # ============================================================

    ros2_control_params = (
        '/home/ubuntu/ros2_ws/src/my_mercy_pkg/'
        'resource/ros2control_mercy.yml'
    )

    use_twist_stamped = (
        'ROS_DISTRO' in os.environ
        and os.environ['ROS_DISTRO'] in ['rolling', 'jazzy']
    )

    # ============================================================
    # LEADER TOPIC REMAPPINGS
    # ============================================================

    if use_twist_stamped:

        mappings = [
            (
                '/diffdrive_controller/cmd_vel',
                '/cmd_vel'
            ),
            (
                '/diffdrive_controller/odom',
                '/odom'
            ),
        ]

    else:

        mappings = [
            (
                '/diffdrive_controller/cmd_vel_unstamped',
                '/cmd_vel'
            ),
            (
                '/diffdrive_controller/odom',
                '/odom'
            ),
        ]

    # ============================================================
    # LEADER WEBOTS CONTROLLER
    # ============================================================

    turtlebot_driver = WebotsController(
        robot_name='TurtleBot3Burger',

        parameters=[
            {
                'robot_description': webots_leader_description_path,
                'use_sim_time': use_sim_time,

                # IMPORTANTISSIMO:
                # il RSP viene gestito esplicitamente sopra.
                'set_robot_state_publisher': False,
            },

            ros2_control_params,
        ],

        remappings=mappings,

        respawn=True,
    )

    # ============================================================
    # FOLLOWER TOPIC REMAPPINGS
    # ============================================================

    if use_twist_stamped:

        follower_mappings = [
            (
                '/follower/diffdrive_controller/cmd_vel',
                '/follower/cmd_vel'
            ),
            (
                '/follower/diffdrive_controller/odom',
                '/follower/odom'
            ),
        ]

    else:

        follower_mappings = [
            (
                '/follower/diffdrive_controller/cmd_vel_unstamped',
                '/follower/cmd_vel'
            ),
            (
                '/follower/diffdrive_controller/odom',
                '/follower/odom'
            ),
        ]

    # ============================================================
    # FOLLOWER WEBOTS CONTROLLER
    # ============================================================

    follower_driver = WebotsController(
        robot_name=FOLLOWER_ROBOT_NAME,

        namespace='follower',

        parameters=[
            {
                'robot_description': webots_follower_description_path,
                'use_sim_time': use_sim_time,

                # Il RSP del follower viene gestito esplicitamente
                # sopra.
                'set_robot_state_publisher': False,
            },

            ros2_control_params,
        ],

        remappings=follower_mappings,

        respawn=True,
    )

    # ============================================================
    # FOLLOWER CONTROLLERS
    # ============================================================

    follower_diffdrive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=[
            'diffdrive_controller',
            '--controller-manager',
            '/follower/controller_manager',
        ] + controller_manager_timeout,
    )

    follower_joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/follower/controller_manager',
        ] + controller_manager_timeout,
    )

    # ============================================================
    # POSE SUPERVISOR
    # ============================================================

    pose_supervisor_urdf = os.path.join(
        pose_supervisor_dir,
        'resource',
        'pose_supervisor.urdf'
    )

    pose_supervisor_driver = WebotsController(
        robot_name='pose_supervisor',

        parameters=[
            {
                'robot_description': pose_supervisor_urdf
            }
        ],

        respawn=True,
    )

    # ============================================================
    # NAVIGATION / SLAM
    # ============================================================

    navigation_nodes = []
    

    os.environ['TURTLEBOT3_MODEL'] = 'burger'

    nav2_map = os.path.join(
        package_dir,
        'resource',
        'turtlebot3_burger_example_map.yaml'
    )

    nav2_params = os.path.join(
        package_dir,
        'resource',
        'nav2_params.yaml'
    )

    # ------------------------------------------------------------
    # NAV2
    # ------------------------------------------------------------

    if 'turtlebot3_navigation2' in get_packages_with_prefixes():

        turtlebot_navigation = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory(
                        'turtlebot3_navigation2'
                    ),
                    'launch',
                    'navigation2.launch.py'
                )
            ),

            launch_arguments=[
                ('map', nav2_map),
                ('params_file', nav2_params),
                ('use_sim_time', use_sim_time),
                ('autostart','true'),
            ],

            condition=launch.conditions.IfCondition(
                use_nav
            )
        )

        navigation_nodes.append(
            turtlebot_navigation
        )

    # ------------------------------------------------------------
    # CARTOGRAPHER
    # ------------------------------------------------------------

    if 'turtlebot3_cartographer' in get_packages_with_prefixes():

        turtlebot_slam = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory(
                        'turtlebot3_cartographer'
                    ),
                    'launch',
                    'cartographer.launch.py'
                )
            ),

            launch_arguments=[
                ('use_sim_time', use_sim_time)
            ],

            condition=launch.conditions.IfCondition(
                use_slam
            )
        )

        navigation_nodes.append(
            turtlebot_slam
        )

    # ============================================================
    # WAIT FOR CONTROLLER CONNECTION
    # ============================================================

    waiting_nodes = WaitForControllerConnection(
        target_driver=turtlebot_driver,

        nodes_to_start=(
            navigation_nodes
            + ros_control_spawners
            + [
                follower_driver,
                follower_diffdrive_spawner,
                follower_joint_state_spawner,
                pose_supervisor_driver,
            ]
        )
    )

    # ============================================================
    # LAUNCH DESCRIPTION
    # ============================================================

    return LaunchDescription([

        DeclareLaunchArgument(
            'world',

            default_value='turtlebot3_burger_example.wbt',

            description=(
                'Choose one of the world files from '
                '`/webots_ros2_turtlebot/world` directory'
            )
        ),

        DeclareLaunchArgument(
            'mode',

            default_value='realtime',

            description='Webots startup mode'
        ),

        DeclareLaunchArgument(
            'nav',
            default_value='true',
            description='Launch Nav2'
        ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),

        # --------------------------------------------------------
        # WEBOTS
        # --------------------------------------------------------

        webots,

        webots._supervisor,

        # --------------------------------------------------------
        # TF / ROBOT DESCRIPTION
        # --------------------------------------------------------

        leader_robot_state_publisher,

        follower_robot_state_publisher,


        # --------------------------------------------------------
        # LEADER
        # --------------------------------------------------------

        turtlebot_driver,

        # --------------------------------------------------------
        # CONTROLLERS + FOLLOWER + SUPERVISOR
        # --------------------------------------------------------

        waiting_nodes,

        # --------------------------------------------------------
        # SHUTDOWN WHEN WEBOTS CLOSES
        # --------------------------------------------------------

        launch.actions.RegisterEventHandler(

            event_handler=launch.event_handlers.OnProcessExit(

                target_action=webots,

                on_exit=[
                    launch.actions.EmitEvent(
                        event=launch.events.Shutdown()
                    )
                ],
            )
        ),
    ])
