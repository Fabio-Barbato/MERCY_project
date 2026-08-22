"""
Launch file esteso rispetto a quello ufficiale di webots_ros2_turtlebot:
leader (TurtleBot3Burger, invariato, con Nav2 gia' configurato altrove nel
tuo setup) + follower (secondo TurtleBot3Burger, namespace "follower",
NESSUN Nav2/SLAM: viene guidato dalla policy SAC via /follower/cmd_vel).

Basato sullo schema ufficiale multi-robot di webots_ros2
(vedi webots_ros2_universal_robot/launch/multirobot_launch.py su github.com/
cyberbotics/webots_ros2): un'istanza di WebotsController per robot, ciascuna
col proprio namespace.

NON TESTATO qui (richiede Webots + ROS2, che non ho a disposizione in
questo ambiente): e' una ricostruzione fedele ai pattern ufficiali, ma va
verificata/debuggata sul tuo setup reale. I punti piu' probabili da dover
aggiustare sono segnalati nei commenti.

Adatta:
  - il path del tuo world file (variabile WORLD_FILE)
  - se il tuo launch attuale del leader ha gia' altri nodi (Nav2, RViz,
    ecc.), integra questo file con quelli invece di sostituirli
"""
import os
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
import launch
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import WaitForControllerConnection

WORLD_FILE = 'YOUR_WORLD.wbt'  # <-- il tuo world file, con leader + follower + Supervisor aggiunti


def generate_launch_description():
    package_dir = get_package_share_directory('webots_ros2_turtlebot')
    mode = LaunchConfiguration('mode')
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', WORLD_FILE),
        mode=mode,
        ros2_supervisor=True
    )

    robot_description_path = os.path.join(package_dir, 'resource', 'turtlebot_webots.urdf')
    ros2_control_params = os.path.join(package_dir, 'resource', 'ros2control.yml')

    use_twist_stamped = 'ROS_DISTRO' in os.environ and (os.environ['ROS_DISTRO'] in ['rolling', 'jazzy', 'kilted'])
    cmd_vel_src = '/diffdrive_controller/cmd_vel' if use_twist_stamped else '/diffdrive_controller/cmd_vel_unstamped'

    # ---------------- LEADER: stesso schema del launch ufficiale ----------------
    leader_driver = WebotsController(
        robot_name='TurtleBot3Burger',          # deve combaciare col campo "name" nel .wbt
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time': use_sim_time,
             'set_robot_state_publisher': True},
            ros2_control_params
        ],
        remappings=[(cmd_vel_src, '/cmd_vel'), ('/diffdrive_controller/odom', '/odom')],
        respawn=True
    )
    leader_diffdrive_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=['diffdrive_controller', '--controller-manager-timeout', '50'],
    )
    leader_joint_state_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '50'],
    )

    # ---------------- FOLLOWER: stesso modello, namespace separato ----------------
    # i target delle remap sono relativi (senza "/" iniziale): col namespace
    # "follower" impostato sotto, diventano automaticamente /follower/cmd_vel
    # e /follower/odom.
    follower_driver = WebotsController(
        robot_name='TurtleBot3Burger_follower',  # deve combaciare col campo "name" nel .wbt
        namespace='follower',
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time': use_sim_time,
             'set_robot_state_publisher': True},
            ros2_control_params
        ],
        remappings=[(cmd_vel_src, 'cmd_vel'), ('/diffdrive_controller/odom', 'odom')],
        respawn=True
    )
    # NOTA: la propagazione del namespace al controller_manager di ros2_control
    # e' il punto piu' probabile da dover aggiustare a mano. Se lo spawner non
    # trova il controller_manager del follower, prova esplicitando
    # --controller-manager /follower/controller_manager come sotto, oppure
    # verifica con `ros2 node list` come si chiama davvero una volta avviato.
    follower_diffdrive_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        namespace='follower',
        arguments=['diffdrive_controller', '--controller-manager-timeout', '50',
                   '--controller-manager', '/follower/controller_manager'],
    )
    follower_joint_state_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        namespace='follower',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '50',
                   '--controller-manager', '/follower/controller_manager'],
    )

    # Supervisor: pubblica le pose ground-truth (vedi
    # integration/pose_supervisor_package/). E' un plugin webots_ros2_driver,
    # va lanciato come una WebotsController in piu' esattamente come i robot -
    # NON parte da solo: a differenza di un controller Webots interno, gira
    # nella VM e si connette a Webots via <extern>, quindi va nel launch file.
    pose_supervisor_urdf = os.path.join(
        get_package_share_directory('pose_supervisor_package'), 'resource', 'pose_supervisor.urdf'
    )
    pose_supervisor_driver = WebotsController(
        robot_name='pose_supervisor',   # deve combaciare col campo "name" nel .wbt
        parameters=[{'robot_description': pose_supervisor_urdf}],
        respawn=True,
    )

    # Nodo della policy SAC: legge le pose ground-truth + le odometrie,
    # pubblica /follower/cmd_vel
    policy_node = Node(
        package='<il_tuo_package_ros2>',   # <-- adatta al pacchetto dove metti ros2_follower_node.py
        executable='ros2_follower_node',
        arguments=['--policy-path', '/percorso/assoluto/a/policy.onnx'],
        output='screen',
    )

    waiting_nodes = WaitForControllerConnection(
        target_driver=leader_driver,
        nodes_to_start=[
            leader_diffdrive_spawner, leader_joint_state_spawner,
            follower_driver, follower_diffdrive_spawner, follower_joint_state_spawner,
            pose_supervisor_driver, policy_node,
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='realtime'),
        webots,
        webots._supervisor,
        leader_driver,
        waiting_nodes,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ])
