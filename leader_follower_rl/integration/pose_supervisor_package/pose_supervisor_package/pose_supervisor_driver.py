"""
Plugin per webots_ros2_driver: pubblica la pose "ground truth" di leader e
follower leggendola direttamente dal simulatore Webots.

PERCHE' UN PLUGIN E NON UNO SCRIPT WEBOTS NORMALE: nel setup di questo
progetto (Webots nativo su macOS via local_simulation_server.py, ROS2 nella
VM Linux via UTM), Webots non ha accesso a rclpy: gira sul Mac, non nella
VM. Un controller Webots "interno" che facesse `import rclpy` fallirebbe
all'avvio. Il pattern ufficiale e' questo: il codice Python gira nella VM
(dove rclpy e' disponibile) e si connette a Webots via lo stesso
meccanismo <extern> (connessione TCP) gia' usato dai driver dei due
TurtleBot3 — qui lo riusiamo solo per LEGGERE posizioni invece che
scrivere velocita' alle ruote.

Il metodo init() riceve `webots_node`, il cui attributo `.robot` e' gia'
un'istanza Supervisor (perche' il nodo Robot a cui questo plugin si
collega ha il campo `supervisor TRUE` nel .wbt) — quindi getFromDef()
funziona direttamente su di esso, senza dover istanziare nulla a parte.

Requisiti nel file .wbt sul nodo Robot a cui questo plugin si collega:
  - supervisor TRUE        (necessario per leggere la posizione di ALTRI nodi)
  - controller "<extern>"
  - name "pose_supervisor" (deve combaciare col robot_name nel launch file)

ATTENZIONE: LEADER_DEF/FOLLOWER_DEF devono combaciare con i DEF assegnati
ai due nodi Robot{} dei TurtleBot3 nel tuo .wbt. Il calcolo dello yaw
assume coordinateSystem "ENU" (Z verticale) nel WorldInfo del tuo .wbt:
se il tuo mondo usa invece "NUE" (default Webots, Y verticale), la riga
e' commentata sotto con l'alternativa.
"""
from __future__ import annotations
import math
import rclpy
from geometry_msgs.msg import PoseStamped

LEADER_DEF = "TURTLEBOT3_BURGER"
FOLLOWER_DEF = "TURTLEBOT3_BURGER_FOLLOWER"


class PoseSupervisorDriver:
    def init(self, webots_node, properties):
        # webots_node.robot e' gia' un'istanza Supervisor: il nodo Robot
        # a cui e' collegato questo plugin ha supervisor TRUE nel .wbt.
        self.__robot = webots_node.robot

        self.__leader_node = self.__robot.getFromDef(LEADER_DEF)
        self.__follower_node = self.__robot.getFromDef(FOLLOWER_DEF)
        if self.__leader_node is None or self.__follower_node is None:
            raise RuntimeError(
                f"DEF non trovato nel .wbt: LEADER_DEF='{LEADER_DEF}', "
                f"FOLLOWER_DEF='{FOLLOWER_DEF}'. Controlla i DEF name reali "
                f"assegnati ai due nodi TurtleBot3Burger."
            )

        rclpy.init(args=None)
        self.__node = rclpy.create_node('pose_supervisor_driver')
        self.__leader_pub = self.__node.create_publisher(PoseStamped, '/leader/ground_truth_pose', 10)
        self.__follower_pub = self.__node.create_publisher(PoseStamped, '/follower/ground_truth_pose', 10)

    def _publish_pose(self, node, publisher):
        pos = node.getPosition()        # [x, y, z] nel frame mondo di Webots
        rot = node.getOrientation()     # matrice di rotazione 3x3, row-major (9 valori)

        # coordinateSystem "ENU" (Z verticale):
        x, y = pos[0], pos[1]
        yaw = math.atan2(rot[3], rot[0])
        # Se il tuo .wbt usa invece "NUE" (Y verticale, default Webots),
        # commenta le due righe sopra e usa queste:
        # x, y = pos[0], pos[2]
        # yaw = math.atan2(rot[6], rot[8])

        msg = PoseStamped()
        msg.header.stamp = self.__node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'      # deve coincidere col frame usato da Nav2/AMCL sul leader
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        publisher.publish(msg)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        self._publish_pose(self.__leader_node, self.__leader_pub)
        self._publish_pose(self.__follower_node, self.__follower_pub)
