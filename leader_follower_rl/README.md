# Leader-Follower RL — SAC isolato e deploy modulare su Webots/ROS2

L’architettura del progetto è stata pensata per tenere separati l’algoritmo di Reinforcement Learning e l’ambiente di esecuzione reale.

L’idea è semplice: SAC, e in futuro anche gli altri algoritmi RL, devono poter essere addestrati e valutati senza dipendere da ROS2 o Webots. Una volta ottenuta una policy, questa viene esportata e utilizzata attraverso un layer di inferenza separato, che permette di portarla su Webots senza legare direttamente la policy al framework utilizzato per eseguirla.

In questo modo è possibile lavorare sul training in maniera molto più rapida e, allo stesso tempo, mantenere il deploy indipendente dall’implementazione dell’algoritmo.


## Struttura del progetto
```
rl_core/
  base_agent.py
  networks.py
  sac_agent.py
  agent_factory.py
  replay_buffer.py
envs/
  leader_follower_env.py
scripts/
  train.py
  export_policy.py
inference/
  policy_runtime.py
integration/
  pose_supervisor_package/
  ros2_follower_node.py
  multirobot_launch_example.py
```

*rl_core/* contiene tutta la parte relativa al Reinforcement Learning. Non deve avere dipendenze da ROS2, Webots o da ambienti specifici di simulazione.

*base_agent.py* definisce l’interfaccia comune che dovranno rispettare gli algoritmi, con operazioni come select_action, update, save, load ed export_actor.

*networks.py* contiene le reti utilizzate da SAC, in particolare il GaussianActor con policy squashata tramite tanh e i due critic TwinQCritic.

*sac_agent.py* contiene l’implementazione completa di SAC, compreso l’auto-tuning di alpha.

*agent_factory.py* mantiene il registro degli algoritmi e permette di creare un agente con una chiamata del tipo:

```make_agent("sac", config=...)```

*replay_buffer.py* implementa il replay buffer utilizzando NumPy, senza introdurre una dipendenza da PyTorch.

*La directory envs/* contiene l’ambiente di training. In particolare, leader_follower_env.py è un ambiente Gymnasium 2D completamente autonomo. Il leader viene simulato utilizzando una logica pure-pursuit verso waypoint casuali, che rappresenta una versione semplificata del comportamento che in futuro sarà fornito da Nav2. Il follower utilizza invece una cinematica unicycle e riceve un’azione [v, w], analoga a un cmd_vel.

*La directory scripts/* contiene gli script necessari per il training e l’esportazione della policy. Il training può quindi essere eseguito senza avviare né Webots né la VM ROS2.

*export_policy.py* si occupa esclusivamente di esportare l’actor addestrato in TorchScript (.pt) e/o ONNX (.onnx).

*inference/policy_runtime.py* contiene il runtime minimo necessario per eseguire la policy. L’obiettivo è che questo sia l’unico componente della parte RL importato direttamente dal codice ROS2/Webots.

## Integrazione con Webots e ROS2

Per il setup multi-robot viene utilizzato il TurtleBot3 ufficiale di webots_ros2 per il leader, con Nav2 già funzionante.

Il leader viene avviato tramite:

```ros2 launch webots_ros2_turtlebot robot_launch.py```

utilizzando il world e il robot di default, robot_name impostato su TurtleBot3Burger, il WebotsController e ros2_control con diffdrive_controller.

Webots viene eseguito nativamente sul Mac attraverso local_simulation_server.py e WEBOTS_SHARED_FOLDER, mentre Nav2 viene avviato separatamente tramite turtlebot3_navigation2 navigation2.launch.py.

Il collegamento tra Webots e ROS2 viene effettuato utilizzando il nome del robot. Il valore di robot_name specificato nel launch deve quindi corrispondere al campo name del nodo Robot presente nel file .wbt.

Questo permette di aggiungere il follower senza modificare il setup del leader: basta creare una seconda istanza del robot con un nome differente.

## Supervisor e ground truth

Per ottenere la ground truth delle pose di leader e follower viene utilizzato un pacchetto ROS2 separato:

```integration/pose_supervisor_package/```

Il pacchetto contiene un plugin webots_ros2_driver con le funzioni init e step. Il plugin legge direttamente da Webots la posizione dei due robot e pubblica le informazioni su:
```
/leader/ground_truth_pose
/follower/ground_truth_pose
```
La scelta di utilizzare un pacchetto separato, anziché un semplice script Python, dipende dal modo in cui è configurato attualmente l’ambiente. Webots gira sul Mac mentre ROS2 gira all’interno della VM Linux. In questa configurazione il codice ROS2 non può utilizzare direttamente rclpy dall’interno di un controller Webots tradizionale.

Il plugin viene quindi eseguito nella VM e si collega a Webots attraverso <extern>, seguendo lo stesso meccanismo utilizzato dai driver dei due TurtleBot3.

Il Supervisor ha quindi il solo compito di fornire la ground truth necessaria per lo sviluppo e la valutazione. Non deve essere coinvolto nel controllo del follower.

## Configurazione del secondo robot

Nel file .wbt è necessario aggiungere un secondo TurtleBot3Burger con un nome differente, ad esempio:

*TurtleBot3Burger_follower*

Il robot dovrà utilizzare:

*controller "<extern>"*

e avere un proprio DEF, ad esempio:

```TURTLEBOT3_BURGER_FOLLOWER```

È utile avere un DEF anche per il leader, ad esempio:

```TURTLEBOT3_BURGER```

Per il Supervisor viene invece aggiunto un terzo nodo:
```
Robot {
  supervisor TRUE
  controller "<extern>"
  name "pose_supervisor"
}
```
Il pacchetto del Supervisor deve poi essere compilato nel workspace ROS2:

```colcon build --packages-select pose_supervisor_package```

e successivamente deve essere fatto il source dell’overlay.

Nel launch file viene creata una seconda istanza di WebotsController per il follower, utilizzando ad esempio:

```namespace='follower'```

e una terza istanza per il pose_supervisor_driver.

Il controller del Supervisor non utilizza ros2_control, perché il suo compito è esclusivamente leggere e pubblicare la ground truth.

È inoltre necessario verificare il valore di coordinateSystem presente in WorldInfo nel file .wbt, perché il plugin Supervisor deve tenerne conto quando calcola lo yaw.

## Nodo ROS2 del follower

*ros2_follower_node.py* rappresenta il collegamento tra il mondo ROS2 e la policy addestrata.

Il nodo raccoglie le informazioni necessarie per costruire l’osservazione, chiama la policy e pubblica il comando di velocità del follower su:

*/follower/cmd_vel*

L’idea è che la policy non debba sapere nulla di ROS2. Il nodo ROS2 si occupa quindi di trasformare i dati provenienti dai topic nel formato esatto utilizzato durante il training.

Per la prima fase del progetto, la ground truth del Supervisor può essere utilizzata per costruire l’osservazione. In seguito, le informazioni provenienti dal Supervisor potranno essere sostituite progressivamente da stime ottenute dai sensori del robot.

Questo permette di mantenere separata la validazione dell’algoritmo dal problema, più complesso, della percezione.

Isolamento del Reinforcement Learning

Il punto centrale dell’architettura è che *rl_core/* non deve importare direttamente *rclpy*, librerie di Webots o altri componenti specifici dell’ambiente di esecuzione.

L’unica parte che rappresenta un ambiente esterno è:

*envs/leader_follower_env.py*

ma anche questo ambiente è una simulazione 2D scritta appositamente per il progetto e non richiede Webots o Gazebo.

Questo rende molto più veloce il ciclo di sviluppo: training, modifica degli iperparametri e valutazione possono essere eseguiti in pochi minuti senza dover avviare ogni volta Webots.

Inoltre, aggiungere un nuovo algoritmo non richiede modifiche allo script di training. Per esempio, per aggiungere TD3 o DDPG sarebbe sufficiente creare una nuova classe compatibile con BaseAgent e registrarla in agent_factory.py.

Lo script di training rimarrebbe invariato.

## Training e Deploy della policy
```

python scripts/train.py --algo sac --total-steps 200000
```
I risultati possono essere controllati tramite TensorBoard:
```
tensorboard --logdir runs
```
Una volta ottenuto un checkpoint soddisfacente, si può esportare l’actor:
```
python scripts/export_policy.py \
    --checkpoint checkpoints/best.pt \
    --obs-dim 9 \
    --act-dim 2 \
    --act-limit 0.6 \
    --out policy.pt \
    --onnx
```
A questo punto policy.onnx può essere copiato nella VM Linux e utilizzato dal nodo ROS2:
```
python integration/ros2_follower_node.py \
    --policy-path policy.onnx
```
Prima del test finale sarà comunque necessario verificare i nomi effettivi dei topic pubblicati dal setup Nav2 + webots_ros2, in particolare:

/leader/odom
/follower/odom

Se i topic reali hanno nomi diversi, sarà sufficiente adattare il nodo di integrazione senza modificare la policy o il codice di SAC.

L’obiettivo finale è quindi avere una pipeline separata in tre parti:

TRAINING
ambiente 2D → SAC → policy
DEPLOY
policy → runtime di inferenza → ROS2/Webots
FUTURO SIM-TO-REAL
sensori → stima dello stato → policy → cmd_vel → motori
