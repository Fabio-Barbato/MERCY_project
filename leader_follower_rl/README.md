# Leader-Follower RL — SAC isolato + deploy modulare su Webots/ROS2

Questo pacchetto risolve le due task richieste:

1. SAC (e algoritmi futuri) completamente isolato dall'ambiente reale, con training
   indipendente da ROS2/Webots.
2. Un layer di deploy modulare per portare la policy addestrata su Webots, senza
   accoppiare la policy al framework di esecuzione.

Testato in questo ambiente: training di 6k step converge (return medio da -112 a
+367), export TorchScript/ONNX verificato, i due backend di inferenza producono
la stessa azione (diff < 1e-7).

## Struttura

```
rl_core/                  <- SAC, isolato. NESSUN import di ROS2/Webots/Gymnasium-specifico.
  base_agent.py            interfaccia comune a tutti gli algoritmi (select_action, update,
                            save, load, export_actor)
  networks.py               GaussianActor (policy squashata tanh) + TwinQCritic
  sac_agent.py               implementazione SAC completa (auto-tuning di alpha incluso)
  agent_factory.py           registro algoritmo -> classe: make_agent("sac", config=...)
  replay_buffer.py           replay buffer numpy, nessuna dipendenza da torch

envs/
  leader_follower_env.py    ambiente Gymnasium 2D standalone: simula leader (pure-pursuit
                             verso waypoint casuali, stand-in per Nav2) e follower
                             (cinematica unicycle, azione = [v, w] come un cmd_vel)

scripts/
  train.py                  training loop, isolato: gira senza VM/Webots
  export_policy.py           esporta SOLO l'actor in TorchScript (.pt) e/o ONNX (.onnx)

inference/
  policy_runtime.py         runtime di inferenza MINIMALE (numpy + torch OPPURE
                             onnxruntime). E' l'unico file che il codice ROS2/Webots
                             deve importare.

integration/
  pose_supervisor_package/    pacchetto ROS2 a se' stante: plugin webots_ros2_driver
                              (init/step) che legge la pose ground-truth di leader e
                              follower e la pubblica su /leader|follower/ground_truth_pose.
                              E' un pacchetto e non un singolo script perche' nel setup
                              Webots-su-Mac + ROS2-in-VM, Webots non ha accesso a rclpy:
                              il plugin gira nella VM e si connette a Webots via <extern>,
                              lo stesso meccanismo dei driver dei due TurtleBot3.
  ros2_follower_node.py      nodo ROS2: combina le pose ground-truth con la velocita'
                              propria di ciascun robot (dalla sua odometria locale),
                              chiama la policy, pubblica /follower/cmd_vel
  multirobot_launch_example.py  esempio di launch file per webots_ros2_turtlebot
                              esteso con un secondo TurtleBot3Burger (namespace
                              "follower") + il plugin pose_supervisor — basato sul
                              pattern ufficiale multi-robot di webots_ros2, NON
                              testato su Webots reale qui
```

## Setup multi-robot (TurtleBot3 ufficiale, leader con Nav2 già funzionante)

Confermato: il leader usa il pacchetto ufficiale `webots_ros2_turtlebot` (avviato con
`ros2 launch webots_ros2_turtlebot robot_launch.py`, world/robot di default,
`robot_name TurtleBot3Burger`, driver `WebotsController`, ros2_control con
`diffdrive_controller`), Webots gira nativamente sul Mac tramite
`local_simulation_server.py` con `WEBOTS_SHARED_FOLDER`, e Nav2 viene avviato come
step separato con `turtlebot3_navigation2 navigation2.launch.py`.

Il collegamento Webots↔ROS2 avviene per nome (`robot_name` nel launch deve combaciare
col campo `name` del nodo Robot nel `.wbt`) — questo è il meccanismo che permette di
aggiungere un secondo robot senza toccare quello del leader. Lo stesso meccanismo
serve anche per il Supervisor (vedi sopra: nel vostro setup non può essere un
controller Webots "interno", deve passare per `<extern>` come i robot).

Passaggi concreti:

1. Nel `.wbt`: aggiungi un secondo `TurtleBot3Burger` con `name` diverso (es.
   `"TurtleBot3Burger_follower"`), `controller "<extern>"`, e un `DEF` proprio
   (es. `TURTLEBOT3_BURGER_FOLLOWER`, per il Supervisor). Aggiungi un `DEF` anche
   al leader se non ce l'ha già (es. `TURTLEBOT3_BURGER`). Aggiungi un terzo nodo
   `Robot { supervisor TRUE, controller "<extern>", name "pose_supervisor" }`.
2. Compila il pacchetto `integration/pose_supervisor_package/` nel tuo workspace
   ROS2 (`colcon build --packages-select pose_supervisor_package`, poi sourcia
   l'overlay).
3. Nel launch file: vedi `integration/multirobot_launch_example.py` — una seconda
   istanza di `WebotsController` per il follower (`namespace='follower'`) e una
   terza per `pose_supervisor_driver` (nessun ros2_control, solo il plugin).
4. Verifica il campo `coordinateSystem` in `WorldInfo` nel tuo `.wbt` (serve al
   plugin Supervisor per calcolare correttamente lo yaw, vedi commenti nel file).
5. Avvia tutto e lancia `ros2_follower_node.py --policy-path policy.onnx`.

**Onestà sui limiti**: non ho Webots/ROS2 disponibili in questo ambiente per testare
davvero questi file — sono una ricostruzione fedele ai pattern ufficiali (verificati
sul repository di webots_ros2 e sulla documentazione ROS2 del plugin driver), non un
setup collaudato. Il primo giro di test probabilmente farà emergere qualche dettaglio
da aggiustare (percorsi, propagazione del namespace ai controller_manager di
ros2_control, il coordinateSystem). Condividi gli errori che ottieni e li risolviamo
insieme.

## Perche' questa architettura risolve le due task

**Isolamento (task 1).** `rl_core/` non importa mai `rclpy`, `controller` (Webots) o
qualsiasi libreria legata all'ambiente reale. L'unica dipendenza "esterna" e'
`envs/leader_follower_env.py`, che e' a sua volta una simulazione 2D scritta da zero
(nessun Webots/Gazebo necessario): questo rende il ciclo iterazione-training-valutazione
questione di minuti, non di sessioni Webots. Cambiare algoritmo (TD3, DDPG, ...) significa
solo scrivere una nuova classe che implementa `BaseAgent` e registrarla in
`agent_factory.py` — lo script di training non cambia di una riga.

**Modularita' del deploy (task 2).** La policy addestrata attraversa un confine netto:
`export_policy.py` produce un file (`.pt` o `.onnx`) che non sa nulla di PyTorch training,
replay buffer, o iperparametri SAC — e' solo una funzione `osservazione -> azione`.
`inference/policy_runtime.py` la carica con dipendenze minime (idealmente solo
`onnxruntime`, molto piu' leggero di torch — utile per il deploy su target ARM/embedded).
Sopra questo runtime possono girare "shell" intercambiabili senza toccare la policy:
oggi un nodo ROS2 dentro la VM Linux (che si integra con il setup Nav2/webots_ros2 del
leader), domani — se si vuole eliminare ROS2 per il follower — un controller Webots
nativo. Entrambi costruiscono la STESSA osservazione (stesso ordine, stesse unita' di
misura di `leader_follower_env.py`) e chiamano `policy.act(obs)`.

## Come procedere

1. **Allenare**: `python scripts/train.py --algo sac --total-steps 200000`
   (su una macchina qualunque, non serve la VM/Webots). Aumenta `--total-steps`
   rispetto allo smoke-test (6k) fatto qui: per un comportamento solido servono
   tipicamente 150-300k step su un task di questa complessita'.
2. **Validare** su TensorBoard: `tensorboard --logdir runs`
3. **Esportare**: `python scripts/export_policy.py --checkpoint checkpoints/best.pt --obs-dim 9 --act-dim 2 --act-limit 0.6 --out policy.pt --onnx`
4. **Deploy**: copiare `policy.onnx` nella VM Linux e lanciare
   `integration/ros2_follower_node.py --policy-path policy.onnx`, assicurandosi che
   i topic `/leader/odom` e `/follower/odom` siano quelli effettivamente pubblicati
   dal setup Nav2 + webots_ros2 (da verificare/adattare ai nomi reali dei topic).

## Prossimi passi naturali (non ancora implementati)

- **Domain randomization** nell'ambiente di training (rumore su pose, ritardi di
  comunicazione, velocita' massime variabili) per ridurre il gap training-sim -> Webots.
- **TD3/DDPG** come seconda implementazione di `BaseAgent`, per il confronto
  multi-algoritmo che gia' fate in MERCY — qui l'interfaccia e' pronta.
- **Curriculum**: iniziare con `follow_distance`/`max_lead_dist` piu' permissivi e
  restringerli progressivamente.
- **Verifica dei nomi reali di topic/frame** del setup ROS2 (potrebbero non essere
  `/leader/odom` e `/follower/odom` di default con webots_ros2/Nav2: vanno controllati
  nel launch file del tutorial seguito).
