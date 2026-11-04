import os
import sumo_rl
import traci

os.environ["SUMO_HOME"] = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")

net_path = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2x2grid")
net_file = os.path.join(net_path, "2x2.net.xml")
route_file = os.path.join(net_path, "2x2.rou.xml")

SIMULATION_TIME = 1000
DELTA_TIME = 5
CYCLE_STEPS = 6   # 30 secondes vert / 30 secondes rouge

env = sumo_rl.parallel_env(
    net_file=net_file,
    route_file=route_file,
    use_gui=False,
    num_seconds=SIMULATION_TIME,
    delta_time=DELTA_TIME,
    yellow_time=3,
    min_green=10
)

observations, infos = env.reset()
agents = env.agents

queue_lengths = []
total_waiting_time_accumulated = 0
total_arrived_vehicles = 0
current_phase = {agent: 0 for agent in agents}
step_count = 0

while env.agents:
    # 1. Gestion des feux fixes
    actions = {}
    for agent in env.agents:
        if step_count % CYCLE_STEPS == 0:
            current_phase[agent] = 1 - current_phase[agent]
        actions[agent] = current_phase[agent]

    observations, rewards, terminations, truncations, infos = env.step(actions)
    step_count += 1

    # 2. MESURE 1 : Longueur de la file d'attente à cet instant
    current_stopped = sum(info.get("system_total_stopped", 0) for info in infos.values())
    queue_lengths.append(current_stopped)

    # 3. MESURE 2 : Cumul d'attente
    total_waiting_time_accumulated += sum(info.get("system_total_waiting_time", 0) for info in infos.values())

    # 4. MESURE 3 : Véhicules ayant terminé leur trajet à cette étape (Throughput)
    # traci.simulation.getArrivedNumber() compte les véhicules sortis du réseau
    total_arrived_vehicles += traci.simulation.getArrivedNumber()

# Nombre total de véhicules injectés dans le réseau (arrivés + encore en train de rouler)
total_vehicles_departed = traci.simulation.getDepartedNumber()

env.close()

# CALCUL FINAL DES 3 MÉTRIQUES DU PROFESSEUR
mean_queue_length = sum(queue_lengths) / len(queue_lengths)
# Temps moyen d'attente par véhicule
mean_waiting_per_vehicle = total_waiting_time_accumulated / max(1, total_arrived_vehicles)
throughput = total_arrived_vehicles

print("\n================ RÉSULTATS EXACTS (BASELINE) ================")
print(f"1. Temps d'attente moyen par véhicule : {mean_waiting_per_vehicle:.2f} s / véhicule")
print(f"2. Longueur moyenne de file d'attente : {mean_queue_length:.2f} véhicules")
print(f"3. Débit total écoulé (Throughput)    : {throughput} véhicules arrivés")
print("=============================================================")