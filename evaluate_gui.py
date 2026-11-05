import os
import time
import sumo_rl
import traci
import torch
import torch.nn as nn
import numpy as np

os.environ["SUMO_HOME"] = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")

net_path = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2x2grid")
net_file = os.path.join(net_path, "2x2.net.xml")
route_file = os.path.join(net_path, "2x2.rou.xml")

# --- 1. Même architecture de réseau que pour l'entraînement ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 2. Création de l'environnement AVEC INTERFACE GRAPHIQUE ---
print("\n Lancement de SUMO-GUI...")
env = sumo_rl.parallel_env(
    net_file=net_file,
    route_file=route_file,
    use_gui=True,          # Ouvre l'interface graphique SUMO
    num_seconds=1000,      # Durée de la simulation
    delta_time=5,
    yellow_time=3,
    min_green=10
)

observations, infos = env.reset()
agents = env.agents
state_dim = env.observation_space(agents[0]).shape[0]
action_dim = env.action_space(agents[0]).n

# --- 3. Chargement des modèles entraînés ---
q_networks = {}
for agent in agents:
    model_path = f"dqn_agent_{agent}.pth"
    net = QNetwork(state_dim, action_dim)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        net.eval()
        print(f" Modèle chargé pour l'intersection : {agent}")
    else:
        print(f" Attention: {model_path} non trouvé, l'agent agira au hasard.")
    q_networks[agent] = net

print("\n--- Simulation en cours (Regardez la fenêtre SUMO-GUI) ---")
print("Astuce : Dans SUMO-GUI, ajustez le curseur 'Delay (ms)' en haut pour ralentir ou accélérer la vue.\n")

step_count = 0
total_arrived = 0

while env.agents:
    actions = {}
    for agent in env.agents:
        obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0)
        with torch.no_grad():
            q_values = q_networks[agent](obs_tensor)
            # Décision 100% intelligente (Argmax)
            actions[agent] = torch.argmax(q_values).item()

    observations, rewards, terminations, truncations, infos = env.step(actions)
    step_count += 1
    total_arrived += traci.simulation.getArrivedNumber()

    current_queue = sum(info.get("system_total_stopped", 0) for info in infos.values())
    if step_count % 10 == 0:
        print(f"Pas {step_count:03d} | Véhicules passés : {total_arrived:3d} | File actuelle : {current_queue:2d} voitures")

    # Petit délai pour que l'œil humain puisse apprécier les mouvements
    time.sleep(0.05)

env.close()
print(f"\n Démo terminée ! Total véhicules ayant traversé la ville : {total_arrived}")