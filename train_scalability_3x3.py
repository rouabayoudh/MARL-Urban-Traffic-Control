import os
import sumo_rl
import traci
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque

os.environ["SUMO_HOME"] = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")

# --- 1. DÉTECTION AUTOMATIQUE DES FICHIERS 3X3 ---
net_path = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "3x3grid")

if not os.path.exists(net_path):
    raise FileNotFoundError(f"Dossier introuvable : {net_path}")

net_files = [os.path.join(net_path, f) for f in os.listdir(net_path) if f.endswith(".net.xml")]
route_files = [os.path.join(net_path, f) for f in os.listdir(net_path) if f.endswith(".rou.xml")]

if not net_files or not route_files:
    print(f"Fichiers présents dans le dossier : {os.listdir(net_path)}")
    raise FileNotFoundError("Fichiers .net.xml ou .rou.xml manquants dans 3x3grid.")

net_file = net_files[0]
route_file = route_files[0]

print(f"\n Réseau détecté : {os.path.basename(net_file)}")
print(f" Trafic détecté : {os.path.basename(route_file)}")

# --- 2. ARCHITECTURE DU RÉSEAU DE NEURONES ---
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

# --- 3. PARAMÈTRES D'EXPÉRIMENTATION ---
EPISODES = 10
SIM_TIME = 1000
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.90
BATCH_SIZE = 32
MEMORY_SIZE = 10000
ALPHA = 0.35  # Facteur de coopération entre voisins

print("\n--- Initialisation de la Grille 3x3 (9 Carrefours) ---")
env = sumo_rl.parallel_env(
    net_file=net_file,
    route_file=route_file,
    use_gui=True,
    num_seconds=SIM_TIME,
    delta_time=5,
    yellow_time=3,
    min_green=10,
    reward_fn="diff-waiting-time"
)

env.reset()
agents = env.agents
print(f" Agents détectés ({len(agents)} carrefours) : {agents}")

state_dim = env.observation_space(agents[0]).shape[0]
action_dim = env.action_space(agents[0]).n

# Détection automatique de la topologie spatiale via TraCI
NEIGHBORS = {}
for a1 in agents:
    pos1 = np.array(traci.junction.getPosition(a1))
    NEIGHBORS[a1] = []
    for a2 in agents:
        if a1 != a2:
            pos2 = np.array(traci.junction.getPosition(a2))
            dist = np.linalg.norm(pos1 - pos2)
            if dist <= 350:  # Distance limite pour deux carrefours voisins
                NEIGHBORS[a1].append(a2)

print("\nVoisinage spatial détecté :")
for a, n_list in NEIGHBORS.items():
    print(f"  Carrefour {a:>2} -> Voisins : {n_list}")

# Modèles indépendants pour chacun des 9 carrefours
q_networks = {agent: QNetwork(state_dim, action_dim) for agent in agents}
optimizers = {agent: optim.Adam(q_networks[agent].parameters(), lr=LR) for agent in agents}
memory = {agent: deque(maxlen=MEMORY_SIZE) for agent in agents}

epsilon = EPSILON_START
criterion = nn.MSELoss()

summary_3x3 = {
    "episode": [],
    "avg_queue": [],
    "throughput": [],
    "epsilon": []
}

print(f"\n=================================================================")
print(f"  DÉBUT ENTRAÎNEMENT SCALABILITÉ (3x3 - {len(agents)} AGENTS)")
print(f"=================================================================\n")

for episode in range(1, EPISODES + 1):
    observations, infos = env.reset()
    total_arrived = 0
    queue_lengths = []

    while env.agents:
        actions = {}
        for agent in env.agents:
            obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0)
            if random.random() < epsilon:
                actions[agent] = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    actions[agent] = torch.argmax(q_networks[agent](obs_tensor)).item()

        next_observations, raw_rewards, terminations, truncations, infos = env.step(actions)

        # Calcul coopératif pour chaque carrefour
        coop_rewards = {}
        for agent in env.agents:
            local_r = raw_rewards[agent]
            neighbors_r = sum(raw_rewards.get(n, 0) for n in NEIGHBORS.get(agent, []))
            coop_rewards[agent] = local_r + ALPHA * neighbors_r

        # Entraînement des 9 agents
        for agent in env.agents:
            memory[agent].append((
                observations[agent],
                actions[agent],
                coop_rewards[agent],
                next_observations[agent],
                terminations[agent] or truncations[agent]
            ))

            if len(memory[agent]) >= BATCH_SIZE:
                batch = random.sample(memory[agent], BATCH_SIZE)
                b_obs, b_act, b_rew, b_next, b_done = zip(*batch)

                b_obs = torch.FloatTensor(np.array(b_obs))
                b_act = torch.LongTensor(b_act).unsqueeze(1)
                b_rew = torch.FloatTensor(b_rew).unsqueeze(1)
                b_next = torch.FloatTensor(np.array(b_next))
                b_done = torch.FloatTensor(b_done).unsqueeze(1)

                curr_q = q_networks[agent](b_obs).gather(1, b_act)
                max_next_q = q_networks[agent](b_next).max(1)[0].unsqueeze(1).detach()
                target_q = b_rew + (GAMMA * max_next_q * (1 - b_done))

                loss = criterion(curr_q, target_q)
                optimizers[agent].zero_grad()
                loss.backward()
                optimizers[agent].step()

        total_arrived += traci.simulation.getArrivedNumber()
        queue_lengths.append(sum(info.get("system_total_stopped", 0) for info in infos.values()))
        observations = next_observations

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    ep_avg_queue = sum(queue_lengths) / max(1, len(queue_lengths))

    summary_3x3["episode"].append(episode)
    summary_3x3["avg_queue"].append(ep_avg_queue)
    summary_3x3["throughput"].append(total_arrived)
    summary_3x3["epsilon"].append(epsilon)

    print(f"Épisode {episode:02d}/{EPISODES} | "
          f"Queue (9 carrefours) : {ep_avg_queue:5.1f} vhc | "
          f"Débit Total : {total_arrived:3d} vhc | "
          f"Eps : {epsilon:.2f}")

env.close()

# Sauvegarde des données de scalabilité
pd.DataFrame(summary_3x3).to_csv("scalability_3x3_summary.csv", index=False)
print("\n Résultats de scalabilité sauvegardés dans 'scalability_3x3_summary.csv' !")