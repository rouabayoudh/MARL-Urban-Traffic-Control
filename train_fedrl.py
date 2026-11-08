import os
import copy
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

net_path = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2x2grid")
net_file = os.path.join(net_path, "2x2.net.xml")
route_file = os.path.join(net_path, "2x2.rou.xml")

# --- 1. Architecture du Réseau de Neurones ---
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

# --- 2. Fonction d'Agrégation Fédérée (Federated Averaging - FedAvg) ---
def federated_averaging(local_models):
    """
    Calcule la moyenne des poids de tous les modèles locaux
    pour créer le Modèle Global Unique.
    """
    global_model = copy.deepcopy(local_models[0])
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        # Moyenne arithmétique des tenseurs de poids
        stacked_weights = torch.stack([model.state_dict()[key].float() for model in local_models])
        global_dict[key] = torch.mean(stacked_weights, dim=0)

    global_model.load_state_dict(global_dict)
    return global_model

# --- 3. Hyperparamètres d'Apprentissage Fédéré ---
TOTAL_ROUNDS = 5        # 5 rondes fédérées
LOCAL_EPISODES = 2      # 2 épisodes d'apprentissage local par ronde (Total = 10 épisodes)
SIM_TIME = 1000
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.90
BATCH_SIZE = 32
MEMORY_SIZE = 10000

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
state_dim = env.observation_space(agents[0]).shape[0]
action_dim = env.action_space(agents[0]).n

# Chaque carrefour possède son agent local
local_q_networks = {agent: QNetwork(state_dim, action_dim) for agent in agents}
optimizers = {agent: optim.Adam(local_q_networks[agent].parameters(), lr=LR) for agent in agents}
memory = {agent: deque(maxlen=MEMORY_SIZE) for agent in agents}

epsilon = EPSILON_START
criterion = nn.MSELoss()

fed_history = {
    "round": [],
    "episode": [],
    "avg_queue": [],
    "throughput": [],
    "epsilon": []
}

print(f"\n=================================================================")
print(f"  DÉBUT FEDERATED REINFORCEMENT LEARNING (FedRL) - {len(agents)} FEUX")
print(f"  Configuration : {TOTAL_ROUNDS} Rondes Fédérées x {LOCAL_EPISODES} Épisodes Locaux")
print(f"=================================================================\n")

current_global_ep = 0

for fed_round in range(1, TOTAL_ROUNDS + 1):
    print(f"\n>>> DÉMARRAGE DE LA RONDE FÉDÉRÉE {fed_round}/{TOTAL_ROUNDS} <<<")

    # Étape locale : Les feux s'entraînent indépendamment sans communiquer
    for local_ep in range(1, LOCAL_EPISODES + 1):
        current_global_ep += 1
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
                        actions[agent] = torch.argmax(local_q_networks[agent](obs_tensor)).item()

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Entraînement local sur chaque intersection
            for agent in env.agents:
                memory[agent].append((
                    observations[agent],
                    actions[agent],
                    rewards[agent],
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

                    curr_q = local_q_networks[agent](b_obs).gather(1, b_act)
                    max_next_q = local_q_networks[agent](b_next).max(1)[0].unsqueeze(1).detach()
                    target_q = b_rew + (GAMMA * max_next_q * (1 - b_done))

                    loss = criterion(curr_q, target_q)
                    optimizers[agent].zero_grad()
                    loss.backward()
                    optimizers[agent].step()

            total_arrived += traci.simulation.getArrivedNumber()
            queue_lengths.append(sum(info.get("system_total_stopped", 0) for info in infos.values()))
            observations = next_observations

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        avg_q = sum(queue_lengths) / max(1, len(queue_lengths))

        fed_history["round"].append(fed_round)
        fed_history["episode"].append(current_global_ep)
        fed_history["avg_queue"].append(avg_q)
        fed_history["throughput"].append(total_arrived)
        fed_history["epsilon"].append(epsilon)

        print(f"  [Ronde {fed_round} | Épisode {local_ep}] Queue : {avg_q:5.1f} vhc | Débit : {total_arrived:2d} vhc | Eps : {epsilon:.2f}")

    # --- ÉTAPE DE SYNCHRONISATION FÉDÉRÉE (FedAvg) ---
    print(f"  [FedAvg] Agrégation des poids des {len(agents)} carrefours vers le modèle global...")
    global_model = federated_averaging(list(local_q_networks.values()))

    # Redistribution du modèle global à tous les carrefours
    for agent in agents:
        local_q_networks[agent].load_state_dict(global_model.state_dict())
    print("  [FedAvg] Modèle global redistribué avec succès à toutes les intersections !")

env.close()

# Sauvegarde du modèle fédéré final
torch.save(global_model.state_dict(), "global_federated_model.pth")
pd.DataFrame(fed_history).to_csv("fedrl_summary.csv", index=False)

print("\n" + "="*70)
print(" ENTRAÎNEMENT FÉDÉRÉ TERMINÉ AVEC SUCCÈS !")
print(" - Modèle global unique sauvegardé sous : 'global_federated_model.pth'")
print(" - Historique sauvegardé sous : 'fedrl_summary.csv'")
print("="*70 + "\n")