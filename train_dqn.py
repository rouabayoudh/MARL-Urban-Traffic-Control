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

net_path = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2x2grid")
net_file = os.path.join(net_path, "2x2.net.xml")
route_file = os.path.join(net_path, "2x2.rou.xml")

# Valeurs de votre Baseline (Feux Fixes) pour comparaison automatique
BASELINE_QUEUE = 144.30
BASELINE_THROUGHPUT = 38
BASELINE_WAITING = 312197.16

# --- 1. Réseau de neurones DQN ---
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

# --- 2. Hyperparamètres ---
EPISODES = 10         # Ajusté à 10 épisodes pour une meilleure convergence
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
    use_gui=False,
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

q_networks = {agent: QNetwork(state_dim, action_dim) for agent in agents}
optimizers = {agent: optim.Adam(q_networks[agent].parameters(), lr=LR) for agent in agents}
memory = {agent: deque(maxlen=MEMORY_SIZE) for agent in agents}

epsilon = EPSILON_START
criterion = nn.MSELoss()

# Historique pour le récapitulatif final
summary_history = {
    "episode": [],
    "avg_queue": [],
    "throughput": [],
    "waiting_per_vehicle": [],
    "epsilon": []
}

print(f"\n=================================================================")
print(f"  DÉBUT ENTRAÎNEMENT MARL (IDQN) - {len(agents)} CARREFOURS")
print(f"  Baseline Référence : Queue={BASELINE_QUEUE:.1f} | Débit={BASELINE_THROUGHPUT} vhc")
print(f"=================================================================\n")

for episode in range(1, EPISODES + 1):
    observations, infos = env.reset()
    total_arrived = 0
    total_waiting_accumulated = 0
    queue_lengths = []

    while env.agents:
        actions = {}
        # Sélection d'action Epsilon-Greedy
        for agent in env.agents:
            obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0)
            if random.random() < epsilon:
                actions[agent] = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = q_networks[agent](obs_tensor)
                    actions[agent] = torch.argmax(q_values).item()

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Mémorisation & Apprentissage
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

                curr_q = q_networks[agent](b_obs).gather(1, b_act)
                max_next_q = q_networks[agent](b_next).max(1)[0].unsqueeze(1).detach()
                target_q = b_rew + (GAMMA * max_next_q * (1 - b_done))

                loss = criterion(curr_q, target_q)
                optimizers[agent].zero_grad()
                loss.backward()
                optimizers[agent].step()

        # Métriques
        total_arrived += traci.simulation.getArrivedNumber()
        total_waiting_accumulated += sum(info.get("system_total_waiting_time", 0) for info in infos.values())
        current_stopped = sum(info.get("system_total_stopped", 0) for info in infos.values())
        queue_lengths.append(current_stopped)

        observations = next_observations

    # Décroissance Epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Calcul des métriques de l'épisode
    ep_avg_queue = sum(queue_lengths) / max(1, len(queue_lengths))
    ep_waiting_per_veh = total_waiting_accumulated / max(1, total_arrived)

    summary_history["episode"].append(episode)
    summary_history["avg_queue"].append(ep_avg_queue)
    summary_history["throughput"].append(total_arrived)
    summary_history["waiting_per_vehicle"].append(ep_waiting_per_veh)
    summary_history["epsilon"].append(epsilon)

    print(f"Épisode {episode:02d}/{EPISODES} | "
          f"Queue : {ep_avg_queue:5.1f} vhc | "
          f"Débit : {total_arrived:3d} vhc | "
          f"Attente/vhc : {ep_waiting_per_veh:8.1f} s | "
          f"Eps : {epsilon:.2f}")

env.close()

# Sauvegarde des modèles
for agent in agents:
    torch.save(q_networks[agent].state_dict(), f"dqn_agent_{agent}.pth")

# Sauvegarde des données CSV
df_summary = pd.DataFrame(summary_history)
df_summary.to_csv("training_summary.csv", index=False)

# =====================================================================
#                      RÉSUMÉ FINAL COMPARATIF
# =====================================================================
best_episode = df_summary.loc[df_summary["throughput"].idxmax()]
final_episode = df_summary.iloc[-1]

throughput_gain = ((final_episode["throughput"] - BASELINE_THROUGHPUT) / BASELINE_THROUGHPUT) * 100
queue_reduction = ((BASELINE_QUEUE - final_episode["avg_queue"]) / BASELINE_QUEUE) * 100
waiting_reduction = ((BASELINE_WAITING - final_episode["waiting_per_vehicle"]) / BASELINE_WAITING) * 100

print("\n" + "="*70)
print("                     BILAN COMPARATIF FINAL")
print("="*70)
print(f"{'MÉTRIQUE':<32} | {'BASELINE (Fixe)':<16} | {'IA (Final)':<15} | {'GAIN'}")
print("-" * 70)
print(f"{'Débit (Véhicules arrivés)':<32} | {BASELINE_THROUGHPUT:<16} | {int(final_episode['throughput']):<15} | +{throughput_gain:.1f}%")
print(f"{'File moyenne (Queue)':<32} | {BASELINE_QUEUE:<16.1f} | {final_episode['avg_queue']:<15.1f} | {queue_reduction:+.1f}%")
print(f"{'Attente par véhicule (s)':<32} | {BASELINE_WAITING:<16.1f} | {final_episode['waiting_per_vehicle']:<15.1f} | {waiting_reduction:+.1f}%")
print("="*70)
print(f" Meilleur débit atteint : Épisode {int(best_episode['episode'])} avec {int(best_episode['throughput'])} véhicules.")
print(f" Fichier de synthèse sauvegardé : 'training_summary.csv'")
print(f" Modèles IA enregistrés : 'dqn_agent_*.pth'")
print("="*70 + "\n")