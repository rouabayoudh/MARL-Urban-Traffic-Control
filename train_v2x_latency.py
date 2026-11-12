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
import matplotlib.pyplot as plt
from collections import deque

os.environ["SUMO_HOME"] = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")

net_path = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2x2grid")
net_file = os.path.join(net_path, "2x2.net.xml")
route_file = os.path.join(net_path, "2x2.rou.xml")

# =====================================================================
#   DÉFINITION DES 4 SCÉNARIOS RÉSEAU RÉALISTES (5G/V2X/C-V2X)
# =====================================================================
# Chaque scénario correspond à une condition radio réelle mesurée
# dans les déploiements 5G NR et C-V2X (3GPP TS 22.185)
NETWORK_SCENARIOS = [
    {
        "name": "Canal Parfait (Idéal)",
        "label": "Idéal (0s / 0%)",
        "latency_steps": 0,       # Aucun délai
        "packet_loss": 0.00,      # Aucune perte
        "color": "#2ca02c"
    },
    {
        "name": "5G URLLC (Réaliste)",
        "label": "5G URLLC (1s / 5%)",
        "latency_steps": 1,       # 1 pas * 5s = 5 secondes de latence
        "packet_loss": 0.05,      # 5% de perte de paquets
        "color": "#1f77b4"
    },
    {
        "name": "4G LTE (Congestionné)",
        "label": "4G LTE (3s / 10%)",
        "latency_steps": 3,       # 3 pas * 5s = 15 secondes de latence
        "packet_loss": 0.10,      # 10% de perte
        "color": "#ff7f0e"
    },
    {
        "name": "Réseau Dégradé (Critique)",
        "label": "Dégradé (5s / 20%)",
        "latency_steps": 5,       # 5 pas * 5s = 25 secondes de latence
        "packet_loss": 0.20,      # 20% de perte
        "color": "#d62728"
    }
]

# Topologie des voisins (Grille 2x2)
NEIGHBORS = {
    '1': ['2', '5'],
    '2': ['1', '6'],
    '5': ['1', '6'],
    '6': ['2', '5']
}
ALPHA = 0.35

# Résultats de référence (pour comparaison finale)
BASELINE_THROUGHPUT = 38
BASELINE_QUEUE = 144.30

# =====================================================================
#   ARCHITECTURE DU RÉSEAU DE NEURONES (Identique aux autres scripts)
# =====================================================================
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

# Hyperparamètres
EPISODES = 8
SIM_TIME = 1000
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.88
BATCH_SIZE = 32
MEMORY_SIZE = 10000

# Stockage de tous les résultats pour comparaison finale
all_results = {}

# =====================================================================
#   BOUCLE PRINCIPALE : Test de chaque scénario réseau
# =====================================================================
for scenario in NETWORK_SCENARIOS:

    print(f"\n{'='*70}")
    print(f"  SCÉNARIO RÉSEAU : {scenario['name']}")
    print(f"  Latence : {scenario['latency_steps'] * 5}s | Perte de paquets : {scenario['packet_loss']*100:.0f}%")
    print(f"{'='*70}\n")

    # Initialisation de l'environnement pour ce scénario
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

    # Modèles locaux pour ce scénario
    q_networks = {agent: QNetwork(state_dim, action_dim) for agent in agents}
    optimizers = {agent: optim.Adam(q_networks[agent].parameters(), lr=LR) for agent in agents}
    memory = {agent: deque(maxlen=MEMORY_SIZE) for agent in agents}

    # Buffer de latence : file d'attente pour simuler le délai réseau
    latency_buffers = {agent: {neighbor: deque(maxlen=10) for neighbor in NEIGHBORS.get(agent, [])} for agent in agents}

    epsilon = EPSILON_START
    criterion = nn.MSELoss()

    scenario_throughputs = []
    scenario_queues = []

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

            # ===========================================================
            #   SIMULATION DE LA COMMUNICATION V2X NON-FIABLE
            #   (Latence + Perte de Paquets sur le canal radio 5G/C-V2X)
            # ===========================================================
            coop_rewards = {}
            for agent in env.agents:
                local_r = raw_rewards[agent]
                neighbors_r = 0.0

                for neighbor in NEIGHBORS.get(agent, []):
                    # Étape 1 : Perte de paquets (canal radio bruité)
                    # Un paquet est perdu avec probabilité packet_loss
                    if random.random() < scenario["packet_loss"]:
                        # Paquet perdu → l'agent ne reçoit rien de ce voisin
                        continue

                    # Étape 2 : Mise en buffer de latence
                    # La récompense du voisin est mise en attente dans le buffer
                    latency_buffers[agent][neighbor].append(raw_rewards.get(neighbor, 0.0))

                    # Étape 3 : Lecture avec délai (latency_steps)
                    # On ne lit que si le buffer a accumulé assez de valeurs
                    if len(latency_buffers[agent][neighbor]) > scenario["latency_steps"]:
                        # Récupération de la valeur la plus ancienne (délai simulé)
                        delayed_reward = latency_buffers[agent][neighbor][0]
                        neighbors_r += delayed_reward

                # Récompense coopérative avec signal réseau dégradé
                coop_rewards[agent] = local_r + ALPHA * neighbors_r

            # Entraînement des agents
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
        avg_q = sum(queue_lengths) / max(1, len(queue_lengths))
        scenario_throughputs.append(total_arrived)
        scenario_queues.append(avg_q)

        print(f"  Épisode {episode:02d}/{EPISODES} | Queue : {avg_q:5.1f} vhc | Débit : {total_arrived:3d} vhc | Eps : {epsilon:.2f}")

    env.close()

    # Sauvegarde des résultats de ce scénario
    all_results[scenario["label"]] = {
        "throughputs": scenario_throughputs,
        "queues": scenario_queues,
        "color": scenario["color"]
    }

    # Export CSV pour chaque scénario
    csv_name = f"v2x_{scenario['name'].replace(' ', '_').replace('/', '_')}.csv"
    pd.DataFrame({
        "episode": range(1, EPISODES + 1),
        "throughput": scenario_throughputs,
        "avg_queue": scenario_queues
    }).to_csv(csv_name, index=False)
    print(f"\n Résultats sauvegardés dans '{csv_name}'")

# =====================================================================
#   GÉNÉRATION DU GRAPHIQUE COMPARATIF 5G/V2X
# =====================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Robustesse du MARL face aux Dégradations Réseau 5G / C-V2X\n"
             "(Impact de la Latence et des Pertes de Paquets sur les Performances ITS)",
             fontsize=13, fontweight='bold')

episodes = range(1, EPISODES + 1)

# Graphique 1 : Débit par scénario
ax1 = axes[0]
ax1.axhline(y=BASELINE_THROUGHPUT, color='black', linestyle=':', linewidth=2,
            label=f"Baseline Feux Fixes ({BASELINE_THROUGHPUT} vhc)")
for label, data in all_results.items():
    ax1.plot(episodes, data["throughputs"], marker='o', linewidth=2.2,
             color=data["color"], label=label)
ax1.set_title("1. Débit (Throughput) selon la Qualité du Réseau", fontsize=11, fontweight='bold')
ax1.set_xlabel("Épisode d'entraînement")
ax1.set_ylabel("Nombre de véhicules écoulés")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.6)

# Graphique 2 : File d'attente par scénario
ax2 = axes[1]
ax2.axhline(y=BASELINE_QUEUE, color='black', linestyle=':', linewidth=2,
            label=f"Baseline Feux Fixes ({BASELINE_QUEUE} vhc)")
for label, data in all_results.items():
    ax2.plot(episodes, data["queues"], marker='s', linewidth=2.2,
             color=data["color"], label=label)
ax2.set_title("2. Longueur de File selon la Qualité du Réseau", fontsize=11, fontweight='bold')
ax2.set_xlabel("Épisode d'entraînement")
ax2.set_ylabel("Nombre moyen de véhicules bloqués")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("v2x_latency_benchmark.png", dpi=300)

# =====================================================================
#   TABLEAU RÉCAPITULATIF FINAL
# =====================================================================
print("\n" + "="*72)
print("        BILAN FINAL : ROBUSTESSE DU MARL AUX CONDITIONS RÉSEAU V2X")
print("="*72)
print(f"{'SCÉNARIO':<28} | {'Débit Max':<12} | {'Queue Min':<12} | {'vs Baseline'}")
print("-"*72)
print(f"{'Baseline (Feux Fixes)':<28} | {BASELINE_THROUGHPUT:<12} | {BASELINE_QUEUE:<12.1f} | Référence")
for label, data in all_results.items():
    max_t = max(data["throughputs"])
    min_q = min(data["queues"])
    gain = ((max_t - BASELINE_THROUGHPUT) / BASELINE_THROUGHPUT) * 100
    print(f"{label:<28} | {max_t:<12} | {min_q:<12.1f} | +{gain:.1f}% de débit")
print("="*72)
print(f"\n Graphique sauvegardé : 'v2x_latency_benchmark.png'")