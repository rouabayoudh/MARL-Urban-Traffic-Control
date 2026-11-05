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

# Graphe des voisins directs dans la grille 2x2
# (Dans SUMO 2x2, les agents sont nommés '1', '2', '5', '6')
NEIGHBORS = {
    '1': ['2', '5'],
    '2': ['1', '6'],
    '5': ['1', '6'],
    '6': ['2', '5']
}
ALPHA = 0.35  # Facteur de coopération (35% de partage)

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

EPISODES = 10
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

coop_summary = {
    "episode": [],
    "avg_queue": [],
    "throughput": [],
    "epsilon": []
}

print(f"\n=================================================================")
print(f"  DÉBUT ENTRAÎNEMENT COOPÉRATIF (Coop-MARL, alpha={ALPHA})")
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

        # --- CALCUL DE LA RÉCOMPENSE COOPÉRATIVE ---
        coop_rewards = {}
        for agent in env.agents:
            local_r = raw_rewards[agent]
            # Somme des récompenses des voisins
            neighbors_r = sum(raw_rewards.get(n, 0) for n in NEIGHBORS.get(agent, []))
            # Formule de coopération
            coop_rewards[agent] = local_r + ALPHA * neighbors_r

        # Entraînement avec la récompense coopérative
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

    coop_summary["episode"].append(episode)
    coop_summary["avg_queue"].append(ep_avg_queue)
    coop_summary["throughput"].append(total_arrived)
    coop_summary["epsilon"].append(epsilon)

    print(f"Épisode {episode:02d}/{EPISODES} | "
          f"Queue : {ep_avg_queue:5.1f} vhc | "
          f"Débit Coop : {total_arrived:3d} vhc (IDQN max: 83) | "
          f"Eps : {epsilon:.2f}")

env.close()

# Sauvegarde des modèles coopératifs
for agent in agents:
    torch.save(q_networks[agent].state_dict(), f"coop_agent_{agent}.pth")

pd.DataFrame(coop_summary).to_csv("training_coop_summary.csv", index=False)
print("\n Modèles coopératifs sauvegardés sous 'coop_agent_*.pth' et résumé dans 'training_coop_summary.csv' !")