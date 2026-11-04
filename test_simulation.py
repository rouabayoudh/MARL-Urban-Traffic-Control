import os
import sumo_rl

# Définition automatique de SUMO_HOME
os.environ["SUMO_HOME"] = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")

# Chemin vers le réseau en grille 2x2 pré-intégré
net_path = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2x2grid")
net_file = os.path.join(net_path, "2x2.net.xml")
route_file = os.path.join(net_path, "2x2.rou.xml")

print(" Chargement de l'environnement Multi-Agents (Grille 2x2)...")
env = sumo_rl.parallel_env(
    net_file=net_file,
    route_file=route_file,
    use_gui=True,                 # Mettre True pour ouvrir la fenêtre graphique SUMO
    num_seconds=100000,              # Durée de simulation (secondes)
    delta_time=15,                  # Décision toutes les 5 secondes
    yellow_time=3,                 # Jaune sécuritaire
    min_green=10                   # Temps vert minimal
)

observations, infos = env.reset()
agents = env.agents
print(f" Agents détectés aux carrefours : {agents}")
print(f" Espace d'action de chaque agent : {env.action_space(agents[0])}")
print(f" Taille de l'observation (capteurs) : {env.observation_space(agents[0]).shape}")

total_waiting_time = 0

# Boucle de 100 pas de temps
for step in range(100):
    # Les agents prennent des actions (ici actions de test aléatoires)
    actions = {agent: env.action_space(agent).sample() for agent in agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Calcul du temps d'attente cumulé du réseau
    step_waiting = sum(info.get("system_total_waiting_time", 0) for info in infos.values())
    
    if step % 20 == 0:
        print(f"  [Étape {step:03d}] Temps d'attente total actuel : {step_waiting:.1f} s")

    if not env.agents:
        break

env.close()