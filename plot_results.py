import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Vérification du fichier de données
csv_file = "training_summary.csv"
if not os.path.exists(csv_file):
    print(f"Erreur : '{csv_file}' introuvable. Avez-vous lancé 'train_dqn.py' ?")
    exit(1)

df = pd.read_csv(csv_file)

# 2. Valeurs de la Baseline
BASELINE_THROUGHPUT = 38
BASELINE_QUEUE = 144.30

# Style graphique moderne et lisible
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Benchmark : Feux Fixes (Baseline) vs Agents Intelligents (IDQN)", fontsize=16, fontweight='bold')

# --- GRAPHIQUE 1 : Évolution du Débit (Throughput) ---
ax1 = axes[0, 0]
ax1.plot(df["episode"], df["throughput"], marker='o', color='#2ca02c', linewidth=2.5, label="IDQN (Agents IA)")
ax1.axhline(y=BASELINE_THROUGHPUT, color='#d62728', linestyle='--', linewidth=2, label=f"Baseline Feux Fixes ({BASELINE_THROUGHPUT} vhc)")
ax1.set_title("1. Débit Écoulé (Throughput)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Épisode")
ax1.set_ylabel("Nombre de véhicules arrivés")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.6)

# --- GRAPHIQUE 2 : Évolution de la Longueur de File (Queue Length) ---
ax2 = axes[0, 1]
ax2.plot(df["episode"], df["avg_queue"], marker='s', color='#1f77b4', linewidth=2.5, label="IDQN (Agents IA)")
ax2.axhline(y=BASELINE_QUEUE, color='#d62728', linestyle='--', linewidth=2, label=f"Baseline Feux Fixes ({BASELINE_QUEUE:.1f} vhc)")
ax2.set_title("2. Longueur Moyenne de File d'Attente", fontsize=12, fontweight='bold')
ax2.set_xlabel("Épisode")
ax2.set_ylabel("Nombre moyen de véhicules arrêtés")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.6)

# --- GRAPHIQUE 3 : Comparaison en Barres (Baseline vs IA Finale vs IA GUI) ---
ax3 = axes[1, 0]
categories = ["Baseline (Fixe)", "IDQN (Final Ep.)", "IDQN (Démo GUI)"]
throughputs = [BASELINE_THROUGHPUT, int(df.iloc[-1]["throughput"]), 78]
colors = ['#d62728', '#1f77b4', '#2ca02c']

bars = ax3.bar(categories, throughputs, color=colors, width=0.55)
ax3.set_title("3. Comparaison du Débit Global (Véhicules arrivés)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Véhicules arrivés")
for bar in bars:
    height = bar.get_height()
    ax3.annotate(f'{height}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold')
ax3.grid(axis='y', linestyle="--", alpha=0.6)

# --- GRAPHIQUE 4 : Décroissance d'Epsilon (Exploration -> Exploitation) ---
ax4 = axes[1, 1]
ax4.plot(df["episode"], df["epsilon"], marker='^', color='#ff7f0e', linewidth=2.5)
ax4.set_title("4. Stratégie d'Exploration (Décroissance Epsilon)", fontsize=12, fontweight='bold')
ax4.set_xlabel("Épisode")
ax4.set_ylabel("Valeur d'Epsilon (Taux d'aléa)")
ax4.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
output_image = "benchmark_results.png"
plt.savefig(output_image, dpi=300)
print(f"\n Graphique généré et sauvegardé avec succès dans '{output_image}' !")