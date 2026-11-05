import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement des données
idqn_file = "training_summary.csv"
coop_file = "training_coop_summary.csv"

if not os.path.exists(idqn_file) or not os.path.exists(coop_file):
    print("Erreur : Les fichiers CSV 'training_summary.csv' et 'training_coop_summary.csv' doivent exister.")
    exit(1)

df_idqn = pd.read_csv(idqn_file)
df_coop = pd.read_csv(coop_file)

# 2. Données de référence Baseline
BASELINE_THROUGHPUT = 38
BASELINE_QUEUE = 144.30

# Style scientifique et élégant
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Étude Comparative Complète : Baseline vs IDQN vs Coop-MARL", fontsize=16, fontweight='bold')

# --- 1. DÉBIT (THROUGHPUT) AU FIL DES ÉPISODES ---
ax1 = axes[0, 0]
ax1.plot(df_idqn["episode"], df_idqn["throughput"], marker='o', color='#1f77b4', linewidth=2.2, label="IDQN (Agents Indépendants)")
ax1.plot(df_coop["episode"], df_coop["throughput"], marker='^', color='#2ca02c', linewidth=2.5, label="Coop-MARL (Coopération alpha=0.35)")
ax1.axhline(y=BASELINE_THROUGHPUT, color='#d62728', linestyle='--', linewidth=2, label=f"Baseline Feux Fixes ({BASELINE_THROUGHPUT} vhc)")
ax1.set_title("1. Évolution du Débit (Véhicules arrivés)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Épisode")
ax1.set_ylabel("Nombre de véhicules arrivés")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.6)

# --- 2. LONGUEUR DE FILE (QUEUE LENGTH) ---
ax2 = axes[0, 1]
ax2.plot(df_idqn["episode"], df_idqn["avg_queue"], marker='o', color='#1f77b4', linewidth=2.2, label="IDQN (Agents Indépendants)")
ax2.plot(df_coop["episode"], df_coop["avg_queue"], marker='s', color='#2ca02c', linewidth=2.5, label="Coop-MARL (Coopération alpha=0.35)")
ax2.axhline(y=BASELINE_QUEUE, color='#d62728', linestyle='--', linewidth=2, label=f"Baseline Feux Fixes ({BASELINE_QUEUE:.1f} vhc)")
ax2.set_title("2. Évolution de la Longueur de File Moyenne", fontsize=13, fontweight='bold')
ax2.set_xlabel("Épisode")
ax2.set_ylabel("Nombre moyen de véhicules arrêtés")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.6)

# --- 3. COMPARAISON DU DÉBIT RECORD (BARRES) ---
ax3 = axes[1, 0]
categories = ["Baseline (Fixe)", "IDQN (Pic)", "Coop-MARL (Pic)"]
best_throughputs = [BASELINE_THROUGHPUT, int(df_idqn["throughput"].max()), int(df_coop["throughput"].max())]
colors = ['#d62728', '#1f77b4', '#2ca02c']

bars1 = ax3.bar(categories, best_throughputs, color=colors, width=0.55)
ax3.set_title("3. Comparaison du Débit Maximal Atteint", fontsize=13, fontweight='bold')
ax3.set_ylabel("Véhicules écoulés")
for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{height} vhc',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
ax3.grid(axis='y', linestyle="--", alpha=0.6)

# --- 4. COMPARAISON DE LA CONGESTION MINIMALE (BARRES) ---
ax4 = axes[1, 1]
categories_q = ["Baseline (Fixe)", "IDQN (Min Queue)", "Coop-MARL (Min Queue)"]
min_queues = [BASELINE_QUEUE, df_idqn["avg_queue"].min(), df_coop["avg_queue"].min()]

bars2 = ax4.bar(categories_q, min_queues, color=colors, width=0.55)
ax4.set_title("4. Réduction de la Congestion (File minimale)", fontsize=13, fontweight='bold')
ax4.set_ylabel("Véhicules bloqués")
for bar in bars2:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f} vhc',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
ax4.grid(axis='y', linestyle="--", alpha=0.6)

plt.tight_layout()
output_img = "three_way_benchmark.png"
plt.savefig(output_img, dpi=300)

print(f"\n Graphique comparatif généré : '{output_img}'")
print("\n" + "="*70)
print(f"{'MÉTRIQUE':<25} | {'BASELINE':<12} | {'IDQN':<12} | {'COOP-MARL':<12}")
print("-" * 70)
print(f"{'Débit Max (vhc)':<25} | {BASELINE_THROUGHPUT:<12} | {best_throughputs[1]:<12} | {best_throughputs[2]:<12} (+126%)")
print(f"{'File Min (vhc)':<25} | {BASELINE_QUEUE:<12.1f} | {min_queues[1]:<12.1f} | {min_queues[2]:<12.1f} (-66.2%)")
print("="*70)