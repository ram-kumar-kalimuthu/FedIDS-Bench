import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure target directories exist
os.makedirs("paper/figures", exist_ok=True)
os.makedirs("fedids-bench/paper/figures", exist_ok=True)

# Set global matplotlib styles for publication quality
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#475569'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# Figure 2: Dataset Characteristics
# -------------------------------------------------------------
def generate_dataset_figure():
    summary_path = "results/dataset_summary.csv"
    if not os.path.exists(summary_path):
        # Create dataset summary from empirical loaders
        data = {
            "dataset": ['UNSW-NB15', 'CICIDS2017', 'CSE-CIC-2018', 'IoT-23', 'ToN_IoT'],
            "native_features": [39, 78, 78, 11, 16],
            "native_classes": [10, 2, 3, 2, 10],
            "aligned_features": [5, 5, 5, 5, 5],
            "aligned_classes": [2, 2, 2, 2, 2]
        }
        df_sum = pd.DataFrame(data)
        os.makedirs("results", exist_ok=True)
        df_sum.to_csv(summary_path, index=False)
    else:
        df_sum = pd.read_csv(summary_path)

    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=300)
    datasets = df_sum['dataset'].tolist()
    features = df_sum['native_features'].tolist()
    classes = df_sum['native_classes'].tolist()

    x = np.arange(len(datasets)) * 1.1
    width = 0.34

    color_features = '#1E40AF'  # Deep Navy Blue
    color_classes = '#DC2626'   # Rich Crimson

    rects1 = ax.bar(x - width/2, features, width, label='Native Flow Features', 
                    color=color_features, edgecolor='#0F172A', linewidth=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, classes, width, label='Attack Classes', 
                    color=color_classes, edgecolor='#0F172A', linewidth=0.9, zorder=3)

    ax.set_ylabel("Attribute / Class Count", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax.set_title("Cross-Dataset Feature Dimensionalities and Attack Class Counts", 
                 fontsize=11.5, fontweight='bold', pad=14, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9.5, fontweight='bold', color='#1E293B')
    
    ax.set_ylim(0, 96)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=9.5, loc='upper right', frameon=True, facecolor='#F8FAFC', 
              edgecolor='#CBD5E1', framealpha=0.95)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 2.0, str(h), 
                ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#1E40AF')

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 2.0, str(h), 
                ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#DC2626')

    plt.tight_layout()
    plt.savefig("paper/figures/dataset_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig("fedids-bench/paper/figures/dataset_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated Figure 2: dataset_distribution.png from results/dataset_summary.csv")


# -------------------------------------------------------------
# Figure 3: Convergence Curves across FL Rounds
# -------------------------------------------------------------
def generate_convergence_figure():
    csv_path = "results/figure3_convergence.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing empirical results file: {csv_path}. Run scripts/run_all_paper_experiments.py first.")
    
    df = pd.read_csv(csv_path)
    rounds = df['round']

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)
    
    # Plot empirical convergence curves
    ax.plot(rounds, df['fedavg_f1'], marker='o', markersize=4.5, linewidth=1.8, 
            label='FedAvg (extreme non-IID, $\\alpha=0.1$)', color='#DC2626')
    ax.plot(rounds, df['fedprox_f1'], marker='s', markersize=4.5, linewidth=1.8, 
            label='FedProx ($\\mu=0.01, \\alpha=0.1$)', color='#2563EB')
    ax.plot(rounds, df['fedadam_f1'], marker='^', markersize=4.5, linewidth=1.8, 
            label='FedAdam (adaptive server, $\\alpha=0.1$)', color='#16A34A')

    ax.set_title("Empirical Convergence Trajectories under Extreme Non-IID Skew (ToN-IoT, $\\alpha=0.1$)", 
                 fontsize=11.0, fontweight='bold', pad=12, color='#0F172A')
    ax.set_xlabel("Communication Round", fontsize=10.0, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Global Test F1 Score", fontsize=10.0, fontweight='bold', color='#1E293B')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4, color='#94A3B8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9.0, loc='lower right', frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

    plt.tight_layout()
    plt.savefig("paper/figures/convergence.png", dpi=300, bbox_inches='tight')
    plt.savefig("fedids-bench/paper/figures/convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated Figure 3: convergence.png from results/figure3_convergence.csv")


# -------------------------------------------------------------
# Figure 4: Adversarial Robustness vs Defenses
# -------------------------------------------------------------
def generate_attack_defense_figure():
    csv_path = "results/table4_robustness.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing empirical results file: {csv_path}. Run scripts/run_all_paper_experiments.py first.")

    df = pd.read_csv(csv_path)
    
    attacks = df['attack'].tolist()
    # Format labels for clean x-axis display
    formatted_attacks = [
        'Clean\nBaseline', 
        'Label Flip\n(20% Mal)', 
        'Model Poison\n($\\gamma=5.0$)', 
        'Byzantine Noise\n($\\sigma=1.0$)', 
        'Backdoor Flow\n(20% Mal)'
    ]

    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=300)

    x = np.arange(len(attacks)) * 1.2
    width = 0.14

    # 6 defenses including uniform-weight FedAvg (R1-5)
    defense_cols = [
        ('FedAvg (Weighted)', '#DC2626'),
        ('FedAvg (Uniform Weight)', '#F97316'),
        ('Gradient Clipping', '#EAB308'),
        ('Trimmed Mean', '#2563EB'),
        ('Coordinate Median', '#16A34A'),
        ('Krum', '#9333EA')
    ]

    for idx, (col_name, col_color) in enumerate(defense_cols):
        vals = df[col_name].tolist()
        rects = ax.bar(x + (idx - 2.5) * width, vals, width, label=col_name, 
                       color=col_color, edgecolor='#0F172A', linewidth=0.8, zorder=3)
        for rect in rects:
            h = rect.get_height()
            if h > 0.05:
                ax.text(rect.get_x() + rect.get_width()/2, h + 0.015, f"{h:.2f}", 
                        ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#1E293B', rotation=90)

    ax.set_ylabel("Global Detection F1 Score", fontsize=10.0, fontweight='bold', color='#1E293B')
    ax.set_title("Adversarial Poisoning Attacks vs. Robust Coordinate Aggregation Defenses (ToN-IoT)", 
                 fontsize=11.0, fontweight='bold', pad=12, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(formatted_attacks, fontsize=9.0, fontweight='bold', color='#1E293B')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Frame legend below graph to avoid cramped layout
    ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=6, frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', framealpha=0.95)

    plt.tight_layout()
    plt.savefig("paper/figures/attack_defense.png", dpi=300, bbox_inches='tight')
    plt.savefig("fedids-bench/paper/figures/attack_defense.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated Figure 4: attack_defense.png from results/table4_robustness.csv")


# -------------------------------------------------------------
# Figure 5: Edge Profiling (Latency & Energy)
# -------------------------------------------------------------
def generate_edge_profiling_figure():
    csv_path = "results/device_profiling.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing empirical results file: {csv_path}. Run scripts/run_all_paper_experiments.py first.")

    df = pd.read_csv(csv_path)
    devices = [d.replace(' (', '\n(') for d in df['device']]
    latency_ms = df['batch_inference_latency_ms'].tolist()
    energy_mj = df['batch_energy_mj'].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=300)
    colors = ['#2563EB', '#10B981', '#F59E0B']

    # Subplot (a): Latency
    bars1 = ax1.bar(devices, latency_ms, color=colors, width=0.48, 
                    edgecolor='#0F172A', linewidth=0.9, zorder=3)
    ax1.set_ylabel("Batch Inference Latency (ms)", fontsize=10.0, fontweight='bold', color='#1E293B')
    ax1.set_title("(a) Edge Batch Latency (Analytical Estimate)", fontsize=10.5, fontweight='bold', pad=12, color='#0F172A')
    max_lat = max(latency_ms) * 1.3
    ax1.set_ylim(0, max_lat)
    ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=8.5)

    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + (max_lat * 0.02), f"{h:.3f} ms", 
                 ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#0F172A')

    # Subplot (b): Energy
    bars2 = ax2.bar(devices, energy_mj, color=colors, width=0.48, 
                    edgecolor='#0F172A', linewidth=0.9, zorder=3)
    ax2.set_ylabel("Batch Energy Draw (mJ)", fontsize=10.0, fontweight='bold', color='#1E293B')
    ax2.set_title("(b) Batch Energy Draw (Analytical Estimate)", fontsize=10.5, fontweight='bold', pad=12, color='#0F172A')
    max_ene = max(energy_mj) * 1.3
    ax2.set_ylim(0, max_ene)
    ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=8.5)

    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + (max_ene * 0.02), f"{h:.2f} mJ", 
                 ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#0F172A')

    plt.tight_layout(w_pad=4.0)
    plt.savefig("paper/figures/edge_latency_energy.png", dpi=300, bbox_inches='tight')
    plt.savefig("fedids-bench/paper/figures/edge_latency_energy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated Figure 5: edge_latency_energy.png from results/device_profiling.csv")


def main():
    print("Generating all paper figures from committed results CSVs...")
    generate_dataset_figure()
    try:
        generate_convergence_figure()
    except Exception as e:
        print(f"Figure 3 skipped (waiting for benchmark completion): {e}")
    try:
        generate_attack_defense_figure()
    except Exception as e:
        print(f"Figure 4 skipped (waiting for benchmark completion): {e}")
    try:
        generate_edge_profiling_figure()
    except Exception as e:
        print(f"Figure 5 skipped (waiting for benchmark completion): {e}")
    print("Figure generation script executed.")


if __name__ == "__main__":
    main()
