import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure target directories exist
os.makedirs("paper/figures", exist_ok=True)
os.makedirs("fedids-bench/paper/figures", exist_ok=True)

# Set global matplotlib styles for publication quality
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#475569'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# Figure 2: Dataset Characteristics (Revised: No Overlaps, Generous Headroom)
# -------------------------------------------------------------
def generate_dataset_figure():
    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=300)
    datasets = ['UNSW-NB15', 'CICIDS2017', 'CSE-CIC-2018', 'IoT-23', 'ToN_IoT']
    features = [39, 78, 78, 11, 16]
    classes = [10, 2, 3, 2, 10]

    x = np.arange(len(datasets)) * 1.1
    width = 0.34

    # High-contrast publication colors
    color_features = '#1E40AF'  # Deep Navy Blue
    color_classes = '#DC2626'   # Rich Crimson

    rects1 = ax.bar(x - width/2, features, width, label='Flow Features', 
                    color=color_features, edgecolor='#0F172A', linewidth=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, classes, width, label='Attack Classes', 
                    color=color_classes, edgecolor='#0F172A', linewidth=0.9, zorder=3)

    ax.set_ylabel("Attribute / Class Count", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax.set_title("Cross-Dataset Feature Dimensionalities and Attack Class Counts", 
                 fontsize=11.5, fontweight='bold', pad=14, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9.5, fontweight='bold', color='#1E293B')
    
    # Ample headroom to ensure text never touches the top spine
    ax.set_ylim(0, 96)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Clean, framed legend with generous spacing
    ax.legend(fontsize=9.5, loc='upper right', frameon=True, facecolor='#F8FAFC', 
              edgecolor='#CBD5E1', framealpha=0.95)

    # Crisp value annotations above each bar
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
    print("Successfully generated revised Figure 2: dataset_distribution.png")

# -------------------------------------------------------------
# Figure 4: Adversarial Robustness vs Defenses (Revised: Legend Outside, No Overlap)
# -------------------------------------------------------------
def generate_attack_defense_figure():
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)

    attacks = [
        'Clean Baseline\n(0% Malicious)', 
        'Label Flip\n(20% Malicious)', 
        'Model Poisoning\n(20% Malicious)', 
        'Byzantine Noise\n(20% Malicious)', 
        'Backdoor Flow\n(20% Malicious)'
    ]
    
    # F1 scores from experimental benchmarks (Table 4)
    no_defense = [0.938, 0.612, 0.185, 0.124, 0.540]
    clipping =   [0.931, 0.635, 0.742, 0.680, 0.610]
    trim_mean =  [0.925, 0.884, 0.910, 0.895, 0.820]
    median =     [0.918, 0.871, 0.898, 0.882, 0.805]
    krum =       [0.912, 0.865, 0.887, 0.875, 0.790]

    x = np.arange(len(attacks)) * 1.05
    width = 0.15

    # 5 high-contrast, accessible academic colors
    c_nodef = '#DC2626'   # Crimson
    c_clip  = '#EA580C'   # Amber Orange
    c_trim  = '#2563EB'   # Royal Blue
    c_med   = '#16A34A'   # Emerald Green
    c_krum  = '#9333EA'   # Violet

    ax.bar(x - 2*width, no_defense, width, label='No Defense (FedAvg)', 
           color=c_nodef, edgecolor='#0F172A', linewidth=0.8, zorder=3)
    ax.bar(x - width,   clipping,   width, label='Norm Clipping', 
           color=c_clip,  edgecolor='#0F172A', linewidth=0.8, zorder=3)
    ax.bar(x,           trim_mean,  width, label='Trimmed Mean', 
           color=c_trim,  edgecolor='#0F172A', linewidth=0.8, zorder=3)
    ax.bar(x + width,   median,     width, label='Coord-wise Median', 
           color=c_med,   edgecolor='#0F172A', linewidth=0.8, zorder=3)
    ax.bar(x + 2*width, krum,       width, label='Multi-Krum', 
           color=c_krum,  edgecolor='#0F172A', linewidth=0.8, zorder=3)

    ax.set_ylabel("Global Test F1-Score", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax.set_title("Adversarial Poisoning Attack Degradation and Robust Defense Recovery", 
                 fontsize=11.5, fontweight='bold', pad=28, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=9.0, fontweight='bold', color='#1E293B')
    
    # Y-limit leaves room for grid and annotations without crowding
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # CRITICAL: Place legend completely outside the plot above the chart to avoid ANY bar overlap
    ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=5, frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', framealpha=0.95)

    plt.tight_layout()
    plt.savefig("paper/figures/attack_defense.png", dpi=300, bbox_inches='tight')
    plt.savefig("fedids-bench/paper/figures/attack_defense.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated revised Figure 4: attack_defense.png")

# -------------------------------------------------------------
# Figure 5: Edge Profiling (Revised: Headroom for Text, Clean Subplots)
# -------------------------------------------------------------
def generate_edge_profiling_figure():
    devices = ['Raspberry Pi 4\n(Cortex-A72)', 'NVIDIA Jetson\nOrin Nano', 'Laptop CPU\n(Intel Core i7)']
    latency_ms = [0.706, 0.024, 0.095]      # ms per sample
    energy_mj =  [3.525, 0.357, 2.668]      # mJ per batch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=300)
    colors = ['#2563EB', '#10B981', '#F59E0B']

    # Subplot (a): Latency
    bars1 = ax1.bar(devices, latency_ms, color=colors, width=0.52, 
                    edgecolor='#0F172A', linewidth=0.9, zorder=3)
    ax1.set_ylabel("Per-Sample Latency (ms)", fontsize=10.0, fontweight='bold', color='#1E293B')
    ax1.set_title("(a) Edge Inference Latency", fontsize=11.0, fontweight='bold', pad=12, color='#0F172A')
    ax1.set_ylim(0, 0.88)  # 25% headroom above 0.706 ms
    ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=8.8)

    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.025, f"{h:.3f} ms", 
                 ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#0F172A')

    # Subplot (b): Energy
    bars2 = ax2.bar(devices, energy_mj, color=colors, width=0.52, 
                    edgecolor='#0F172A', linewidth=0.9, zorder=3)
    ax2.set_ylabel("Batch Energy Consumption (mJ)", fontsize=10.0, fontweight='bold', color='#1E293B')
    ax2.set_title("(b) Batch Energy Consumption", fontsize=11.0, fontweight='bold', pad=12, color='#0F172A')
    ax2.set_ylim(0, 4.5)  # 28% headroom above 3.525 mJ
    ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8', zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=8.8)

    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.12, f"{h:.3f} mJ", 
                 ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#0F172A')

    plt.tight_layout()
    plt.savefig("paper/figures/edge_latency_energy.png", dpi=300, bbox_inches='tight')
    plt.savefig("fedids-bench/paper/figures/edge_latency_energy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated revised Figure 5: edge_latency_energy.png")

if __name__ == "__main__":
    generate_dataset_figure()
    generate_attack_defense_figure()
    generate_edge_profiling_figure()
    print("Figures 2, 4, and 5 successfully regenerated with zero overlapping elements!")
