import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def main():
    data_path = 'determinism/determinism_data.pt'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    data = torch.load(data_path, map_location='cpu')
    
    NUM_TRIALS = data["NUM_TRIALS"]
    drift_both_logits = data["drift_both_logits"]
    drift_both_probs = data["drift_both_probs"]
    drift_bs_only_logits = data["drift_bs_only_logits"]
    drift_bs_only_probs = data["drift_bs_only_probs"]
    
    flip_both_count = data.get("flip_both_count", 0)
    flip_bs_only_count = data.get("flip_bs_only_count", 0)
    flip_both_indices = data.get("flip_both_indices", [])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"], 
        "font.size": 32,
        "axes.labelsize": 36,
        "axes.labelweight": "bold",
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
        'hatch.linewidth': 1.0,
        'figure.dpi': 300
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)
    
    color_baseline = '#d62728' 
    color_cogpu = '#1f77b4'
    color_flip = '#ff7f0e'

    line_w = 2.0
    marker_s = 12
    flip_marker_s = 350

    # ==========================================
    # 子图 1: Logits
    # ==========================================
    ax1.axhline(0, color='gray', linewidth=4.0, linestyle='--', alpha=0.7, zorder=1)
    ax1.plot(range(NUM_TRIALS), drift_bs_only_logits, color=color_cogpu, alpha=0.9, marker='s', markersize=marker_s, linewidth=line_w, zorder=2)
    ax1.plot(range(NUM_TRIALS), drift_both_logits, color=color_baseline, alpha=0.85, marker='o', markersize=marker_s, linewidth=line_w, zorder=3)

    if flip_both_indices:
        flip_logits_values = [drift_both_logits[i] for i in flip_both_indices]
        ax1.scatter(flip_both_indices, flip_logits_values, marker='X', s=flip_marker_s, color=color_flip, edgecolor='black', linewidth=1.5, zorder=5)
        for idx in flip_both_indices:
            ax1.axvspan(idx - 0.5, idx + 0.5, color=color_flip, alpha=0.15, zorder=0)
    
    ax1.set_title('Stage 1: LM Head MatMul Logit Drift', fontsize=36, fontweight='bold', pad=15)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # ==========================================
    # 子图 2: Probs
    # ==========================================
    ax2.axhline(0, color='gray', linewidth=4.0, linestyle='--', alpha=0.7, zorder=1)
    ax2.plot(range(NUM_TRIALS), drift_bs_only_probs, color=color_cogpu, alpha=0.9, marker='s', markersize=marker_s, linewidth=line_w, zorder=2)
    ax2.plot(range(NUM_TRIALS), drift_both_probs, color=color_baseline, alpha=0.85, marker='o', markersize=marker_s, linewidth=line_w, zorder=3)

    if flip_both_indices:
        flip_probs_values = [drift_both_probs[i] for i in flip_both_indices]
        ax2.scatter(flip_both_indices, flip_probs_values, marker='X', s=flip_marker_s, color=color_flip, edgecolor='black', linewidth=1.5, zorder=5)
        for idx in flip_both_indices:
            ax2.axvspan(idx - 0.5, idx + 0.5, color=color_flip, alpha=0.15, zorder=0)

    ax2.set_title('Stage 2: Softmax Probability Drift', fontsize=36, fontweight='bold', pad=15)
    ax2.set_xlabel('Trial Number', fontsize=36, fontweight='bold')
    
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.supylabel('Max Absolute Drift', fontweight='bold', fontsize=34, x=0.03)

    # ==========================================
    # 图例处理
    # ==========================================
    label_baseline = f"Baseline"
    label_cogpu = f"CoGPU"
    label_zero = "Absolute Determinism"
    label_flip = f"Argmax Flip Occurred ({flip_both_count} times)"

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=color_baseline, lw=line_w, marker='o', markersize=16), 
        Line2D([0], [0], color=color_cogpu, lw=line_w, marker='s', markersize=16),
        Line2D([0], [0], color='gray', lw=4.0, linestyle='--', alpha=0.7),
        Line2D([0], [0], marker='X', color='w', markerfacecolor=color_flip, markeredgecolor='black', markersize=20, markeredgewidth=1.5)
    ]
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17) 
    
    fig.legend(custom_lines, [label_baseline, label_cogpu, label_zero, label_flip], 
               loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.105), 
               frameon=False, fontsize=32, handlelength=2.5, borderpad=0, columnspacing=1.5)

    plt.savefig('determinism.pdf', format='pdf', bbox_inches='tight') 
    print("\n=== Academic Plots Saved Successfully ===")

if __name__ == "__main__":
    main()