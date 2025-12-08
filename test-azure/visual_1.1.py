import matplotlib.pyplot as plt
import numpy as np

# ================= DATA PREPARATION =================
# 1. Throughput Data
throughput_req_metrics = ['Requests/s']
data_req_baseline = [14.02]
data_req_pd_no_policy = [14.53]
data_req_pd_priority = [14.61]

# Tokens/s Data
throughput_tok_metrics = ['Prefill Tokens/s', 'Decode Tokens/s']
data_tok_baseline = [22841.80, 1615.39]
data_tok_pd_no_policy = [22983.47, 1637.01]
data_tok_pd_priority = [22999.28, 1642.05]

# 2. Latency Data
# TTFT (Seconds)
latency_ttft_labels = ['Avg', 'P50', 'P90', 'P99']
data_ttft_baseline = [0.5103, 0.1583, 2.0259, 3.6601]
data_ttft_pd_no_policy = [1.2911, 0.1452, 1.1639, 21.7737]
data_ttft_pd_priority = [0.1748, 0.1325, 0.3428, 0.6452]

# TPOT (Milliseconds)
latency_tpot_labels = ['Avg', 'P50', 'P90', 'P99']
data_tpot_baseline = [45.61, 18.60, 112.12, 409.46]
data_tpot_pd_no_policy = [54.07, 7.44, 108.41, 786.31]
data_tpot_pd_priority = [26.42, 7.19, 60.48, 238.18]

# ================= AESTHETICS CONFIG =================
# New Professional Palette:
# Baseline: Neutral Gray (Background context)
# No Policy: Strong Blue (Technical implementation)
# Priority: Vivid Red/Orange (Optimization highlight)
colors = ['#95A5A6', '#34495E', '#D35400'] 
labels = ['Baseline', 'PD (No Policy)', 'PD (Decode Priority)']

bar_width = 0.22 
opacity = 1.0 # Solid colors look better for thin bars
edge_color = 'white'
bar_edge_width = 0

# Global Font Settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def style_axis(ax, title, ylabel, max_val):
    """Apply a clean, modern style to the axis and set dynamic limits."""
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='#2c3e50')
    ax.set_ylabel(ylabel, fontsize=9, color='#555555')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Customize grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#bdc3c7', zorder=0)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#555555')
    
    # Dynamic Y-Limit: Add 25% headroom for labels
    ax.set_ylim(0, max_val * 1.25)

def add_labels(rects, ax, unit=""):
    """Attach a text label above each bar."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}{unit}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold', color='#34495e')

# ================= FIGURE GENERATION =================
fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

# --- Helper to plot 3 bars ---
def plot_three_bars(ax, x_indices, data1, data2, data3, x_labels, y_label, title, unit=""):
    r1 = ax.bar(x_indices - bar_width, data1, bar_width, label=labels[0], color=colors[0], alpha=opacity, edgecolor=edge_color, linewidth=bar_edge_width, zorder=3)
    r2 = ax.bar(x_indices, data2, bar_width, label=labels[1], color=colors[1], alpha=opacity, edgecolor=edge_color, linewidth=bar_edge_width, zorder=3)
    r3 = ax.bar(x_indices + bar_width, data3, bar_width, label=labels[2], color=colors[2], alpha=opacity, edgecolor=edge_color, linewidth=bar_edge_width, zorder=3)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)
    
    # Determine max value for scaling
    max_val = max(max(data1), max(data2), max(data3))
    style_axis(ax, title, y_label, max_val)
    
    add_labels(r1, ax, unit)
    add_labels(r2, ax, unit)
    add_labels(r3, ax, unit)
    
    return r1 # Return one rect group for legend handle extraction if needed

# --- Subplot 1: Requests/s ---
ax1 = axs[0, 0]
x_req = np.arange(len(throughput_req_metrics))
plot_three_bars(ax1, x_req, data_req_baseline, data_req_pd_no_policy, data_req_pd_priority, 
                throughput_req_metrics, 'Requests per Second', 'Throughput: Requests/s')
ax1.set_xlim(-0.8, 0.8) # Center single group

# --- Subplot 2: Tokens/s ---
ax2 = axs[0, 1]
x_tok = np.arange(len(throughput_tok_metrics))
plot_three_bars(ax2, x_tok, data_tok_baseline, data_tok_pd_no_policy, data_tok_pd_priority, 
                throughput_tok_metrics, 'Tokens per Second', 'Throughput: Tokens/s')
ax2.set_xlim(-0.6, 1.6) # Better spacing

# --- Subplot 3: TTFT ---
ax3 = axs[1, 0]
x_ttft = np.arange(len(latency_ttft_labels))
plot_three_bars(ax3, x_ttft, data_ttft_baseline, data_ttft_pd_no_policy, data_ttft_pd_priority, 
                latency_ttft_labels, 'Time (Seconds)', 'Time To First Token (TTFT)', "s")

# --- Subplot 4: TPOT ---
ax4 = axs[1, 1]
x_tpot = np.arange(len(latency_tpot_labels))
plot_three_bars(ax4, x_tpot, data_tpot_baseline, data_tpot_pd_no_policy, data_tpot_pd_priority, 
                latency_tpot_labels, 'Time (Milliseconds)', 'Time Per Output Token (TPOT)', "ms")

# --- GLOBAL LEGEND & LAYOUT ---
# Create a single legend for the entire figure at the top
handles, _ = ax4.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), 
          ncol=3, frameon=False, fontsize=10)

plt.suptitle('Benchmark Comparison: PD Disaggregation Performance', fontsize=16, fontweight='bold', color='#2c3e50', y=0.99)
plt.tight_layout()
# Adjust rect to make room for suptitle and legend
plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.2) 

plt.savefig('benchmark_comparison_final.png', dpi=300, bbox_inches='tight')
print("Generated benchmark_comparison_final.png")