from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ========== Global style ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
    'axes.linewidth': 0.9,
    'axes.edgecolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ========== Workloads and data ==========
job1_labels = ['ResNet', 'ResNet', 'BERT', 'BERT']
job2_labels = ['Llama 3', 'GPT-J', 'Llama 3', 'GPT-J']
gpu_counts = ['H20×1', 'H20×1', 'H20×1', 'H20×1']

systems = ['Temporal', 'MIG', 'MPS', 'Orion', 'LithOS', 'PROTEUS']

throughput_data = {
    'Temporal': [0.44, 0.40, 0.42, 0.43],
    'MPS':      [0.50, 0.54, 0.55, 0.54],
    'MIG':      [0.50, 0.50, 0.53, 0.50],
    'Orion':    [0.50, 0.55, 0.53, 0.51],
    'LithOS':   [0.53, 0.53, 0.58, 0.54],
    'PROTEUS':  [0.55, 0.55, 0.60, 0.60],
}

latency_data = {
    'Temporal': [7.2, 8.9, 9.0, 9.6],
    'MPS':      [9.0, 10.2, 9.5, 10.6],
    'MIG':      [10.0, 12.0, 10.0, 12.0],
    'Orion':    [12.0, 15.0, 15.0, 19.0],
    'LithOS':   [5.3, 6.5, 7.6, 8.2],
    'PROTEUS':  [4.5, 5.9, 7.2, 8.0],
}

colors = ['#D9D9D9', '#F2C078', '#B9A7D0', '#8FB6D8', '#9BCB88', '#E64B3C']
hatches = ['', '\\\\', '//', 'xx', '..', '']

x = np.arange(len(job1_labels))
width = 0.125

fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.65))


def draw_bars(ax, data_dict, is_throughput):
    for i, system in enumerate(systems):
        pos = x + (i - 2.5) * width

        if system == 'LithOS':
            bars = ax.bar(
                pos,
                data_dict[system],
                width,
                label=system,
                facecolor='none',
                edgecolor=colors[i],
                linestyle='--',
                linewidth=1.25,
                hatch=hatches[i],
            )
        else:
            bars = ax.bar(
                pos,
                data_dict[system],
                width,
                label=system,
                color=colors[i],
                edgecolor='black',
                linestyle='-',
                linewidth=0.7,
                hatch=hatches[i],
            )

        for bar in bars:
            height = bar.get_height()
            label_text = f'{height:.2f}' if is_throughput else f'{height:.1f}'
            if system == 'LithOS':
                label_text += '†'

            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                rotation=90,
                fontweight='bold',
                fontsize=11.5,
                color='#222222',
            )


draw_bars(axes[0], throughput_data, is_throughput=True)
draw_bars(axes[1], latency_data, is_throughput=False)


def format_axis(ax, ylabel, title, ymax, headroom):
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=12.5, fontweight='bold', loc='left', pad=13)
    ax.set_xlim(-0.5, len(job1_labels) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylim(0, ymax * headroom)
    ax.tick_params(axis='y', labelsize=10.5, direction='in', length=3.5)
    ax.tick_params(axis='x', length=0)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.65, color='#B8B8B8', alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


format_axis(
    axes[0],
    'Normalized Throughput',
    '(a) DNN Training Job Performance (Higher is better)',
    0.6,
    1.30,
)
format_axis(
    axes[1],
    'p99 Latency (s)',
    '(b) LLM Inference Job Performance (Lower is better)',
    19.0,
    1.26,
)

workload_table = axes[1].table(
    cellText=[job1_labels, job2_labels, gpu_counts],
    rowLabels=['Job 1', 'Job 2', 'GPU'],
    cellLoc='center',
    rowLoc='right',
    bbox=[0.0, -0.51, 1.0, 0.42],
    edges='open',
)
workload_table.auto_set_font_size(False)
workload_table.set_fontsize(13.0)
for (row, column), cell in workload_table.get_celld().items():
    cell.set_edgecolor('none')
    cell.set_linewidth(0)
    cell.set_facecolor('none')
    cell.PAD = 0.02
    if column == -1:
        cell.set_text_props(fontweight='bold', fontsize=13.0, ha='right', color='#333333')

handles, labels = axes[0].get_legend_handles_labels()
for i, label in enumerate(labels):
    if label == 'LithOS':
        labels[i] = 'LithOS†'

fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.995),
    ncol=6,
    frameon=False,
    fontsize=13.0,
    handlelength=1.8,
    handleheight=1.0,
    handletextpad=0.45,
    columnspacing=1.15,
)

fig.text(
    0.505,
    0.872,
    '† LithOS carries nondeterminism risk; results are included for completeness.',
    ha='center',
    va='bottom',
    fontsize=13.0,
    color='#444444',
)

fig.subplots_adjust(left=0.125, right=0.992, bottom=0.185, top=0.790, hspace=0.31)
output_path = (
    Path(__file__).resolve().parent
    / 'colocated_training_with_llm_inference.pdf'
)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches='tight', format='pdf')
plt.close(fig)
