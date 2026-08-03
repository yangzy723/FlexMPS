from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ========== 样式设置 ==========
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

# ========== 混部任务 (Job 1 & Job 2) 拆分数据 ==========
# Three unboxed rows identify both jobs and the shared GPU count.
job1_labels = [
    'Qwen3.5-0.8B',
    'Qwen3-0.6B',
    'Qwen3-1.7B',
    'Qwen3.5-0.8B',
    'Qwen3-8B LoRA',
    'Qwen3-32B\nQLoRA †',
    'Qwen3-32B',
    'Qwen3.5-122B-A10B\nQLoRA †',
]

job2_labels = [
    'Qwen3.5-0.8B',
    'Qwen3-0.6B',
    'Qwen3-1.7B',
    'Qwen3-0.6B',
    'Qwen3-8B LoRA',
    'Qwen3-32B\nQLoRA †',
    'Qwen3-32B',
    'Qwen3.5-122B-A10B\nQLoRA †',
]

gpu_counts = ['H20×1', 'H20×1', 'H20×1', 'H20×1', 'H20×1', 'H20×1', 'H20×8', 'H20×8']

# Job 1 数据
job1_throughput = {
    'Temporal': [0.40, 0.45, 0.44, 0.40, 0.53, 0.21, 0.34, 0.12],
    'MIG':      [0.49, 0.51, 0.51, 0.49, 0.92, 0.27, 0.24, 0.09],
    'MPS':      [0.53, 0.55, 0.58, 0.47, 0.63, 0.28, 0.37, 0.11],
    'Orion':    [0.47, 0.45, 0.46, 0.46, 0.83, 0.23, 0.40, 0.15],
    'Salus':    [0.45, 0.49, 0.48, 0.45, 0.76, 0.25, 0.29, 0.19],
    'PROTEUS':  [0.54, 0.57, 0.53, 0.54, 0.90, 0.52, 0.56, 0.34],
}

# Job 2 数据
job2_throughput = {
    'Temporal': [0.39, 0.45, 0.44, 0.46, 0.54, 0.22, 0.33, 0.14],
    'MIG':      [0.49, 0.51, 0.51, 0.51, 0.92, 0.27, 0.24, 0.09],
    'MPS':      [0.48, 0.49, 0.46, 0.48, 0.71, 0.29, 0.35, 0.13],
    'Orion':    [0.49, 0.58, 0.55, 0.51, 0.79, 0.15, 0.43, 0.17],
    'Salus':    [0.45, 0.53, 0.53, 0.51, 0.77, 0.18, 0.47, 0.17],
    'PROTEUS':  [0.56, 0.58, 0.59, 0.57, 0.95, 0.53, 0.53, 0.29],
}

plot_systems = ['Temporal', 'MIG', 'MPS', 'Orion', 'Salus', 'PROTEUS']
colors = ['#D9D9D9', '#F2C078', '#B9A7D0', '#8FB6D8', '#9BCB88', '#E64B3C']
hatches = ['---', r'\\', '///', 'xxx', '...', '|||']

x = np.arange(len(job1_labels))
width = 0.13

fig, ax = plt.subplots(figsize=(14.6, 4.9))

# ========== 绘制逻辑 ==========
for i, sys in enumerate(plot_systems):
    pos = x + (i - 2.5) * width

    j1_data = np.array(job1_throughput[sys])
    j2_data = np.array(job2_throughput[sys])

    bars_job1 = ax.bar(pos, j1_data, width,
                       color=colors[i], edgecolor='black',
                       linestyle='-', linewidth=0.7, hatch='')

    bars_job2 = ax.bar(pos, j2_data, width, bottom=j1_data, label=sys,
                       color=colors[i], edgecolor='black',
                       linestyle='-', linewidth=0.7, hatch=hatches[i])

    # 顶端数值标注 (白底遮罩，防止与虚线重合)
    for j, bar in enumerate(bars_job2):
        total_height = j1_data[j] + j2_data[j]
        ax.annotate(f'{total_height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, total_height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', rotation=90,
                     fontweight='bold', fontsize=11.5, color='#222222',
                     bbox=dict(boxstyle='square,pad=0.06', facecolor='white', edgecolor='none', alpha=0.78))

# ========== 坐标轴、背景与 Upper Bound ==========
ax.set_ylabel('Normalized\nThroughput', fontsize=14, fontweight='bold', labelpad=5)
ax.set_xlim(-0.5, len(job1_labels) - 0.5)
ax.set_xticks([])

ax.set_ylim(0, 2.22)
ax.set_yticks([])

# Nature-style workload rows replace letter-only configuration IDs.
workload_table = ax.table(
    cellText=[job1_labels, job2_labels, gpu_counts],
    rowLabels=['Job 1', 'Job 2', 'GPU'],
    cellLoc='center',
    rowLoc='right',
    bbox=[0.0, -0.72, 1.0, 0.63],
    edges='open',
)
workload_table.auto_set_font_size(False)
workload_table.set_fontsize(12.0)
for (row, column), cell in workload_table.get_celld().items():
    cell.set_edgecolor('none')
    cell.set_linewidth(0)
    cell.set_facecolor('none')
    cell.PAD = 0.02
    if column == -1:
        cell.set_text_props(fontweight='bold', fontsize=13.0, ha='right', color='#333333')

# 基础网格线
ax.yaxis.grid(True, linestyle='--', linewidth=0.65, color='#B8B8B8', alpha=0.55, zorder=0)
ax.set_axisbelow(True)

# 绘制 Upper Bound 虚线
ax.axhline(y=2.0, color='#666666', linestyle='--', linewidth=1.5, zorder=0)
ax.annotate(
    '2',
    xy=(0.006, 2.0),
    xycoords=('axes fraction', 'data'),
    xytext=(0, 3),
    textcoords='offset points',
    ha='left',
    va='bottom',
    fontsize=12,
    color='#333333',
)
ax.text(0.995, 2.03, 'Ideal Upper Bound (2.0)',
        transform=ax.get_yaxis_transform(),
        color='#444444', fontsize=12, fontweight='bold',
        ha='right', va='bottom', zorder=5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ========== Figure-level legend and stack encoding ==========
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.40, 0.965),
    ncol=6,
    frameon=False,
    fontsize=14.5,
    handlelength=1.75,
    handleheight=1.1,
    handletextpad=0.5,
    columnspacing=1.55,
)
fig.text(
    0.995,
    0.905,
    '† QLoRA configurations use CPU offload.',
    ha='right',
    va='center',
    fontsize=13.5,
    color='#444444',
)

fig.subplots_adjust(left=0.085, right=0.985, bottom=0.542, top=0.855)
output_path = Path(__file__).resolve().parent / 'colocated_trainings.pdf'
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02, format='pdf')
plt.close(fig)
