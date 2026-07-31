import matplotlib.pyplot as plt
import numpy as np

# ========== 样式设置 ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['axes.linewidth'] = 1.0

# ========== 混部任务 (Job 1 & Job 2) 拆分数据 ==========
categories = ['A', 'B', 'C', 'D', '1', '2', 'E']

# Job 1 数据
job1_throughput = {
    'Temporal':  [0.44, 0.43, 0.43, 0.46, 0.80, 0.60, 0.53],
    'Tick-Tock': [0.50, 0.62, 0.50, 0.50, 0.88, 0.55, 0.85],
    'Orion':     [0.70, 0.62, 0.50, 0.51, 0.87, 0.55, 0.85],
    'GSlice':    [0.49, 0.51, 0.54, 0.54, 0.86, 0.68, 0.79],
    'Salus':     [0.51, 0.58, 0.54, 0.52, 0.88, 0.68, 0.78],
    'MPS':       [0.55, 0.54, 0.55, 0.54, 0.89, 0.76, 0.67],
    'MIG':       [0.53, 0.52, 0.53, 0.52, 0.91, 0.91, 0.93],
    'PROTEUS':     [0.55, 0.55, 0.58, 0.60, 0.99, 0.92, 0.95]
}

# Job 2 数据
job2_throughput = {
    'Temporal':  [0.44, 0.43, 0.46, 0.40, 0.80, 0.66, 0.53],
    'Tick-Tock': [0.51, 0.40, 0.49, 0.42, 0.89, 0.84, 0.87],
    'Orion':     [0.42, 0.47, 0.53, 0.45, 0.89, 0.94, 0.85],
    'GSlice':    [0.47, 0.49, 0.55, 0.49, 0.87, 0.69, 0.75],
    'Salus':     [0.49, 0.49, 0.56, 0.50, 0.86, 0.68, 0.75],
    'MPS':       [0.55, 0.54, 0.51, 0.51, 0.89, 0.68, 0.67],
    'MIG':       [0.53, 0.52, 0.53, 0.50, 0.91, 0.52, 0.93],
    'PROTEUS':     [0.55, 0.56, 0.58, 0.54, 0.99, 0.88, 0.95]
}

plot_systems = ['Temporal', 'MIG', 'MPS', 'Orion', 'Salus', 'PROTEUS']
colors = ['#d9d9d9', '#f5c687', '#c4b5db', '#9dc3e6', '#a9d18e', '#f44336']
hatches = ['', '\\\\\\', '///', 'xxx', '...', '|||']

x = np.arange(len(categories))
width = 0.13

fig, ax = plt.subplots(figsize=(14, 4.5))

# ========== 绘制逻辑 ==========
for i, sys in enumerate(plot_systems):
    pos = x + (i - 2.5) * width 
    
    j1_data = np.array(job1_throughput[sys])
    j2_data = np.array(job2_throughput[sys])
    
    bars_job1 = ax.bar(pos, j1_data, width, 
                       color=colors[i], edgecolor='black', 
                       linestyle='-', linewidth=0.75, hatch='')
    
    bars_job2 = ax.bar(pos, j2_data, width, bottom=j1_data, label=sys,
                       color=colors[i], edgecolor='black', 
                       linestyle='-', linewidth=0.75, hatch=hatches[i])
    
    # 顶端数值标注 (白底遮罩，防止与虚线重合)
    for j, bar in enumerate(bars_job2):
        total_height = j1_data[j] + j2_data[j]
        ax.annotate(f'{total_height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, total_height),
                     xytext=(0, 4), 
                     textcoords="offset points",
                     ha='center', va='bottom', rotation=90,
                     fontweight='bold', fontsize=10, color='#222222',
                     bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))

# ========== 坐标轴、背景与 Upper Bound ==========
ax.set_ylabel('Total Normalized Throughput', fontsize=14, fontweight='bold')
ax.set_xlabel('Configuration Group', fontsize=14, fontweight='bold', labelpad=10)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14, fontweight='normal')

ax.set_ylim(0, 2.5)
ax.tick_params(axis='y', labelsize=12)

# 基础网格线
ax.yaxis.grid(True, linestyle='--', color='#cccccc', alpha=0.6, zorder=0)
ax.set_axisbelow(True) 

# 绘制 Upper Bound 虚线
ax.axhline(y=2.0, color='#666666', linestyle='--', linewidth=1.5, zorder=0)
ax.text(0.005, 2.03, 'Ideal Upper Bound (2.0)', 
        transform=ax.get_yaxis_transform(),
        color='#444444', fontsize=13, fontweight='bold', 
        ha='left', va='bottom', zorder=5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ========== 图例设置 ==========
ax.legend(loc='upper center', 
          bbox_to_anchor=(0.5, 1.15), 
          ncol=6, 
          frameon=False, 
          fontsize=13, 
          handlelength=2.0, 
          handleheight=1.2,
          columnspacing=2.5)

plt.tight_layout()
plt.savefig('colocated_trainings.pdf', bbox_inches='tight', format='pdf')
plt.show()