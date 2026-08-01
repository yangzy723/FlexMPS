import matplotlib.pyplot as plt
import numpy as np

# ========== 全局样式 ==========
# 统一使用 Serif 字体，匹配 ACM/IEEE 论文排版
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['axes.linewidth'] = 1.0

# ========== 数据定义 ==========
categories = ['1', '2', '3', '4']

# 严格对齐顺序 (Temporal, MIG, MPS, Orion, LithOS, PROTEUS)
systems = ['Temporal', 'MIG', 'MPS', 'Orion', 'LithOS', 'PROTEUS']

throughput_data = {
    'Temporal': [0.44, 0.40, 0.42, 0.43],
    'MPS':      [0.50, 0.54, 0.55, 0.54],
    'MIG':      [0.50, 0.50, 0.53, 0.50],
    'Orion':    [0.50, 0.55, 0.53, 0.51],
    'LithOS':   [0.53, 0.53, 0.58, 0.54],
    'PROTEUS':    [0.55, 0.55, 0.60, 0.60]
}

latency_data = {
    'Temporal': [7.2, 8.9, 9.0, 9.6],
    'MPS':      [9.0, 10.2, 9.5, 10.6],
    'MIG':      [10.0, 12.0, 10.0, 12.0],
    'Orion':    [12.0, 15.0, 15.0, 19.0],
    'LithOS':   [5.3, 6.5, 7.6, 8.2],
    'PROTEUS':    [4.5, 5.9, 7.2, 8.0]
}

# 严格对齐颜色与纹理
colors = ['#d9d9d9', '#f5c687', '#c4b5db', '#9dc3e6', '#a9d18e', '#f44336']
hatches = ['', '\\\\', '//', 'xx', '..', '']

x = np.arange(len(categories))
width = 0.12

# 调整画布大小，使其更适合双栏横跨布局
fig, axes = plt.subplots(2, 1, figsize=(10, 6.9))

# ========== 核心绘制逻辑 ==========
def draw_bars(ax, data_dict, is_throughput):
    for i, sys in enumerate(systems):
        pos = x + (i - 2.5) * width 
        
        if sys == 'LithOS':
            # LithOS：幽灵虚线框，强调没有 Determinism 保证
            bars = ax.bar(pos, data_dict[sys], width, label=sys,
                          facecolor='none',          
                          edgecolor=colors[i],       
                          linestyle='--',            
                          linewidth=1.5,             
                          hatch=hatches[i])          
        else:
            # 常规系统与 PROTEUS
            bars = ax.bar(pos, data_dict[sys], width, label=sys,
                          color=colors[i], 
                          edgecolor='black', 
                          linestyle='-', 
                          linewidth=0.75, 
                          hatch=hatches[i])
        
        # 数值标签添加逻辑
        for bar in bars:
            height = bar.get_height()
            label_text = f'{height:.2f}' if is_throughput else f'{height:.1f}'
            
            if sys == 'LithOS':
                label_text += '*'
                
            ax.annotate(label_text,
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), 
                         textcoords="offset points",
                         ha='center', va='bottom', rotation=90,
                         fontweight='bold', fontsize=13, color='#222222')

# 执行绘制
draw_bars(axes[0], throughput_data, is_throughput=True)
draw_bars(axes[1], latency_data, is_throughput=False)

# ========== 坐标轴与背景 ==========
def format_axis(ax, ylabel, title, ymax):
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', loc='left', pad=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14, fontweight='normal')
    
    ax.set_ylim(0, ymax * 1.30) 
    ax.tick_params(axis='y', labelsize=13)
    
    ax.yaxis.grid(True, linestyle='--', color='#cccccc', alpha=0.7, zorder=0)
    ax.set_axisbelow(True) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

format_axis(axes[0], 'Normalized Throughput', '(a) DNN Training Job Performance (Higher is better)', 0.6)
format_axis(axes[1], 'p99 Latency (s)', '(b) LLM Inference Job Performance (Lower is better)', 19.0)
axes[0].tick_params(axis='x', labelbottom=False)
axes[1].set_xlabel('Configuration Group', fontsize=16, fontweight='bold', labelpad=8)

# ========== 图例 ==========
handles, labels = axes[0].get_legend_handles_labels()

for i in range(len(labels)):
    if labels[i] == 'LithOS':
        labels[i] = 'LithOS (No Determinism Guarantee)'

fig.legend(handles, labels, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.985), 
           ncol=3, 
           frameon=False, 
           fontsize=14.5, 
           handlelength=2.5, 
           handleheight=1.2,
           columnspacing=2.0)

# 调整子图布局：顶部刻度仅在底部 panel 显示，避免与 (b) 标题冲突
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.11, top=0.80, hspace=0.36) 

# 导出为无损 PDF
plt.savefig('colocated_training_with_LLM_inference.pdf', bbox_inches='tight', format='pdf')
plt.show()