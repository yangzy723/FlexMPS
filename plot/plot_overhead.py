import matplotlib.pyplot as plt
import numpy as np

# ========== 数据准备 ==========
categories = ['Switching Overhead', 'Preemption Overhead']
exclusive = [1.00, 1.00]
cogpu = [1.04, 1.12]
x = np.array([0, 0.6]) 
fig, ax = plt.subplots(figsize=(6, 3)) 
bar_width = 0.15
# 字体设置 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ========== 绘制条形图 ==========
rects1 = ax.bar(x - bar_width/2, exclusive, bar_width, 
                label='Exclusive GPU Baseline', 
                color='#e0e0e0',  
                edgecolor='black', 
                hatch='//',       
                linewidth=1.2)    

rects2 = ax.bar(x + bar_width/2, cogpu, bar_width, 
                label='CoGPU', 
                color='#555555',  
                edgecolor='black', 
                hatch='\\\\',     
                linewidth=1.2)

# ========== 添加数据标签 ==========
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', 
                    fontsize=11,
                    fontweight='bold') 

add_labels(rects1)
add_labels(rects2)

# ========== 坐标轴设置 ==========
ax.set_ylabel('Normalized Overhead', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 1.4)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

ax.set_xlim(x[0] - 0.35, x[1] + 0.35)

# ========== 图例设置 ==========
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.28), 
          ncol=2, fancybox=False, shadow=False, 
          fontsize=11, edgecolor='black', framealpha=1.0)

# ========== 边框与刻度清理 ==========
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(axis='both', which='major', width=1.2, labelsize=11)

# ========== 网格设置 ==========
ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax.set_axisbelow(True)

# ========== 布局调整与保存 ==========
plt.tight_layout()

plt.savefig('overhead.png', bbox_inches='tight', dpi=300)
plt.savefig('overhead.pdf', bbox_inches='tight')

plt.show()