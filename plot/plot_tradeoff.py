import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_composite_system_figure(block_configs, triton_results):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"], 
        "font.size": 10,           
        "axes.labelsize": 11,      
        "xtick.labelsize": 10,     
        "ytick.labelsize": 10,
        "legend.fontsize": 8,      
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'hatch.linewidth': 0.8
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), gridspec_kw={'width_ratios': [1, 1.6]})
    
    color_blue = '#1f77b4' 
    color_red = '#d62728' 

    # ==========================================
    # --- 子图 A : HoL 阻塞延迟 ---
    # ==========================================
    ax_a = axes[0]
    metrics = ['TTFT\n(Prefill)', 'TPOT\n(Decode)']
    baseline = [90.01, 51.87]
    colocated = [6496.94, 1959.27]

    x_a = np.arange(len(metrics))
    width_a = 0.32 

    rects1 = ax_a.bar(x_a - width_a/2, baseline, width_a, label='Isolated', 
                      color=color_blue, edgecolor='black', linewidth=1.0, hatch='//', zorder=3)
    rects2 = ax_a.bar(x_a + width_a/2, colocated, width_a, label='Co-located', 
                      color=color_red, edgecolor='black', linewidth=1.0, hatch='\\\\', zorder=3)

    ax_a.set_yscale('log')
    ax_a.set_ylim(10, 100000) 

    ax_a.set_ylabel('P90 End-to-End Latency (ms)', fontweight='bold')
    ax_a.set_title('(a) Tail Latency Spike', fontweight='bold', pad=12)
    
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(metrics, fontweight='bold')
    ax_a.legend(loc='upper right', frameon=True, framealpha=1.0, edgecolor='black', fontsize=8)

    def autolabel_Composite(rects, ax, base_data=None):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            text = f'{height:.1f}'
            if base_data is not None:
                multiplier = height / base_data[i]
                text += f'\n({multiplier:.1f}x)'
            ax.annotate(text, xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    autolabel_Composite(rects1, ax_a)
    autolabel_Composite(rects2, ax_a, base_data=baseline)


    # ==========================================
    # --- 子图 B : 语义不确定性误差分析 ---
    # ==========================================
    ax_b = axes[1]
    fp16_mae = np.array(triton_results["FP16"]["mad"])
    fp16_diff = np.array(triton_results["FP16"]["max_ad"]) - fp16_mae
    bf16_mae = np.array(triton_results["BF16"]["mad"])
    bf16_diff = np.array(triton_results["BF16"]["max_ad"]) - bf16_mae

    x_b = np.arange(len(block_configs))
    width_b = 0.35  

    ax_b.bar(x_b - width_b/2, fp16_mae, width_b, color=color_blue, edgecolor='black', zorder=3, linewidth=1.0)
    ax_b.bar(x_b - width_b/2, fp16_diff, width_b, bottom=fp16_mae, color=color_blue, 
             edgecolor='black', hatch='////', zorder=3, alpha=0.85, linewidth=1.0)

    ax_b.bar(x_b + width_b/2, bf16_mae, width_b, color=color_red, edgecolor='black', zorder=3, linewidth=1.0)
    ax_b.bar(x_b + width_b/2, bf16_diff, width_b, bottom=bf16_mae, color=color_red, 
             edgecolor='black', hatch='////', zorder=3, alpha=0.85, linewidth=1.0)

    ax_b.set_ylabel('Absolute Numerical Deviation', fontweight='bold')
    ax_b.set_xlabel('Kernel Slicing Granularity (Grid Splits)', fontweight='bold')
    ax_b.set_title('(b) Numerical Deviations in Custom Op', fontweight='bold', pad=12)
    
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([str(bs) for bs in block_configs], fontweight='bold')
    
    fp16_patch = mpatches.Patch(facecolor=color_blue, edgecolor='black', label='FP16')
    bf16_patch = mpatches.Patch(facecolor=color_red, edgecolor='black', label='BF16')
    mae_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Mean Abs. Dev.') 
    max_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Max Abs. Dev.')
    
    ax_b.legend(handles=[fp16_patch, bf16_patch, mae_patch, max_patch], 
                loc='upper left', ncol=2, framealpha=0.9, edgecolor='black',
                handletextpad=0.4, columnspacing=0.8, fontsize=8)

    # 布局收尾
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25) 
    
    # 保存输出
    pdf_path = 'none-determinism.pdf'
    png_path = 'none-determinism.png'
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Plots successfully generated: {pdf_path} and {png_path}")

if __name__ == "__main__":
    with open("result1.txt", "r") as f:
        data = json.load(f)
    plot_composite_system_figure(data["configs"], data["results"])