import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 配置 ---
FILE_PATH = '../AzureLLMInferenceTrace_conv_1week.csv' # 替换为你的文件名

def analyze_trace(file_path):
    print(f"正在加载数据: {file_path} ... (文件较大，请耐心等待)")
    
    # 1. 读取数据
    # parse_dates 会消耗一些时间，但对于时间序列分析是必须的
    df = pd.read_csv(file_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed')
    
    # 按时间排序（防止数据乱序）
    df = df.sort_values('TIMESTAMP')
    
    print(f"数据加载完成，共 {len(df)} 条请求。\n")

    # --- 2. 基础统计 ---
    print("=== [1] 基础 Token 统计 ===")
    print(df[['ContextTokens', 'GeneratedTokens']].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    
    # 计算 Prefill 与 Decode 的比例
    total_input = df['ContextTokens'].sum()
    total_output = df['GeneratedTokens'].sum()
    ratio = total_input / total_output
    print(f"\n总 Input Tokens: {total_input}")
    print(f"总 Output Tokens: {total_output}")
    print(f"Prefill / Decode 比例: {ratio:.2f} : 1 (每生成1个token，需要处理 {ratio:.2f} 个历史token)")
    
    # --- 3. 流量负载分析 (RPS) ---
    print("\n=== [2] 流量负载特征 (RPS) ===")
    # 将数据按秒重采样，计算每秒的请求数
    # 将时间戳设为索引
    df_time = df.set_index('TIMESTAMP')
    
    # 'S' 代表按秒统计 .size() 计数
    rps_series = df_time.resample('1s').size()
    
    avg_rps = rps_series.mean()
    max_rps = rps_series.max()
    p95_rps = rps_series.quantile(0.95)
    
    print(f"平均 RPS: {avg_rps:.2f}")
    print(f"峰值 RPS: {max_rps}")
    print(f"95% 负载 RPS: {p95_rps:.2f}")

    # --- 4. 特殊特征检测 ---
    print("\n=== [3] 特殊特征检测 ===")
    # 检测极短生成 (通常是 Filter 拦截或简单的 ACK)
    short_gen_count = len(df[df['GeneratedTokens'] <= 5])
    print(f"极短生成 (<=5 tokens) 占比: {short_gen_count / len(df) * 100:.2f}% (可能是被安全拦截或无内容)")
    
    # 检测超长上下文 (显存杀手)
    long_ctx_count = len(df[df['ContextTokens'] > 8000]) # 假设 8k 是个坎
    print(f"长上下文 (>8000 tokens) 数量: {long_ctx_count}")

    # --- 5. 绘图 (可选) ---
    # 绘制输入输出长度分布图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['ContextTokens'], bins=50, color='skyblue', log=True)
    plt.title('Input Token Distribution (Log Scale)')
    plt.xlabel('Length')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.hist(df['GeneratedTokens'], bins=50, color='salmon', log=True)
    plt.title('Output Token Distribution (Log Scale)')
    plt.xlabel('Length')
    
    plt.tight_layout()
    plt.savefig('token_dist.png')
    print("\n图表已保存为 token_dist.png")

if __name__ == "__main__":
    analyze_trace(FILE_PATH)