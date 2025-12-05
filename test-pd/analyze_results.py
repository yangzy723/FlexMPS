import json
import statistics
import sys
import os

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def print_section(title, p_data, d_data, total_wall_time):
    print(f"\n【{title} 模式数据】")
    print(f"  总墙钟耗时: {total_wall_time:.4f}s")
    
    p_avg = statistics.mean(p_data['latencies']) if p_data else 0
    d_avg = statistics.mean(d_data['latencies']) if d_data else 0
    
    print(f"  Prefill Avg Latency: {p_avg:.4f}s")
    print(f"  Decode  Avg Latency: {d_avg:.4f}s")
    return p_avg, d_avg

def main():
    print("="*50)
    print("MPS / Interference Benchmark Final Report")
    print("="*50)

    # 1. 加载串行数据
    s_prefill = load_json("res_serial_prefill.json")
    s_decode = load_json("res_serial_decode.json")
    
    # 串行总时间 = P的时间 + D的时间 (因为是顺序跑的)
    s_total_time = s_prefill['wall_time'] + s_decode['wall_time']
    s_p_avg, s_d_avg = print_section("Serial (无干扰)", s_prefill, s_decode, s_total_time)

    # 2. 加载并行数据
    p_prefill = load_json("res_parallel_prefill.json")
    p_decode = load_json("res_parallel_decode.json")
    
    # 并行总时间 = 两个文件中记录的较长的那个 (或者是外部传入的实际重叠时间，
    # 但这里取两者最大值近似表示，因为它们是同时启动的)
    p_total_time = max(p_prefill['wall_time'], p_decode['wall_time'])
    p_p_avg, p_d_avg = print_section("Parallel (有干扰)", p_prefill, p_decode, p_total_time)

    # 3. 对比分析
    print("\n【关键指标分析】")
    
    # 加速比
    speedup = s_total_time / p_total_time if p_total_time > 0 else 0
    print(f"1. 吞吐量收益 (Throughput Speedup):")
    print(f"   >>> {speedup:.2f}x (数值 > 1.0 说明 MPS/并行 带来了总吞吐提升)")
    
    # 延迟退化
    p_deg = (p_p_avg - s_p_avg) / s_p_avg * 100
    d_deg = (p_d_avg - s_d_avg) / s_d_avg * 100
    
    print(f"\n2. 延迟干扰/退化 (Latency Degradation):")
    print(f"   Prefill 变慢: {p_deg:+.1f}%")
    print(f"   Decode  变慢: {d_deg:+.1f}%")
    
    if speedup > 1.1 and abs(d_deg) < 20:
        print("\n>>> 结论: MPS 效果显著，在牺牲少量延迟的情况下显著提升了吞吐。")
    elif speedup < 1.0:
        print("\n>>> 结论: 负优化。并行执行比串行还慢，可能是显存带宽瓶颈或 MPS 未开启。")
    else:
        print("\n>>> 结论: 效果一般。")

if __name__ == "__main__":
    main()