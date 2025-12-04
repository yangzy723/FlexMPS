import asyncio
import aiohttp
import time
import statistics
from dataclasses import dataclass

# ================= 配置区域 =================
DECODE_URL = "http://localhost:30001/generate"
PREFILL_URL = "http://localhost:30002/generate"

# 任务量定义 (控制变量：无论串行还是并行，总数都一样)
NUM_PREFILL_REQUESTS = 128  # P 任务总数
NUM_DECODE_REQUESTS = 256  # D 任务总数

# 负载定义
PREFILL_PROMPT_LEN = 32000
PREFILL_OUTPUT_LEN = 1
DECODE_PROMPT_LEN = 10
DECODE_OUTPUT_LEN = 256
# ===========================================

@dataclass
class Result:
    latency: float
    type: str # 'P' or 'D'

def generate_prompt(length):
    return "Hello " * length

async def send_request(session, url, prompt, out_len, tag, r_type):
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": out_len,
            "ignore_eos": True
        }
    }
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            await response.read()
            success = True
    except Exception as e:
        print(f"[{tag}] Failed: {e}")
        success = False
    
    cost = time.perf_counter() - start
    return Result(cost, r_type) if success else None

async def run_batch(session, mode_name, p_count, d_count):
    """
    执行一批任务，返回 (总耗时, 结果列表)
    """
    tasks = []
    p_prompt = generate_prompt(PREFILL_PROMPT_LEN)
    d_prompt = generate_prompt(DECODE_PROMPT_LEN)
    
    print(f"\n--- [{mode_name}] 准备发射: {p_count} Prefill + {d_count} Decode ---")
    
    # 构造 Prefill 任务
    for i in range(p_count):
        tasks.append(send_request(session, PREFILL_URL, p_prompt, PREFILL_OUTPUT_LEN, f"P-{i}", 'P'))
        
    # 构造 Decode 任务
    for i in range(d_count):
        tasks.append(send_request(session, DECODE_URL, d_prompt, DECODE_OUTPUT_LEN, f"D-{i}", 'D'))
        
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    
    valid_results = [r for r in results if r is not None]
    total_time = end_time - start_time
    
    return total_time, valid_results

def print_stats(title, results, total_time):
    p_lats = [r.latency for r in results if r.type == 'P']
    d_lats = [r.latency for r in results if r.type == 'D']
    
    print(f"  >>> {title} 总墙钟耗时: {total_time:.4f}s")
    
    if p_lats:
        print(f"      Prefill Avg: {statistics.mean(p_lats):.4f}s")
    if d_lats:
        print(f"      Decode  Avg: {statistics.mean(d_lats):.4f}s")
    return p_lats, d_lats

async def main():
    async with aiohttp.ClientSession() as session:
        # ==========================================
        # 1. 串行测试 (Serial: PP -> DD)
        # ==========================================
        print("=== 阶段 1: 串行基准测试 (先跑完P，再跑完D) ===")
        
        # 1.1 先跑 Prefill
        t1, r1 = await run_batch(session, "Serial-Prefill", NUM_PREFILL_REQUESTS, 0)
        # 1.2 休息一下，让 GPU 喘口气
        await asyncio.sleep(2)
        # 1.3 再跑 Decode
        t2, r2 = await run_batch(session, "Serial-Decode", 0, NUM_DECODE_REQUESTS)
        
        serial_total_time = t1 + t2
        serial_results = r1 + r2
        
        # ==========================================
        # 2. 并行测试 (Parallel: PD 同时)
        # ==========================================
        print("\n\n=== 阶段 2: 并行干扰测试 (P+D 同时跑) ===")
        await asyncio.sleep(2) # 确保之前任务彻底结束
        
        parallel_total_time, parallel_results = await run_batch(
            session, "Parallel-Mixed", NUM_PREFILL_REQUESTS, NUM_DECODE_REQUESTS
        )

        # ==========================================
        # 3. 最终对比报告
        # ==========================================
        print("\n" + "="*50)
        print("FINAL REPORT: MPS / Interference Analysis")
        print("="*50)
        
        print(f"[配置] Prefill任务数: {NUM_PREFILL_REQUESTS}, Decode任务数: {NUM_DECODE_REQUESTS}")
        
        print("\n【模式 A: 串行执行 (No Interference)】")
        # 重新计算单纯的平均值
        p_base_lats, d_base_lats = print_stats("Serial", serial_results, serial_total_time)
        p_base_avg = statistics.mean(p_base_lats)
        d_base_avg = statistics.mean(d_base_lats)

        print("\n【模式 B: 并行执行 (With Interference)】")
        p_mix_lats, d_mix_lats = print_stats("Parallel", parallel_results, parallel_total_time)
        p_mix_avg = statistics.mean(p_mix_lats)
        d_mix_avg = statistics.mean(d_mix_lats)
        
        print("\n【关键指标分析】")
        
        # 1. 吞吐量收益 (Throughput Gain)
        # 如果并行总时间 < 串行总时间，说明我们利用空闲资源赚到了时间
        time_saved = serial_total_time - parallel_total_time
        speedup = (serial_total_time / parallel_total_time) if parallel_total_time > 0 else 0
        print(f"1. 总耗时对比 (越小越好):")
        print(f"   串行: {serial_total_time:.2f}s  vs  并行: {parallel_total_time:.2f}s")
        print(f"   >>> 加速比: {speedup:.2f}x ( >1.0 表示并行更高效)")

        # 2. 延迟退化 (Latency Degradation)
        # 看看因为干扰，单个请求变慢了多少
        p_slowdown = (p_mix_avg - p_base_avg) / p_base_avg * 100
        d_slowdown = (d_mix_avg - d_base_avg) / d_base_avg * 100
        
        print(f"\n2. 延迟干扰情况 (越小越好):")
        print(f"   Prefill 变慢了: {p_slowdown:+.1f}%  (Baseline: {p_base_avg:.3f}s -> Mix: {p_mix_avg:.3f}s)")
        print(f"   Decode  变慢了: {d_slowdown:+.1f}%  (Baseline: {d_base_avg:.3f}s -> Mix: {d_mix_avg:.3f}s)")

if __name__ == "__main__":
    asyncio.run(main())