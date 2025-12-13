import asyncio
import aiohttp
import pandas as pd
import time
import secrets
import random
import json
import numpy as np
import sys

# ================= CONFIGURATION AREA =================

# Address of the two MPS instances
URL_PREFILL = "http://127.0.0.1:30001/v1/completions" # Prefill Node
URL_DECODE  = "http://127.0.0.1:30002/v1/completions" # Decode Node

DATASET_PATH = "BurstGPT_without_fails_2.csv"
MODEL_NAME = "/data/datasets/models-hf/Llama-3.1-8B-Instruct/" 
READ_LIMIT = 200000   

# --- CONTROL PARAMETERS ---
SAMPLE_INTERVAL = 1     # Sampling: Pick 1 request every N rows
SPEEDUP_FACTOR = 50    # Time Compression: >1.0 speeds up, <1.0 slows down

MAX_REQUESTS = 6000     # Total trace rows to process

# --- SLO CONFIGURATION  ---
SLO_TTFT = 1    # Seconds (Time To First Token 阈值, 仅适用于 Prefill Worker, Request级)
SLO_TPOT = 0.1  # Seconds (Time Per Output Token 阈值, 仅适用于 Decode Worker, Token级)

# ===================================================

VOCAB = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "system", "model", "inference", "performance", "latency", "throughput", "gpu",
    "compute", "memory", "cache", "token", "context", "decode", "prefill", "batch",
    "queue", "request", "server", "client", "python", "async", "await", "test"
]

def get_random_prompt_words(token_count):
    prefix = f"REQ_{secrets.token_hex(3)}: "
    # 处理可能的 NaN
    if pd.isna(token_count):
        num_words = 10
    else:
        num_words = int(token_count)
        
    if num_words <= 0: return prefix
    selected_words = random.choices(VOCAB, k=num_words)
    return prefix + " ".join(selected_words)

async def send_request(session, row, start_timestamp_ref, results_list, role):
    # role: "prefill_worker" or "decode_worker"
    
    target_time_offset = row['relative_timestamp'] / SPEEDUP_FACTOR

    req_tokens_val = row['Request tokens'] if not pd.isna(row['Request tokens']) else 10
    res_tokens_val = row['Response tokens'] if not pd.isna(row['Response tokens']) else 10

    # === Differentiate Input/Output ===
    if role == "prefill_worker":
        target_url = URL_PREFILL
        # Prefill Worker: Compute Bound
        prompt_text = get_random_prompt_words(req_tokens_val)
        req_max_tokens = 1
        recorded_input_len = int(req_tokens_val)
    else:
        target_url = URL_DECODE
        # Decode Worker: Memory Bound
        prompt_text = get_random_prompt_words(5)
        req_max_tokens = int(res_tokens_val)
        recorded_input_len = 5

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "max_tokens": req_max_tokens,
        "temperature": 0, 
        "ignore_eos": True,
        "stream": True 
    }

    # Wait for launch
    now_offset = time.time() - start_timestamp_ref
    wait_seconds = target_time_offset - now_offset
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    req_start = time.time()
    ttft = 0.0
    first_token_time = 0.0
    last_token_time = 0.0
    token_count = 0
    token_intervals = []

    try:
        async with session.post(target_url, json=payload) as response:
            if response.status != 200:
                await response.read()
                return

            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or line == 'data: [DONE]': continue
                
                if line.startswith('data: '):
                    try:
                        chunk_json = json.loads(line[6:]) 
                        choices = chunk_json.get("choices", [])
                        if not choices: continue
                        delta_content = choices[0].get("text", "")
                        
                        if delta_content:
                            current_time = time.time()
                            token_count += 1
                            
                            if first_token_time == 0.0:
                                first_token_time = current_time
                                last_token_time = current_time
                                ttft = first_token_time - req_start
                            else:
                                interval = current_time - last_token_time
                                token_intervals.append(interval)
                                last_token_time = current_time
                    except:
                        continue

            req_end = time.time()
            
            # --- SLO Check Logic ---
            ttft_violated = False
            bad_token_intervals_count = 0
            
            if role == "prefill_worker":
                # Prefill 关注 TTFT (Request 粒度)
                if ttft > SLO_TTFT:
                    ttft_violated = True
                    
            elif role == "decode_worker":
                # Decode 关注 TPOT (Token 粒度)
                bad_token_intervals_count = sum(1 for t in token_intervals if t > SLO_TPOT)

            results_list.append({
                "role": role,
                "input_len": recorded_input_len,
                "output_len": token_count, 
                "ttft": ttft,
                "token_intervals": token_intervals,
                "status": response.status,
                "ttft_violated": ttft_violated,  # For Prefill
                "bad_token_cnt": bad_token_intervals_count # For Decode
            })

    except Exception as e:
        pass

async def main():
    print(f"Loading trace: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH, nrows=READ_LIMIT)
    
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df = df.sort_values('Timestamp')
    
    if SAMPLE_INTERVAL > 1:
        print(f"Applying sampling: picking 1 request every {SAMPLE_INTERVAL} rows.")
        df = df.iloc[::SAMPLE_INTERVAL].copy()
    
    if MAX_REQUESTS and MAX_REQUESTS < len(df): 
        print(f"Limiting execution to first {MAX_REQUESTS} requests (after sampling).")
        df = df.head(MAX_REQUESTS)
    
    # [修改] 计算相对时间戳 (直接数值相减，不再使用 .dt.total_seconds)
    start_time_base = df['Timestamp'].iloc[0]
    df['relative_timestamp'] = df['Timestamp'] - start_time_base

    total_trace_rows = len(df)
    total_http_requests = total_trace_rows * 2 

    print(f"\n=== PD Separation Simulation (MPS) ===")
    print(f"Prefill Node : {URL_PREFILL}   (High Input, Output=1)")
    print(f"Decode Node  : {URL_DECODE}   (Low Input, Output=Trace)")
    print(f"Speedup      : {SPEEDUP_FACTOR}x")
    print(f"Trace Rows   : {total_trace_rows}")
    print(f"Total Requests: {total_http_requests}")
    print(f"SLO Thresholds: TTFT < {SLO_TTFT}s (Request level), TPOT < {SLO_TPOT}s (Token level)")
    
    results = []
    connector = aiohttp.TCPConnector(limit=4000) 
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        
        benchmark_start_time = time.time()
        print("Starting requests...")
        
        for _, row in df.iterrows():
            tasks.append(asyncio.create_task(
                send_request(session, row, benchmark_start_time, results, "prefill_worker")
            ))
            tasks.append(asyncio.create_task(
                send_request(session, row, benchmark_start_time, results, "decode_worker")
            ))
        
        completed_count = 0
        for future in asyncio.as_completed(tasks):
            await future
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_http_requests:
                percent = (completed_count / total_http_requests) * 100
                sys.stdout.write(f"\r[Progress]: {completed_count}/{total_http_requests} ({percent:.1f}%) requests finished.")
                sys.stdout.flush()
        
        benchmark_end_time = time.time()
        benchmark_duration = benchmark_end_time - benchmark_start_time
        print(f"\nAll tasks completed in {benchmark_duration:.2f} seconds.")

    # --- Statistics ---
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No results.")
        return

    print("\n" + "="*50)
    print("      PD SEPARATION SIMULATION RESULTS")
    print("="*50)
    
    # System-wide Throughput
    total_system_input = res_df['input_len'].sum()
    total_system_output = res_df['output_len'].sum()
    total_successful_reqs = len(res_df[res_df['status'] == 200])

    sys_rps = total_successful_reqs / benchmark_duration
    sys_prefill_tps = total_system_input / benchmark_duration
    sys_decode_tps = total_system_output / benchmark_duration

    print(f"Benchmark Duration     : {benchmark_duration:.2f} s")
    print(f"Total Input Tokens     : {total_system_input}")
    print(f"Total Output Tokens    : {total_system_output}")
    print(f"\n[System Throughput]")
    print(f"  Total Requests/s     : {sys_rps:.2f} req/s")
    print(f"  Total Prefill Tok/s  : {sys_prefill_tps:.2f} tokens/s")
    print(f"  Total Decode Tok/s   : {sys_decode_tps:.2f} tokens/s")

    # 1. Prefill Worker (Check TTFT SLO - Request Level)
    prefill_df = res_df[res_df['role'] == 'prefill_worker']
    if not prefill_df.empty:
        ttft_avg = prefill_df['ttft'].mean()
        ttft_p50 = prefill_df['ttft'].quantile(0.50)
        ttft_p90 = prefill_df['ttft'].quantile(0.90)
        ttft_p99 = prefill_df['ttft'].quantile(0.99)

        # SLO Calc (Request Based)
        ttft_violation_count = prefill_df['ttft_violated'].sum()
        ttft_violation_rate = (ttft_violation_count / len(prefill_df)) * 100

        print(f"\n[Prefill Worker Stats] (Compute Bound -> TTFT)")
        print(f"  Requests: {len(prefill_df)}")
        print(f"  TTFT Avg: {ttft_avg:.4f} s")
        print(f"  TTFT P50: {ttft_p50:.4f} s")
        print(f"  TTFT P90: {ttft_p90:.4f} s")
        print(f"  TTFT P99: {ttft_p99:.4f} s")
        print(f"  SLO Violation Rate (> {SLO_TTFT}s): {ttft_violation_rate:.2f}% ({ttft_violation_count}/{len(prefill_df)} Requests)")
    
    # 2. Decode Worker (Check TPOT SLO - Token Level)
    decode_df = res_df[res_df['role'] == 'decode_worker']
    if not decode_df.empty:
        all_intervals = []
        for intervals in decode_df['token_intervals']:
            all_intervals.extend(intervals)
        
        print(f"\n[Decode Worker Stats] (Memory Bound -> TPOT)")
        print(f"  Requests: {len(decode_df)}")
        
        if all_intervals:
            all_intervals_array = np.array(all_intervals)
            all_intervals_ms = all_intervals_array * 1000 
            
            tpot_avg = np.mean(all_intervals_ms)
            tpot_p50 = np.percentile(all_intervals_ms, 50)
            tpot_p90 = np.percentile(all_intervals_ms, 90)
            tpot_p99 = np.percentile(all_intervals_ms, 99)
            
            # --- SLO Calc ---
            total_intervals_count = len(all_intervals_array)
            total_bad_intervals = np.sum(all_intervals_array > SLO_TPOT)
            tpot_violation_rate = (total_bad_intervals / total_intervals_count) * 100

            print(f"  Global TPOT Avg: {tpot_avg:.2f} ms")
            print(f"  Global TPOT P50: {tpot_p50:.2f} ms")
            print(f"  Global TPOT P90: {tpot_p90:.2f} ms")
            print(f"  Global TPOT P99: {tpot_p99:.2f} ms")
            print(f"  Total Token Intervals: {total_intervals_count}")
            print(f"  SLO Violation Rate (> {SLO_TPOT}s): {tpot_violation_rate:.2f}% ({total_bad_intervals}/{total_intervals_count} Intervals)")
        else:
            print("  No tokens generated.")

    filename = f"result_pd_speed{SPEEDUP_FACTOR}x.csv"
    res_df.drop(columns=['token_intervals'], errors='ignore').to_csv(filename, index=False)
    print(f"\nData saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())