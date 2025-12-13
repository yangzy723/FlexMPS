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
API_URL = "http://127.0.0.1:30001/v1/completions"
DATASET_PATH = "BurstGPT_without_fails_2.csv"
MODEL_NAME = "/data/datasets/models-hf/Llama-3.1-8B-Instruct/"

READ_LIMIT = 200000

# --- SLO CONFIGURATION ---
SLO_TTFT = 1     # Seconds (Time To First Token 阈值)
SLO_TPOT = 0.1   # Seconds (Time Per Output Token 阈值)

# ===================================================

# --- CORE CONTROL PARAMETERS ---
SAMPLE_INTERVAL = 1     # Sampling: Pick 1 request every N rows
SPEEDUP_FACTOR = 50    # Time Compression

MAX_REQUESTS = 6000

# ===================================================

VOCAB = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "system", "model", "inference", "performance", "latency", "throughput", "gpu",
    "compute", "memory", "cache", "token", "context", "decode", "prefill", "batch",
    "queue", "request", "server", "client", "python", "async", "await", "test",
    "analysis", "design", "implementation", "result", "discussion", "future", "work"
]

def get_random_prompt_words(token_count):
    prefix = f"REQ_ID_{secrets.token_hex(6)}: "
    # 确保 token_count 是有效的数字，处理可能的 NaN
    if pd.isna(token_count):
        num_words = 10
    else:
        num_words = int(token_count)

    if num_words <= 0: return prefix
    selected_words = random.choices(VOCAB, k=num_words)
    return prefix + " ".join(selected_words)

async def send_request(session, row, start_timestamp_ref, results_list):
    target_time_offset = row['relative_timestamp'] / SPEEDUP_FACTOR

    prompt_text = get_random_prompt_words(row['Request tokens'])

    max_tokens = int(row['Response tokens']) if not pd.isna(row['Response tokens']) else 10

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True
    }

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
        async with session.post(API_URL, json=payload) as response:
            if response.status != 200:
                await response.read()
                return

            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or line == 'data: [DONE]':
                    continue

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

                    except json.JSONDecodeError:
                        continue

            req_end = time.time()
            total_latency = req_end - req_start

            # --- SLO Check ---
            # 1. Check TTFT
            ttft_violated = ttft > SLO_TTFT

            # 2. Check TPOT
            bad_token_intervals_count = sum(1 for t in token_intervals if t > SLO_TPOT)

            results_list.append({
                "send_time": req_start,
                "input_len": int(row['Request tokens']),
                "expected_output_len": max_tokens,
                "actual_output_tokens": token_count,
                "latency": total_latency,
                "ttft": ttft,
                "status": response.status,
                "token_intervals": token_intervals,
                "ttft_violated": ttft_violated,
                "bad_token_intervals_count": bad_token_intervals_count
            })

    except Exception as e:
        pass

async def main():
    print(f"Loading top {READ_LIMIT} rows from trace: {DATASET_PATH}")

    # 读取数据
    df = pd.read_csv(DATASET_PATH, nrows=READ_LIMIT)

    # 假设 Timestamp 列已经是浮点数/数值型 (例如 5270414.0)
    # 确保它是数值型以便排序
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df = df.sort_values('Timestamp')

    if SAMPLE_INTERVAL > 1:
        print(f"Applying sampling: picking 1 request every {SAMPLE_INTERVAL} rows.")
        df = df.iloc[::SAMPLE_INTERVAL].copy()

    if MAX_REQUESTS and MAX_REQUESTS < len(df):
        print(f"Limiting execution to first {MAX_REQUESTS} requests (after sampling).")
        df = df.head(MAX_REQUESTS)

    # 计算相对时间戳 (直接数值相减)
    start_time_base = df['Timestamp'].iloc[0]
    df['relative_timestamp'] = df['Timestamp'] - start_time_base

    total_reqs = len(df)
    print(f"\n=== Benchmark Config ===")
    print(f"Sample Interval : Every {SAMPLE_INTERVAL}th request")
    print(f"Speedup Factor  : {SPEEDUP_FACTOR}x")
    print(f"Actual Requests : {total_reqs}")
    print(f"Mode            : Streaming")
    print(f"SLO Thresholds  : TTFT < {SLO_TTFT}s, TPOT < {SLO_TPOT}s")

    results = []
    connector = aiohttp.TCPConnector(limit=2000)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        benchmark_start_time = time.time()
        print("Starting requests...")

        for _, row in df.iterrows():
            tasks.append(asyncio.create_task(
                send_request(session, row, benchmark_start_time, results)
            ))

        completed_count = 0
        for future in asyncio.as_completed(tasks):
            await future
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_reqs:
                percent = (completed_count / total_reqs) * 100
                sys.stdout.write(f"\r[Progress]: {completed_count}/{total_reqs} ({percent:.1f}%) requests finished.")
                sys.stdout.flush()

        benchmark_end_time = time.time()
        benchmark_duration = benchmark_end_time - benchmark_start_time

        print(f"\nAll tasks completed in {benchmark_duration:.2f} seconds.")

    # --- Statistics ---
    res_df = pd.DataFrame(results)
    csv_df = res_df.drop(columns=['token_intervals'], errors='ignore')
    filename = f"result_sample{SAMPLE_INTERVAL}_speed{SPEEDUP_FACTOR}x.csv"
    csv_df.to_csv(filename, index=False)

    print(f"\n" + "="*45)
    print(f"  RESULTS (Sample: 1/{SAMPLE_INTERVAL}, Speed: {SPEEDUP_FACTOR}x)")
    print(f"="*45)

    if not res_df.empty:
        valid_df = res_df[res_df['status'] == 200]
        count = len(valid_df)
        if count == 0:
            print("No valid requests.")
            return

        # --- 1. E2E Latency ---
        avg_lat = valid_df['latency'].mean()
        p50_lat = valid_df['latency'].quantile(0.50)
        p90_lat = valid_df['latency'].quantile(0.90)
        p99_lat = valid_df['latency'].quantile(0.99)

        # --- 2. TTFT ---
        avg_ttft = valid_df['ttft'].mean()
        p50_ttft = valid_df['ttft'].quantile(0.50)
        p90_ttft = valid_df['ttft'].quantile(0.90)
        p99_ttft = valid_df['ttft'].quantile(0.99)

        # --- Throughput ---
        total_input_tokens = valid_df['input_len'].sum()
        total_output_tokens = valid_df['actual_output_tokens'].sum()

        rps = count / benchmark_duration
        prefill_tps = total_input_tokens / benchmark_duration
        decode_tps = total_output_tokens / benchmark_duration

        # --- 3. TPOT (Token Granularity) ---
        all_intervals = []
        for intervals in valid_df['token_intervals']:
            all_intervals.extend(intervals)

        if all_intervals:
            all_intervals_array = np.array(all_intervals)
            all_intervals_ms = all_intervals_array * 1000

            avg_tpot = np.mean(all_intervals_ms)
            p50_tpot = np.percentile(all_intervals_ms, 50)
            p90_tpot = np.percentile(all_intervals_ms, 90)
            p99_tpot = np.percentile(all_intervals_ms, 99)

            # --- TPOT Violation Calculation (Token Level) ---
            total_intervals_count = len(all_intervals_array)
            total_bad_intervals = np.sum(all_intervals_array > SLO_TPOT)

            if total_intervals_count > 0:
                tpot_violation_rate = (total_bad_intervals / total_intervals_count) * 100
            else:
                tpot_violation_rate = 0.0

        else:
            avg_tpot = 0; p50_tpot=0; p90_tpot=0; p99_tpot=0
            total_bad_intervals = 0
            total_intervals_count = 0
            tpot_violation_rate = 0.0

        # --- 4. TTFT SLO Violation (Request Level) ---
        ttft_violation_count = valid_df['ttft_violated'].sum()
        ttft_violation_rate = (ttft_violation_count / count) * 100


        print(f"Total Successful Requests: {count}")
        print(f"Benchmark Duration     : {benchmark_duration:.2f} s")
        print(f"Total Input Tokens     : {total_input_tokens}")
        print(f"Total Output Tokens    : {total_output_tokens}")
        print(f"Total Token Intervals  : {total_intervals_count} (Excludes first token)")

        print(f"\n[Throughput System-wide]")
        print(f"  Requests/s      : {rps:.2f} req/s")
        print(f"  Prefill Tokens/s: {prefill_tps:.2f} tokens/s")
        print(f"  Decode Tokens/s : {decode_tps:.2f} tokens/s")

        print(f"\n[E2E Latency]")
        print(f"  Avg: {avg_lat:.4f} s")
        print(f"  P50: {p50_lat:.4f} s")
        print(f"  P90: {p90_lat:.4f} s")
        print(f"  P99: {p99_lat:.4f} s")

        print(f"\n[TTFT - Time To First Token]")
        print(f"  Avg: {avg_ttft:.4f} s")
        print(f"  P50: {p50_ttft:.4f} s")
        print(f"  P90: {p90_ttft:.4f} s")
        print(f"  P99: {p99_ttft:.4f} s")

        print(f"\n[Global TPOT - Inter-Token Latency]")
        print(f"  Avg: {avg_tpot:.2f} ms")
        print(f"  P50: {p50_tpot:.2f} ms")
        print(f"  P90: {p90_tpot:.2f} ms")
        print(f"  P99: {p99_tpot:.2f} ms")

        print(f"\n[SLO Violation Rates]")
        print(f"  TTFT Violation Rate (> {SLO_TTFT}s)           : {ttft_violation_rate:.2f}% ({ttft_violation_count}/{count} Requests)")
        print(f"  TPOT Violation Rate (> {SLO_TPOT}s) : {tpot_violation_rate:.2f}% ({total_bad_intervals}/{total_intervals_count} Intervals)")

    else:
        print("No valid responses received.")

if __name__ == "__main__":
    asyncio.run(main())