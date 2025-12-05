#!/bin/bash

# 请求数量配置
REQ_COUNT=32

echo "========================================"
echo "      LLM Inference Benchmark Tool      "
echo "========================================"

# 清理旧数据
rm -f res_*.json

# ----------------------------------------------
# 阶段 1: 串行基准测试 (Serial Baseline)
# ----------------------------------------------
echo ""
echo ">>> [Phase 1] Running SERIAL Mode (Baseline)..."
echo "    Running Prefill..."
python prefill_client.py --count $REQ_COUNT --output res_serial_prefill.json
echo "    Prefill Done. Sleeping 2s..."
sleep 2

echo "    Running Decode..."
python decode_client.py --count $REQ_COUNT --output res_serial_decode.json
echo "    Decode Done."

# ----------------------------------------------
# 阶段 2: 并行干扰测试 (Parallel Interference)
# ----------------------------------------------
echo ""
echo ">>> [Phase 2] Running PARALLEL Mode (Interference)..."
echo "    Sleeping 3s to cool down GPU..."
sleep 3

echo "    Launching BOTH Prefill and Decode simultaneously..."

# start_time=$(date +%s%N)

# 后台运行 (&)
python prefill_client.py --count $REQ_COUNT --output res_parallel_prefill.json &
PID_P=$!

python decode_client.py --count $REQ_COUNT --output res_parallel_decode.json &
PID_D=$!

echo "    Jobs launched (PIDs: $PID_P, $PID_D). Waiting for completion..."
wait $PID_P $PID_D

# end_time=$(date +%s%N)
# duration=$((($end_time - $start_time)/1000000))
echo "    Parallel run finished."

# ----------------------------------------------
# 阶段 3: 生成报告
# ----------------------------------------------
echo ""
echo ">>> Generating Analysis Report..."
python analyze_results.py