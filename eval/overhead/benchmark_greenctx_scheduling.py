#!/usr/bin/env python3
"""Benchmark GreenCtx lifecycle, policy traversal, and model-shaped GEMM phases."""
import argparse
import ctypes
import json
import math
import statistics
import time

class CUdevResource(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("internal_padding", ctypes.c_ubyte * 92),
        ("payload", ctypes.c_ubyte * 48),
    ]

class GreenContextFactory:
    """Provision and destroy one real minimal Hopper Green Context."""
    def __init__(self):
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.cuda.cuDeviceGetDevResource.argtypes = [
            ctypes.c_int, ctypes.POINTER(CUdevResource), ctypes.c_int]
        self.cuda.cuDevSmResourceSplitByCount.argtypes = [
            ctypes.POINTER(CUdevResource), ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(CUdevResource), ctypes.POINTER(CUdevResource),
            ctypes.c_uint, ctypes.c_uint]
        self.cuda.cuDevResourceGenerateDesc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUdevResource),
            ctypes.c_uint]
        self.cuda.cuGreenCtxCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int,
            ctypes.c_uint]
        self.cuda.cuGreenCtxDestroy.argtypes = [ctypes.c_void_p]

    @staticmethod
    def check(result, operation):
        if result != 0:
            raise RuntimeError(f"{operation} failed with CUresult={result}")

    def create_one(self):
        source, selected, remainder = CUdevResource(), CUdevResource(), CUdevResource()
        groups = ctypes.c_uint(1)
        descriptor, green_ctx = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.cuda.cuDeviceGetDevResource(
            0, ctypes.byref(source), 1), "cuDeviceGetDevResource")
        self.check(self.cuda.cuDevSmResourceSplitByCount(
            ctypes.byref(selected), ctypes.byref(groups), ctypes.byref(source),
            ctypes.byref(remainder), 0, 8), "cuDevSmResourceSplitByCount")
        self.check(self.cuda.cuDevResourceGenerateDesc(
            ctypes.byref(descriptor), ctypes.byref(selected), 1),
            "cuDevResourceGenerateDesc")
        self.check(self.cuda.cuGreenCtxCreate(
            ctypes.byref(green_ctx), descriptor, 0, 1), "cuGreenCtxCreate")
        return green_ctx

    def destroy(self, green_ctx):
        self.check(self.cuda.cuGreenCtxDestroy(green_ctx), "cuGreenCtxDestroy")

def run_policy(candidates):
    """Mock O(N) policy: choose the least-loaded feasible Green Context."""
    best_id, best_score = -1, 1 << 62
    for ctx_id, active_kernels, free_sms in candidates:
        if free_sms >= 8:
            score = active_kernels * 4096 - free_sms
            if score < best_score:
                best_id, best_score = ctx_id, score
    return best_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=28672)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--greenctx-create", action="store_true",
                        help="create and destroy one real GreenCtx per sample")
    parser.add_argument("--policy-candidates", type=int, default=0,
                        help="number of candidates scanned by the mock O(N) policy")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if ctypes.sizeof(CUdevResource) != 144:
        raise SystemExit(f"unexpected CUdevResource size: {ctypes.sizeof(CUdevResource)}")

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.bfloat16)
    factory = GreenContextFactory() if args.greenctx_create else None
    candidates = [(i, (i * 17 + 3) % 23, 8 + (i * 13) % 125)
                  for i in range(args.policy_candidates)]
    for _ in range(args.warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    samples, greenctx_ms, destroy_ms, policy_ms = [], [], [], []
    selected_ctx = -1
    for _ in range(args.iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        phase_start = time.perf_counter()
        green_ctx = None
        if factory:
            green_ctx = factory.create_one()
        greenctx_ms.append((time.perf_counter() - phase_start) * 1000)
        phase_start = time.perf_counter()
        selected_ctx = run_policy(candidates)
        policy_ms.append((time.perf_counter() - phase_start) * 1000)
        start_event.record()
        out = torch.mm(a, b)
        end_event.record()
        torch.cuda.synchronize()
        phase_start = time.perf_counter()
        if green_ctx:
            factory.destroy(green_ctx)
        destroy_ms.append((time.perf_counter() - phase_start) * 1000)
        samples.append((start_event.elapsed_time(end_event),
                        (time.perf_counter() - wall_start) * 1000))

    gpu_ms = [x[0] for x in samples]
    wall_ms = [x[1] for x in samples]
    flops = 2.0 * args.m * args.n * args.k
    result = {
        "label": args.label,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "shape": [args.m, args.k, args.n],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "greenctx_create": args.greenctx_create,
        "greenctx_ms_mean": statistics.mean(greenctx_ms),
        "greenctx_ms_median": statistics.median(greenctx_ms),
        "destroy_ms_mean": statistics.mean(destroy_ms),
        "destroy_ms_median": statistics.median(destroy_ms),
        "policy_candidates": args.policy_candidates,
        "policy_ms_mean": statistics.mean(policy_ms),
        "policy_ms_median": statistics.median(policy_ms),
        "selected_ctx": selected_ctx,
        "gpu_ms_mean": statistics.mean(gpu_ms),
        "gpu_ms_median": statistics.median(gpu_ms),
        "wall_ms_mean": statistics.mean(wall_ms),
        "wall_ms_median": statistics.median(wall_ms),
        "wall_ms_p95": sorted(wall_ms)[math.ceil(0.95 * len(wall_ms)) - 1],
        "wall_ms_max": max(wall_ms),
        "wall_ms_stdev": statistics.stdev(wall_ms) if len(wall_ms) > 1 else 0,
        "tflops_from_wall_median":
            flops / (statistics.median(wall_ms) / 1000) / 1e12,
        "checksum": float(out[0, 0]),
    }
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
