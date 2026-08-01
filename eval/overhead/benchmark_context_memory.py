#!/usr/bin/env python3
"""Measure resident GPU and host memory of retained CUDA context pools."""
import argparse
import ctypes
import json
import os
import subprocess
import time

class CUdevResource(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("internal_padding", ctypes.c_ubyte * 92),
        ("payload", ctypes.c_ubyte * 48),
    ]

class CUexecAffinityParam(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("sm_count", ctypes.c_uint)]

def gpu_used_mib():
    output = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True)
    return int(output.strip().splitlines()[0])

def rss_mib():
    with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0

class Driver:
    def __init__(self):
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.cuda.cuInit.argtypes = [ctypes.c_uint]
        self.cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.cuda.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.cuda.cuDeviceGetDevResource.argtypes = [
            ctypes.c_int, ctypes.POINTER(CUdevResource), ctypes.c_int]
        self.cuda.cuDevResourceGenerateDesc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUdevResource),
            ctypes.c_uint]
        self.cuda.cuGreenCtxCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int,
            ctypes.c_uint]
        self.cuda.cuGreenCtxDestroy.argtypes = [ctypes.c_void_p]
        self.cuda.cuCtxCreate_v3.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUexecAffinityParam),
            ctypes.c_int, ctypes.c_uint, ctypes.c_int]
        self.cuda.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self.check(self.cuda.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self.check(self.cuda.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")

    @staticmethod
    def check(result, operation):
        if result != 0:
            raise RuntimeError(f"{operation} failed with CUresult={result}")

    def total_sms(self):
        value = ctypes.c_int()
        self.check(self.cuda.cuDeviceGetAttribute(
            ctypes.byref(value), 16, self.device), "cuDeviceGetAttribute")
        return value.value

    def create_green_pool(self, count):
        resource = CUdevResource()
        descriptor = ctypes.c_void_p()
        self.check(self.cuda.cuDeviceGetDevResource(
            self.device, ctypes.byref(resource), 1), "cuDeviceGetDevResource")
        self.check(self.cuda.cuDevResourceGenerateDesc(
            ctypes.byref(descriptor), ctypes.byref(resource), 1),
            "cuDevResourceGenerateDesc")
        contexts, failure, snapshots = [], None, []
        milestones = {count}
        level = 1
        while level < count:
            milestones.add(level)
            level *= 2
        for index in range(count):
            context = ctypes.c_void_p()
            result = self.cuda.cuGreenCtxCreate(
                ctypes.byref(context), descriptor, self.device, 1)
            if result != 0:
                failure = {"index": index, "curesult": result}
                break
            contexts.append(context)
            if len(contexts) in milestones:
                snapshots.append({
                    "created": len(contexts),
                    "gpu_used_mib": gpu_used_mib(),
                    "rss_mib": rss_mib(),
                })
        return contexts, failure, snapshots

    def create_mps_pool(self, sm_counts):
        contexts, failure, snapshots = [], None, []
        for index, sm_count in enumerate(sm_counts):
            affinity = CUexecAffinityParam(0, sm_count)
            context = ctypes.c_void_p()
            result = self.cuda.cuCtxCreate_v3(
                ctypes.byref(context), ctypes.byref(affinity), 1, 0, self.device)
            if result != 0:
                failure = {"index": index, "sm_count": sm_count, "curesult": result}
                break
            popped = ctypes.c_void_p()
            self.check(self.cuda.cuCtxPopCurrent_v2(ctypes.byref(popped)),
                       "cuCtxPopCurrent_v2")
            contexts.append(context)
            snapshots.append({
                "created": len(contexts),
                "sm_count": sm_count,
                "gpu_used_mib": gpu_used_mib(),
                "rss_mib": rss_mib(),
            })
        return contexts, failure, snapshots

    def destroy_green(self, contexts):
        for context in reversed(contexts):
            self.check(self.cuda.cuGreenCtxDestroy(context), "cuGreenCtxDestroy")

    def destroy_mps(self, contexts):
        for context in reversed(contexts):
            self.check(self.cuda.cuCtxDestroy_v2(context), "cuCtxDestroy_v2")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("green", "mps"), required=True)
    parser.add_argument("--count", type=int, default=350)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    baseline_gpu = gpu_used_mib()
    baseline_rss = rss_mib()
    driver = Driver()
    after_init_gpu = gpu_used_mib()
    after_init_rss = rss_mib()
    total_sms = driver.total_sms()
    if args.mode == "green":
        contexts, failure, snapshots = driver.create_green_pool(args.count)
        sm_counts = None
    else:
        sm_counts = [max(1, round(total_sms * level / args.count))
                     for level in range(1, args.count + 1)]
        contexts, failure, snapshots = driver.create_mps_pool(sm_counts)
    time.sleep(args.settle_seconds)
    held_gpu = gpu_used_mib()
    held_rss = rss_mib()

    result = {
        "mode": args.mode,
        "requested": args.count,
        "created": len(contexts),
        "sm_counts": sm_counts,
        "device_sm_count": total_sms,
        "measurement_kind": (
            "green_context_pool" if args.mode == "green"
            else "execution_affinity_context_hierarchy"
        ),
        "failure": failure,
        "snapshots": snapshots,
        "baseline_gpu_mib": baseline_gpu,
        "after_driver_init_gpu_mib": after_init_gpu,
        "held_gpu_mib": held_gpu,
        "pool_gpu_delta_from_baseline_mib": held_gpu - baseline_gpu,
        "pool_gpu_delta_after_driver_init_mib": held_gpu - after_init_gpu,
        "baseline_rss_mib": baseline_rss,
        "after_driver_init_rss_mib": after_init_rss,
        "held_rss_mib": held_rss,
        "pool_rss_delta_from_baseline_mib": held_rss - baseline_rss,
        "pool_rss_delta_after_driver_init_mib": held_rss - after_init_rss,
    }
    print(json.dumps(result, sort_keys=True))

    if args.mode == "green":
        driver.destroy_green(contexts)
    else:
        driver.destroy_mps(contexts)
    time.sleep(args.settle_seconds)
    print(json.dumps({
        "mode": args.mode,
        "after_destroy_gpu_mib": gpu_used_mib(),
        "after_destroy_rss_mib": rss_mib(),
    }, sort_keys=True))

if __name__ == "__main__":
    main()
