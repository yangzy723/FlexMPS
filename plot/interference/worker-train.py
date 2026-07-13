# 文件名: training_worker.py
import torch

def run_heavy_workload():
    print("[Training] 启动后台高吞吐负载...")
    N = 16384 
    A = torch.randn(N, N, device='cuda:0', dtype=torch.float16)
    B = torch.randn(N, N, device='cuda:0', dtype=torch.float16)

    print("[Training] 持续霸占 SM (按 Ctrl+C 停止)...")
    try:
        while True:
            _ = torch.matmul(A, B)
    except KeyboardInterrupt:
        print("[Training] 停止后台负载。")

if __name__ == "__main__":
    run_heavy_workload()