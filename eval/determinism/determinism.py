import torch
import triton
import triton.language as tl
import random
from tqdm import tqdm

# =====================================================================
# 算子 1: Batched MatMul (Split-K)
# =====================================================================
@triton.jit
def split_k_lm_head_kernel(
    hidden_ptr, weight_ptr, logits_ptr,
    stride_hm, stride_hk, stride_wv, stride_wk, stride_lm, stride_lv,
    BATCH_SIZE: tl.constexpr, VOCAB_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m, pid_v, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K) 

    h_ptrs = hidden_ptr + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk)
    w_ptrs = weight_ptr + (offs_v[:, None] * stride_wv + offs_k[None, :] * stride_wk)
    
    h_mask = (offs_m[:, None] < BATCH_SIZE) & (offs_k[None, :] < HIDDEN_DIM)
    w_mask = (offs_v[:, None] < VOCAB_SIZE) & (offs_k[None, :] < HIDDEN_DIM)

    h = tl.load(h_ptrs, mask=h_mask, other=0.0)
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)
    acc = tl.dot(h, tl.trans(w), allow_tf32=False)

    l_ptrs = logits_ptr + (offs_m[:, None] * stride_lm + offs_v[None, :] * stride_lv)
    l_mask = (offs_m[:, None] < BATCH_SIZE) & (offs_v[None, :] < VOCAB_SIZE)
    tl.atomic_add(l_ptrs, acc, mask=l_mask)

def run_lm_head_split_k(hidden, weight, block_m, block_v, block_k):
    BATCH_SIZE, HIDDEN_DIM = hidden.shape
    VOCAB_SIZE = weight.shape[0]
    logits = torch.zeros((BATCH_SIZE, VOCAB_SIZE), device=hidden.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(BATCH_SIZE, meta['BLOCK_M']),
        triton.cdiv(VOCAB_SIZE, meta['BLOCK_V']),
        triton.cdiv(HIDDEN_DIM, meta['BLOCK_K'])
    )
    split_k_lm_head_kernel[grid](
        hidden, weight, logits,
        hidden.stride(0), hidden.stride(1), weight.stride(0), weight.stride(1), logits.stride(0), logits.stride(1),
        BATCH_SIZE, VOCAB_SIZE, HIDDEN_DIM,
        BLOCK_M=block_m, BLOCK_V=block_v, BLOCK_K=block_k
    )
    return logits

# =====================================================================
# 算子 2: Safe Softmax (Row-wise Chunked)
# =====================================================================
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    m_i = -float('inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        m_i = tl.maximum(m_i, tl.max(vals, axis=0))
        
    d_i = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        d_i += tl.sum(tl.exp(vals - m_i), axis=0)
        
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        out_vals = tl.exp(vals - m_i) / d_i
        tl.store(out_row_start_ptr + cols, out_vals, mask=mask)

def run_triton_softmax(logits, block_size):
    BATCH_SIZE, VOCAB_SIZE = logits.shape
    probs = torch.empty_like(logits)
    grid = (BATCH_SIZE, ) 
    softmax_kernel[grid](
        probs, logits, logits.stride(0), probs.stride(0), VOCAB_SIZE, BLOCK_SIZE=block_size
    )
    return probs

def main():
    NUM_TRIALS = 100
    HIDDEN_DIM, VOCAB_SIZE = 4096, 128256
    TARGET_BATCH_SIZE = 4 
    BASE_MATMUL_BLOCK_K, BASE_SOFTMAX_BLOCK_V = 1024, 4096
    
    drift_both_logits, drift_both_probs = [], []
    drift_bs_only_logits, drift_bs_only_probs = [], []
    
    flip_both_count, top5_miss_both = 0, 0
    flip_bs_only_count, top5_miss_bs_only = 0, 0
    
    flip_both_indices = []
    flip_bs_only_indices = []
    
    print(f"=== Starting {NUM_TRIALS} Trials for Data Generation ===")
    
    for i in tqdm(range(NUM_TRIALS), desc="Simulating"):
        raw_hidden = torch.randn((TARGET_BATCH_SIZE, HIDDEN_DIM), device='cuda', dtype=torch.float32)
        target_hidden = (raw_hidden / raw_hidden.norm(dim=-1, keepdim=True) * (HIDDEN_DIM ** 0.5)).to(torch.float16)
        
        weight = (torch.randn((VOCAB_SIZE, HIDDEN_DIM), device='cuda', dtype=torch.float32) * 50).to(torch.float16)

        temp_logits = run_lm_head_split_k(target_hidden, weight, 16, 128, BASE_MATMUL_BLOCK_K)[0]
        top2_vals, top2_indices = torch.topk(temp_logits, 2)
        diff = top2_vals[0] - top2_vals[1]
        
        adjustment_factor = (diff.item() - 1e-4) 
        target_w = weight[top2_indices[0], 0].float()
        target_h = target_hidden[0, 0].float() + 1e-9
        weight[top2_indices[0], 0] = (target_w - adjustment_factor / target_h).half()

        # 1. Baseline
        base_logits = run_lm_head_split_k(target_hidden, weight, 16, 128, BASE_MATMUL_BLOCK_K)
        base_probs = run_triton_softmax(base_logits, BASE_SOFTMAX_BLOCK_V)
        
        base_argmax = torch.argmax(base_probs, dim=1)
        base_top5 = torch.topk(base_probs, k=5, dim=1).indices
        
        noise_batch_size = random.randint(1, 15)
        raw_noise = torch.randn((noise_batch_size, HIDDEN_DIM), device='cuda', dtype=torch.float32)
        noise_hidden = (raw_noise / raw_noise.norm(dim=-1, keepdim=True) * (HIDDEN_DIM ** 0.5)).to(torch.float16)
        batched_hidden = torch.cat([target_hidden, noise_hidden], dim=0)
        
        # 2. Change Both (Spatial Sharing) -> 对应 Baseline
        rand_matmul_block_k = random.choice([64, 128, 256, 512])
        rand_softmax_block_v = random.choice([1024, 2048, 8192])
        
        logits_both = run_lm_head_split_k(batched_hidden, weight, 16, 128, rand_matmul_block_k)
        probs_both = run_triton_softmax(logits_both, rand_softmax_block_v)
        
        target_logits_both = logits_both[:TARGET_BATCH_SIZE, :]
        diffs_both_logits = target_logits_both - base_logits
        max_idx_logits_both = torch.argmax(torch.abs(diffs_both_logits))
        drift_both_logits.append(diffs_both_logits.flatten()[max_idx_logits_both].item())
        
        target_probs_both = probs_both[:TARGET_BATCH_SIZE, :]
        diffs_both_probs = target_probs_both - base_probs
        max_idx_probs_both = torch.argmax(torch.abs(diffs_both_probs))
        drift_both_probs.append(diffs_both_probs.flatten()[max_idx_probs_both].item())
        
        if not torch.equal(base_argmax, torch.argmax(target_probs_both, dim=1)): 
            flip_both_count += 1
            flip_both_indices.append(i)
            
        if not torch.equal(base_top5, torch.topk(target_probs_both, k=5, dim=1).indices): 
            top5_miss_both += 1
            
        # 3. Change Batch Only -> 对应 Vitamin-E
        logits_bs_only = run_lm_head_split_k(batched_hidden, weight, 16, 128, BASE_MATMUL_BLOCK_K)
        probs_bs_only = run_triton_softmax(logits_bs_only, BASE_SOFTMAX_BLOCK_V)
        
        target_logits_bs_only = logits_bs_only[:TARGET_BATCH_SIZE, :]
        diffs_bs_only_logits = target_logits_bs_only - base_logits
        max_idx_logits_bs_only = torch.argmax(torch.abs(diffs_bs_only_logits))
        drift_bs_only_logits.append(diffs_bs_only_logits.flatten()[max_idx_logits_bs_only].item())
        
        target_probs_bs_only = probs_bs_only[:TARGET_BATCH_SIZE, :]
        diffs_bs_only_probs = target_probs_bs_only - base_probs
        max_idx_probs_bs_only = torch.argmax(torch.abs(diffs_bs_only_probs))
        drift_bs_only_probs.append(diffs_bs_only_probs.flatten()[max_idx_probs_bs_only].item())

        if not torch.equal(base_argmax, torch.argmax(target_probs_bs_only, dim=1)): 
            flip_bs_only_count += 1
            flip_bs_only_indices.append(i)
            
        if not torch.equal(base_top5, torch.topk(target_probs_bs_only, k=5, dim=1).indices): 
            top5_miss_bs_only += 1

    experiment_data = {
        "NUM_TRIALS": NUM_TRIALS,
        "drift_both_logits": drift_both_logits,
        "drift_both_probs": drift_both_probs,
        "drift_bs_only_logits": drift_bs_only_logits,
        "drift_bs_only_probs": drift_bs_only_probs,
        "flip_both_count": flip_both_count,
        "flip_bs_only_count": flip_bs_only_count,
        "flip_both_indices": flip_both_indices,
        "flip_bs_only_indices": flip_bs_only_indices
    }
    
    save_path = 'determinism_data.pt'
    torch.save(experiment_data, save_path)
    print(f"\n=== Data Generation Complete. Results saved to {save_path} ===")

if __name__ == "__main__":
    main()