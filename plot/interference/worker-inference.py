# 文件名: inference_worker.py
import torch
import time
import numpy as np
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

def run_inference_with_metrics():
    model_path = "/data/datasets/models-hf/Qwen3-32B"
    print(f"[Inference] 加载模型 {model_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="cuda:0" # 依赖 CUDA_VISIBLE_DEVICES=1 进行系统级映射
    )
    
    print("[Inference] 构造 512 Tokens 的输入上下文...")
    dummy_text = "system scheduling and virtualization " * 200 
    encoded = tokenizer(dummy_text, return_tensors="pt")
    
    # 精确截断至 512
    input_ids = encoded.input_ids[:, :512].to("cuda:0")
    attention_mask = torch.ones_like(input_ids).to("cuda:0")
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("[Inference] 开始预热...")
    _ = model.generate(**inputs, max_new_tokens=5)
    torch.cuda.synchronize()

    num_requests = 10
    print(f"[Inference] 开始正式测试 ({num_requests} 次请求, Prefill=512)...")
    
    ttft_list = []
    tpot_list = []
    
    for i in range(num_requests):
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=20,
            min_new_tokens=20
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        thread.start()
        
        first_token_received = False
        tokens_received = 0
        first_token_time = 0
        
        for new_text in streamer:
            current_time = time.perf_counter()
            if not first_token_received:
                ttft = (current_time - start_time) * 1000
                ttft_list.append(ttft)
                first_token_time = current_time
                first_token_received = True
            else:
                tokens_received += 1
                
        thread.join()
        torch.cuda.synchronize()
        
        if tokens_received > 0:
            tpot = ((time.perf_counter() - first_token_time) * 1000) / tokens_received
            tpot_list.append(tpot)
        else:
            tpot = 0.0

        print(f"  -> 请求 {i+1}: TTFT = {ttft:.2f} ms | TPOT = {tpot:.2f} ms")

    print("\n" + "="*50)
    print("[Inference] 实验结果汇总 (Prefill Length = 512)")
    print(f"P99 TTFT (首字延迟): {np.percentile(ttft_list, 99):.2f} ms")
    print(f"P99 TPOT (单字延迟): {np.percentile(tpot_list, 99):.2f} ms")
    print("="*50)

if __name__ == "__main__":
    run_inference_with_metrics()