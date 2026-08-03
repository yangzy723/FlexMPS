# Operator-level determinism benchmark

This benchmark measures how changing a CUDA reduction tree changes floating-point outputs under the accumulation semantics commonly used by production mixed-precision GEMMs:

- FP32 inputs with FP32 accumulation.
- FP16 inputs converted to FP32 for FP32 accumulation.
- BF16 inputs converted to FP32 for FP32 accumulation.

The controlled kernels use scalar CUDA FMA and reductions rather than cuBLAS or WMMA, so this is an arithmetic-semantics benchmark, not a Tensor Core throughput benchmark. It reproduces the relevant low-precision-input/FP32-accumulator behavior while keeping the reduction tree explicitly controllable.

## Experimental contrast

- **Reshaped execution:** changing the partition count or Split-K factor changes the FP32 reduction tree and can change output bits.
- **PROTEUS / fixed structure:** the submitted arithmetic structure stays fixed, producing zero drift for identical inputs.

The experiment compares every reshaped output with a same-dtype fixed-structure baseline. It measures structure-induced drift, not total error against an FP64 oracle. Consequently, BF16 is not expected to have larger drift than FP16 or FP32: input quantization is shared by the baseline and variant, while the changed FP32 reduction tree is the source of the measured difference.

## Input distribution

Inputs follow `x = sign * mantissa * 2^exponent`, where the sign is equiprobable, the mantissa is uniform in `[0.5, 1.0)`, and the integer exponent is uniform in the configured inclusive range. The default is `[-4, 4]`.

## Compile

```bash
nvcc -O3 -std=c++17 operator_determinism.cu -o operator_determinism \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90
```

For A800 only, use `-arch=sm_80`. For H200 only, use `-arch=sm_90`.

## Reduction

```bash
./operator_determinism \
  --experiment reduction \
  --dtype all \
  --trials 100 \
  --size 16777216 \
  --threads 256 \
  --baseline 64 \
  --variants 32,64,128,256 \
  --exp-min -4 --exp-max 4 \
  --allocation-tag proteus-fixed-vs-reshaped \
  --csv reduction.csv
```

## Split-K GEMM

```bash
./operator_determinism \
  --experiment gemm \
  --dtype all \
  --trials 100 \
  --hidden 4096 \
  --vocab 4096 \
  --threads 256 \
  --baseline 1 \
  --variants 1,2,4,8,16 \
  --exp-min -4 --exp-max 4 \
  --allocation-tag proteus-fixed-vs-reshaped \
  --csv splitk_gemm.csv
```

For each trial, the fixed baseline and every reshaped variant receive identical typed inputs.

## Plot

```bash
MPLBACKEND=Agg python plot_operator_determinism.py
```

The figure focuses on FP32 accumulation drift. It combines all non-baseline variants into normalized histograms whose bar heights report the percentage of trials in each bin. The Reduction Kernel and Split-K GEMM Kernel panels use independent linear scales, with the `x10^-2` and `x10^-3` units integrated into their titles. The dashed zero line denotes PROTEUS fixed-structure execution. FP16 and BF16 results remain available in the CSV files but are intentionally omitted from this figure.

The source datasets, `reduction.csv` and `splitk_gemm.csv`, are versioned so the figure can be reproduced. The generated PDF is intentionally excluded from version control.

## CSV fields

- `baseline_shape`, `test_shape`: fixed and reshaped reduction structures.
- `bitwise_equal`: 1 only when all output bits match the fixed baseline.
- `mismatch_fraction`: fraction of output elements with different bits.
- `max_abs`, `max_rel`, `max_ulp`: deviation from the same-dtype fixed baseline.
- `argmax_flip`: whether Split-K changes the largest-logit index.
- `reference_hash`, `test_hash`: exact output hashes.
- `reference_nan_count`, `test_nan_count`, `reference_inf_count`, `test_inf_count`: non-finite diagnostics.
- `exp_min`, `exp_max`: input exponent range.
- `accumulation`: `fp32` for the current benchmark definition.
