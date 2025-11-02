# Rebuilt-PyTorch

## General

```shell
# env
source <CONDA_INSTALL_DIR>/bin/activate
conda create -y -n <CONDA_NAME>
conda activate <CONDA_NAME>
```

```shell
# code
git clone https://github.com/yangzy723/Rebuilt-PyTorch.git
cd Rebuilt-PyTorch
git submodule sync
git submodule update --init --recursive
```

```shell
# Init
# https://github.com/pytorch/pytorch/tree/v2.8.0
# https://github.com/flashinfer-ai/flashinfer/tree/v0.4.1
# https://github.com/sgl-project/sglang/tree/v0.5.4
```
---

## SgLang
```shell
cd sglang-0.5.4
pip install --upgrade pip
python -m pip install -e "python"
```

## FlashInfer
```shell
cd flashinfer-0.4.1
python -m pip install -e . --no-deps
```

## PyTorch
```shell
cd pytorch-2.8.0
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop
```

## Run
```shell
export CUDA_VISIBLE_DEVICES=3
python -m sglang.bench_one_batch --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ --batch-size 64 --input-len 512
```

## Tips

- 编译`sglang`时使用最新版本（3.13）的 Python 疑似会出现找不到 Rust 编译器的问题
    - Python 3.11
- GLIBCXX_3.4.32 not found
    - conda install -c conda-forge libstdcxx-ng
- https://github.com/sgl-project/sglang/issues/8661
- 自己编译`pytorch 2.8.0`后运行`sglang`，可能需要一个对应版本的`torchvision`，但是`pip`会检查`torchvision`的依赖是否存在（官方的），不存在会帮你下 pytorch
    - pip install torchvision==0.23.0 --no-deps

## Versions

| 模块名称 | 版本 |
|-----------|--------|
| flashinfer_python | 0.4.1 |
| torch | 2.8.0 |
| torch wheel | torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl |
| nvidia_cutlass_dsl | 4.2.1 |
| nvidia_cutlass_dsl wheel | nvidia_cutlass_dsl-4.2.1-cp313-cp313-manylinux_2_28_x86_64.whl |
