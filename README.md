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

## FlashInfer
```shell
cd flashinfer-v0.4.1
python -m pip install -e . --no-deps
```

## SgLang
```shell
cd sglang-v0.5.4
pip install --upgrade pip
python -m pip install -e "python"
pip uninstall torch
```

## PyTorch
```shell
cd pytorch-v2.8.0
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop
```

## Run
```shell
export CUDA_VISIBLE_DEVICES=1
python -m sglang.bench_one_batch --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ --batch-size 64 --input-len 512
```

## Tips
- 编译`sglang`时使用最新版本（3.13）的 Python 疑似会出现找不到 Rust 编译器的问题
    - Python 3.11
- GLIBCXX_3.4.32 not found
    - conda install -c conda-forge libstdcxx-ng
- H200 最低支持的 CUDA 版本为 12.4，不支持 gcc-13/g++-13，需要手动软链接为gcc-12
    - 如果使用 conda，方法为：
    - ls /usr/bin | grep gcc
    - cd $(dirname $(which python))
    - ln -s /usr/bin/gcc-12 gcc
    - ln -s /usr/bin/g++-12 g++
- 更改 pytorch 编译时需要的 CUDA 版本
    - build/CMakeCache.txt -> CMAKE_CUDA_COMPILER:STRING=/usr/local/cuda-12.9/bin/nvcc
- 自己编译`pytorch 2.8.0`后运行`sglang`，可能需要一个对应版本的`torchvision`，但是`pip`会检查`torchvision`的依赖是否存在（官方的），不存在会帮你下 pytorch
    - pip install torchvision==0.23.0 --no-deps
- https://github.com/sgl-project/sglang/issues/8661

## Versions

|模块名称       | 版本  |
|--------------|--------|
|cuda          |12.9  |
|python        |3.11  |
|flashinfer    |0.4.1 |
|torch         |2.8.0 |
|sglang        |0.5.4 |
