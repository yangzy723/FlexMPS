cd ./sglang-v0.5.4
python -m pip install -e "python"
pip uninstall torch
cd ../pytorch-v2.8.0
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop
cd ..
g++ -std=c++11 -pthread -lrt -o scheduler scheduler.cpp
export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
export CUDA_VISIBLE_DEVICES=1