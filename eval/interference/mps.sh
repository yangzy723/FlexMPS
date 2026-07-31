# 设置 MPS 环境变量（使用独立目录以防权限冲突）
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p /tmp/nvidia-mps /tmp/nvidia-log

# 启动 MPS 控制守护进程
nvidia-cuda-mps-control -d

# 检查是否启动成功
ps -ef | grep mps