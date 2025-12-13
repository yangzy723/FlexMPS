#pragma once
#include "ipc.h"
#include <vector>
#include <thread>
#include <atomic>
#include <map>
#include <mutex>

class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    // 收到新连接的回调
    void onNewClient(std::unique_ptr<IChannel> channel);
    
    // 停止所有服务
    void stop();
    
    // 获取活跃连接数
    size_t getActiveCount();

private:
    void clientHandler(std::unique_ptr<IChannel> channel);
    
    // 业务逻辑
    std::pair<bool, std::string> makeDecision(const std::string& kernelType);
    std::atomic<long long> globalKernelId{0};

    // 线程管理
    std::atomic<bool> running{true};
    std::mutex threadsMutex;
    std::vector<std::thread> workers;
};