#include "logger.h"
#include "ipc.h"
#include "shm_core.h"
#include "scheduler.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <signal.h>
#include <unistd.h>

std::atomic<bool> g_app_running(true);

void signalHandler(int signum) {
    std::cout << "\n[Main] Received signal " << signum << ", shutting down..." << std::endl;
    g_app_running = false;
}

int main() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // 初始化日志
    Logger::instance().init();

    // 初始化核心调度器
    Scheduler scheduler;

    // 初始化 IPC 服务 (使用共享内存实现)
    ShmServer ipcServer;
    
    std::cout << "[Main] Initializing IPC..." << std::endl;
    if (!ipcServer.init()) {
        std::cerr << "[Main] Failed to init IPC" << std::endl;
        return 1;
    }

    // 绑定：当 IPC 发现新客户端时，交给 Scheduler 处理
    ipcServer.start([&scheduler](std::unique_ptr<IChannel> channel) {
        scheduler.onNewClient(std::move(channel));
    });

    std::cout << "[Main] System running. Press Ctrl+C to exit." << std::endl;

    // 主循环 (监控与维护)
    auto lastRotate = std::chrono::steady_clock::now();

    while (g_app_running) {
        sleep(1);

        auto now = std::chrono::steady_clock::now();
        
        // 日志轮转 (每分钟)
        if (std::chrono::duration_cast<std::chrono::minutes>(now - lastRotate).count() >= 1) {
            Logger::instance().rotateLogFile();
            lastRotate = now;
        }
    }

    // 优雅退出
    std::cout << "[Main] Stopping services..." << std::endl;
    ipcServer.stop();
    scheduler.stop();
    Logger::instance().shutdown();

    std::cout << "[Main] Bye." << std::endl;
    return 0;
}