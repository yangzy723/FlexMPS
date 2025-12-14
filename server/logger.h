#pragma once

#include <string>
#include <mutex>
#include <fstream>
#include <unordered_map>
#include <map>
#include <memory>
#include <atomic>
#include <vector>

class LogManager;

/**
 * @brief 负责单个 Session/Client 的日志写入和统计
 * 每个客户端连接对应一个 Logger 实例
 */
class Logger {
public:
    // 禁止默认构造和拷贝，强制通过 Manager 创建
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    ~Logger();

    // 核心功能
    void write(const std::string& message);
    void recordKernelStat(const std::string& kernelType);
    void kernelIdIncrement();
    long long getKernelId() const;
    
    // 显式关闭，通常由 Manager 调用，或者析构时自动调用
    void finalize();

private:
    // 仅允许 LogManager 创建 Logger 实例
    friend class LogManager;
    Logger(const std::string& id, const std::string& dirPath);

private:
    const std::string id_;
    const std::string dirPath_;
    std::ofstream fileStream_;
    std::mutex opMutex_;
    bool isClosed_ = false;
    std::atomic<long long> kernelId{0};

    // 统计数据
    std::map<std::string, long long> kernelStats_;
};

/**
 * @brief 全局日志管理器 (单例)
 * 负责目录创建逻辑 (0->1) 和 Logger 生命周期的管理
 */
class LogManager {
public:
    static LogManager& instance();

    // 获取或创建一个指定 ID 的 Logger
    // 如果这是第一个连接，会自动初始化目录
    std::shared_ptr<Logger> getLogger(const std::string& unique_id);

    void sessionIdIncrement();
    long long getSessionId();

    // 当客户端断开连接时调用，触发统计写入并释放资源
    void removeLogger(const std::string& unique_id);

private:
    LogManager() = default;
    ~LogManager(); // 析构时会关闭所有剩余 Logger

    void initDirectory();
    std::string generateTimeStr();

private:
    mutable std::mutex managerMutex_;
    std::unordered_map<std::string, std::shared_ptr<Logger>> activeLoggers_;
    std::string currentSessionDir_;
    std::atomic<long long> sessionId_{0};
};