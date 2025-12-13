#pragma once

#include <string>
#include <mutex>
#include <fstream>
#include <unordered_map>
#include <atomic>
#include <vector>

class Logger {
public:
    static void initDir();
    static Logger& instance(std::string unique_id = "");

    void write(const std::string& message);
    void recordKernelStat(const std::string& kernelType);
    void recordConnectionStat(const std::string& clientKey);
    long long incrementConnectionCount();
    
    void shutdown();

private:
    Logger() = default;
    ~Logger();
    
    void openLogFile(const std::string& unique_id);
    void flushStatsToLog();
    static std::string getSessionDirectory();
    static std::string getCurrentTimeStr();

    static std::mutex registryMutex;
    static std::unordered_map<std::string, Logger*> loggerMap;
    static std::string sessionDirectory;

    std::string loggerId;
    std::ofstream globalLogFile;
    std::mutex logMutex;
    
    std::mutex statsMutex;
    std::unordered_map<std::string, long long> currentLogKernelStats;
    std::unordered_map<std::string, long long> currentLogConnectionStats;
    std::atomic<long long> connectionCount{0};
    
    // 禁止拷贝
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};