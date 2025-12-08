#include "logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::init() {
    mkdir("logs", 0777);
    std::lock_guard<std::mutex> lock(logMutex);
    rotateLogFile(); // Initial creation
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (globalLogFile.is_open()) {
        flushStatsAndReset();
        globalLogFile.close();
    }
}

void Logger::write(const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex); 
    if (globalLogFile.is_open()) {
        globalLogFile << message << std::endl;
        globalLogFile.flush(); 
    }
}

void Logger::recordKernelStat(const std::string& kernelType) {
    std::lock_guard<std::mutex> lock(statsMutex);
    currentLogKernelStats[kernelType]++;
}

void Logger::recordConnectionStat(const std::string& clientKey) {
    std::lock_guard<std::mutex> lock(statsMutex);
    currentLogConnectionStats[clientKey]++;
}

long long Logger::incrementConnectionCount() {
    return connectionCount.fetch_add(1, std::memory_order_relaxed);
}

// 内部辅助函数：必须在持有 logMutex 下调用
void Logger::flushStatsAndReset() {
    std::lock_guard<std::mutex> lock(statsMutex); // 锁住统计数据

    if (!globalLogFile.is_open()) return;

    globalLogFile << "\n-------------------------------------------------------\n";
    globalLogFile << "      Session Statistics\n";
    globalLogFile << "-------------------------------------------------------\n";
    globalLogFile << "Total Connections: " << connectionCount.load() << "\n";

    if (!currentLogConnectionStats.empty()) {
        globalLogFile << "\nConnections by Client:\n";
        for (const auto& item : currentLogConnectionStats) {
            globalLogFile << "  " << item.first << ": " << item.second << " session(s)\n";
        }
    }

    globalLogFile << "\n-------------------------------------------------------\n";
    globalLogFile << "      Kernel Statistics\n";
    globalLogFile << "-------------------------------------------------------\n";

    if (currentLogKernelStats.empty()) {
        globalLogFile << "No kernels recorded in this session.\n";
    } else {
        using PairType = std::pair<std::string, long long>;
        std::vector<PairType> sortedStats(currentLogKernelStats.begin(), currentLogKernelStats.end());

        std::sort(sortedStats.begin(), sortedStats.end(), 
            [](const PairType& a, const PairType& b) {
                return a.second > b.second;
            });

        globalLogFile << std::left << std::setw(50) << "Kernel Name" << " | " << "Count" << "\n";
        globalLogFile << "----------------------------------------------|--------\n";
        
        long long total = 0;
        for (const auto& item : sortedStats) {
            globalLogFile << std::left << std::setw(45) << item.first << " | " << item.second << "\n";
            total += item.second;
        }
        globalLogFile << "----------------------------------------------|--------\n";
        globalLogFile << std::left << std::setw(45) << "TOTAL" << " | " << total << "\n";
    }
    
    globalLogFile << "-------------------------------------------------------\n\n";
    globalLogFile.flush();

    currentLogKernelStats.clear();
    currentLogConnectionStats.clear();
}

std::string Logger::getCurrentTimeStrForFile() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void Logger::rotateLogFile() {
    // 调用者已持有 logMutex
    if (globalLogFile.is_open()) {
        flushStatsAndReset();
        globalLogFile.close();
        std::cout << "[Logger] Rotated log file." << std::endl;
    }

    std::string filename = "logs/" + getCurrentTimeStrForFile() + ".log";
    globalLogFile.open(filename, std::ios::out | std::ios::app);
    
    if (!globalLogFile.is_open()) {
        std::cerr << "[Logger] Fatal: Cannot create " << filename << std::endl;
    } else {
        std::cout << "[Logger] New log file: " << filename << std::endl;
    }
}

Logger::~Logger() {
    shutdown();
}