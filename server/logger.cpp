#include "logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

std::mutex Logger::registryMutex;
std::unordered_map<std::string, Logger*> Logger::loggerMap;
std::string Logger::sessionDirectory = "";

void Logger::initDir() {
#ifdef _WIN32
    mkdir("logs");
#else
    mkdir("logs", 0777);
#endif

    std::string timeStr = getCurrentTimeStr();
    std::string dirPath = "logs/" + timeStr;

#ifdef _WIN32
    mkdir(dirPath.c_str());
#else
    mkdir(dirPath.c_str(), 0777);
#endif

    sessionDirectory = dirPath;
}

Logger& Logger::instance(std::string unique_id) {
    std::lock_guard<std::mutex> lock(registryMutex);

    auto it = loggerMap.find(unique_id);
    if (it == loggerMap.end()) {
        Logger* newLogger = new Logger();
        newLogger->loggerId = unique_id;
        loggerMap[unique_id] = newLogger;
        newLogger->openLogFile(unique_id);
        return *newLogger;
    }
    return *(it->second);
}

std::string Logger::getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void Logger::openLogFile(const std::string& unique_id) {
    std::lock_guard<std::mutex> lock(logMutex);

    std::string filename;
    if (unique_id.empty()) {
        filename = sessionDirectory + "/meta.log";
    } else {
        filename = sessionDirectory + "/process_" + unique_id + ".log";
    }

    globalLogFile.open(filename, std::ios::out | std::ios::app);
    if (!globalLogFile.is_open()) {
        std::cerr << "[Logger] Fatal: Cannot create " << filename << std::endl;
    } else {
        // std::cout << "[Logger] Opened file: " << filename << std::endl;
    }
}

void Logger::write(const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex); 
    if (globalLogFile.is_open()) {
        globalLogFile << message << std::endl;
        globalLogFile.flush(); 
    }
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (globalLogFile.is_open()) {
        flushStatsToLog();
        globalLogFile.close();
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

void Logger::flushStatsToLog() {
    std::lock_guard<std::mutex> lock(statsMutex);

    if (!globalLogFile.is_open()) return;

    globalLogFile << "\n=======================================================\n";
    globalLogFile << "      FINAL SESSION STATISTICS\n";
    globalLogFile << "=======================================================\n";
    
    // 1. 连接总数
    globalLogFile << "Total Connections Processed: " << connectionCount.load() << "\n";

    // 2. 客户端连接详情
    if (!currentLogConnectionStats.empty()) {
        globalLogFile << "\n--- Connections by Client ---\n";
        for (const auto& item : currentLogConnectionStats) {
            globalLogFile << "  [" << item.first << "] : " << item.second << " session(s)\n";
        }
    }

    // 3. Kernel 统计
    globalLogFile << "\n--- Kernel Execution Statistics ---\n";
    if (currentLogKernelStats.empty()) {
        globalLogFile << "No kernels recorded.\n";
    } else {
        using PairType = std::pair<std::string, long long>;
        std::vector<PairType> sortedStats(currentLogKernelStats.begin(), currentLogKernelStats.end());

        // 按调用次数降序排列
        std::sort(sortedStats.begin(), sortedStats.end(), 
            [](const PairType& a, const PairType& b) {
                return a.second > b.second;
            });

        globalLogFile << std::left << std::setw(50) << "Kernel Name" << " | " << "Count" << "\n";
        globalLogFile << "---------------------------------------------------|--------\n";
        
        long long total = 0;
        for (const auto& item : sortedStats) {
            globalLogFile << std::left << std::setw(50) << item.first << " | " << item.second << "\n";
            total += item.second;
        }
        globalLogFile << "---------------------------------------------------|--------\n";
        globalLogFile << std::left << std::setw(50) << "TOTAL KERNEL CALLS" << " | " << total << "\n";
    }
    
    globalLogFile << "=======================================================\n\n";
    globalLogFile.flush();
}

Logger::~Logger() {
    shutdown();
}