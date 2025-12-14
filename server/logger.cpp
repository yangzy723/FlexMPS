#include "logger.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>

// ==========================================
// Logger Implementation (Individual Session)
// ==========================================

Logger::Logger(const std::string& id, const std::string& dirPath) 
    : id_(id), dirPath_(dirPath) {
    
    std::string filename;
    if (id_.empty()) {
        filename = dirPath_ + "/meta.log";
    } else {
        filename = dirPath_ + "/process_" + id_ + ".log";
    }

    fileStream_.open(filename, std::ios::out | std::ios::app);
    if (!fileStream_.is_open()) {
        std::cerr << "[Logger] Error: Failed to open log file: " << filename << std::endl;
    }
}

Logger::~Logger() {
    finalize();
}

void Logger::write(const std::string& message) {
    std::lock_guard<std::mutex> lock(opMutex_);
    if (fileStream_.is_open()) {
        fileStream_ << message << "\n";
        fileStream_.flush(); 
    }
}

void Logger::recordKernelStat(const std::string& kernelType) {
    std::lock_guard<std::mutex> lock(opMutex_);
    kernelStats_[kernelType]++;
}

void Logger::finalize() {
    std::lock_guard<std::mutex> lock(opMutex_);
    
    if (isClosed_ || !fileStream_.is_open()) {
        return;
    }

    fileStream_ << "\n=======================================================\n";
    fileStream_ << "      SESSION STATISTICS (" << (id_.empty() ? "Global" : id_) << ")\n";
    fileStream_ << "=======================================================\n";

    if (kernelStats_.empty()) {
        fileStream_ << "No kernels executed.\n";
    } else {
        using PairType = std::pair<std::string, long long>;
        std::vector<PairType> sortedStats(kernelStats_.begin(), kernelStats_.end());

        std::sort(sortedStats.begin(), sortedStats.end(), 
            [](const PairType& a, const PairType& b) {
                return a.second > b.second;
            });

        fileStream_ << std::left << std::setw(50) << "Kernel Name" << " | " << "Count" << "\n";
        fileStream_ << "---------------------------------------------------|--------\n";
        
        long long total = 0;
        for (const auto& item : sortedStats) {
            fileStream_ << std::left << std::setw(50) << item.first << " | " << item.second << "\n";
            total += item.second;
        }
        fileStream_ << "---------------------------------------------------|--------\n";
        fileStream_ << std::left << std::setw(50) << "TOTAL KERNEL CALLS" << " | " << total << "\n";
    }
    fileStream_ << "=======================================================\n";
    
    fileStream_.flush();
    fileStream_.close();
    isClosed_ = true;
}

// ==========================================
// LogManager Implementation (Global Singleton)
// ==========================================

LogManager& LogManager::instance() {
    static LogManager instance;
    return instance;
}

std::shared_ptr<Logger> LogManager::getLogger(const std::string& unique_id) {
    std::lock_guard<std::mutex> lock(managerMutex_);

    // 1. 如果 Logger 已存在，直接返回
    auto it = activeLoggers_.find(unique_id);
    if (it != activeLoggers_.end()) {
        return it->second;
    }

    // 2. 如果当前没有活跃的 Logger (0 -> 1)，初始化新目录
    if (activeLoggers_.empty()) {
        initDirectory();
    }

    // 3. 创建新的 Logger
    std::shared_ptr<Logger> newLogger(new Logger(unique_id, currentSessionDir_));
    activeLoggers_[unique_id] = newLogger;

    return newLogger;
}

void LogManager::removeLogger(const std::string& unique_id) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    auto it = activeLoggers_.find(unique_id);
    if (it != activeLoggers_.end()) {
        it->second->finalize();
        activeLoggers_.erase(it);
    }
}

LogManager::~LogManager() {
    std::lock_guard<std::mutex> lock(managerMutex_);
    for (auto& pair : activeLoggers_) {
        pair.second->finalize();
    }
    activeLoggers_.clear();
}

// 调用者已持有锁
void LogManager::initDirectory() {
#ifdef _WIN32
    _mkdir("logs");
#else
    mkdir("logs", 0777);
#endif

    std::string timeStr = generateTimeStr();
    currentSessionDir_  = "logs/" + timeStr;
    
#ifdef _WIN32
    _mkdir(currentSessionDir_.c_str());
#else
    mkdir(currentSessionDir_.c_str(), 0777);
#endif
}

std::string LogManager::getSessionDir() const {
    std::lock_guard<std::mutex> lock(managerMutex_);
    return currentSessionDir_;
}

std::string LogManager::generateTimeStr() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}