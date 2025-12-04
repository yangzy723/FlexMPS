#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <map>
#include <algorithm>
#include <signal.h>
#include <memory>

#include "IPCProtocol.h"

// ---------------- 结构定义 ----------------

// 描述一个进程的会话状态（多个 Socket 共享此对象）
struct ProcessSession {
    std::mutex sessionMutex;                 // 保护该 Session 的文件和统计数据
    std::ofstream logFile;                   // 专属日志文件流
    std::map<std::string, long long> stats;  // 聚合统计数据
    int activeConnectionCount = 0;           // 当前活跃的 Socket 连接数
    bool isLogOpen = false;
};

// ---------------- 全局变量 ----------------

std::mutex consoleMutex; 
std::atomic<long long> globalKernelId(0);

std::mutex globalSessionMapMutex; 
std::map<std::string, std::shared_ptr<ProcessSession>> globalSessionMap;

// ---------------- 辅助函数 ----------------

std::string getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::pair<bool, std::string> makeDecision(const std::string& kernelType) {
    return {true, "OK"};
}

void flushStatsToStream(std::ofstream& logFile, std::map<std::string, long long>& stats, const std::string& pid) {
    if (!logFile.is_open()) return;

    logFile << "\n-------------------------------------------------------\n";
    logFile << "             Summary for Process " << pid << "\n";
    logFile << "-------------------------------------------------------\n";

    if (stats.empty()) {
        logFile << "No kernels recorded in this session.\n";
    } else {
        using PairType = std::pair<std::string, long long>;
        std::vector<PairType> sortedStats(stats.begin(), stats.end());

        std::sort(sortedStats.begin(), sortedStats.end(), 
            [](const PairType& a, const PairType& b) {
                return a.second > b.second;
            });

        logFile << std::left << std::setw(45) << "Kernel Name" << " | " << "Count" << "\n";
        logFile << "----------------------------------------------|--------\n";
        
        long long total = 0;
        for (const auto& item : sortedStats) {
            logFile << std::left << std::setw(45) << item.first << " | " << item.second << "\n";
            total += item.second;
        }
        logFile << "----------------------------------------------|--------\n";
        logFile << std::left << std::setw(45) << "TOTAL" << " | " << total << "\n";
    }
    logFile << "-------------------------------------------------------\n\n";
    logFile.flush();
}

// ---------------- 核心逻辑 ----------------

std::shared_ptr<ProcessSession> getOrCreateSession(const std::string& pid) {
    std::lock_guard<std::mutex> lock(globalSessionMapMutex);
    
    if (globalSessionMap.find(pid) == globalSessionMap.end()) {
        auto newSession = std::make_shared<ProcessSession>();
        
        std::string filename = "logs/process_" + pid + ".log";
        newSession->logFile.open(filename, std::ios::out | std::ios::app);
        
        if (newSession->logFile.is_open()) {
            newSession->isLogOpen = true;
            newSession->logFile << "=== Session Started (New Socket Group): " << getCurrentTimeStr() << " ===\n";
        }
        
        globalSessionMap[pid] = newSession;
        
        {
            std::lock_guard<std::mutex> clk(consoleMutex);
            std::cout << "[Manager] 进程 " << pid << " 新建会话组 (Log: " << filename << ")" << std::endl;
        }
    }
    
    auto session = globalSessionMap[pid];
    {
        std::lock_guard<std::mutex> slock(session->sessionMutex);
        session->activeConnectionCount++;
    }
    
    return session;
}

void handleDisconnect(const std::string& pid, std::shared_ptr<ProcessSession> session) {
    if (!session) return;

    std::lock_guard<std::mutex> gLock(globalSessionMapMutex);
    std::lock_guard<std::mutex> sLock(session->sessionMutex);

    session->activeConnectionCount--;
    
    {
        std::lock_guard<std::mutex> clk(consoleMutex);
        std::cout << "[Manager] 进程 " << pid << " 一个连接断开。剩余连接数: " << session->activeConnectionCount << std::endl;
    }

    if (session->activeConnectionCount <= 0) {
        if (session->isLogOpen) {
            flushStatsToStream(session->logFile, session->stats, pid);
            session->logFile << "=== Session Ended: " << getCurrentTimeStr() << " ===\n";
            session->logFile.close();
        }
        globalSessionMap.erase(pid);
        
        std::lock_guard<std::mutex> clk(consoleMutex);
        std::cout << "[Manager] 进程 " << pid << " 所有连接已断开，统计信息已写入。" << std::endl;
    }
}

void serviceClient(int clientSocket) {
    std::string currentPid = "";
    std::shared_ptr<ProcessSession> currentSession = nullptr;

    char rawBuffer[1024];
    std::string bufferStr = ""; 

    while (true) {
        ssize_t bytesRead = read(clientSocket, rawBuffer, 1023);

        if (bytesRead <= 0) {
            break;
        }

        rawBuffer[bytesRead] = '\0';
        bufferStr += rawBuffer;

        size_t pos = 0;
        while ((pos = bufferStr.find('\n')) != std::string::npos) {
            std::string message = bufferStr.substr(0, pos);
            bufferStr.erase(0, pos + 1);

            if (!message.empty() && message.back() == '\r') message.pop_back();
            if (message.empty()) continue;

            // 协议: KernelType|ReqId|Source|PID
            auto parts = split(message, '|');
            
            if (parts.size() < 4) {
                // 格式错误，如果是初次连接且没 PID，无法归档
                continue; 
            }

            std::string kernelType = parts[0];     
            std::string reqId = parts[1];        
            std::string source = parts[2];
            std::string pid = parts[3];

            // 第一次收到有效消息时，绑定 Session
            if (currentSession == nullptr) {
                currentPid = pid;
                currentSession = getOrCreateSession(pid);
            } else {
                // 安全检查：防止同一个 Socket 突然发了别的 PID
                if (pid != currentPid) {
                    // 记录个警告，继续处理
                }
            }

            long long currentId = ++globalKernelId;

            // --- 记录数据到共享 Session ---
            if (currentSession) {
                std::lock_guard<std::mutex> slock(currentSession->sessionMutex);
                
                // 聚合统计
                currentSession->stats[kernelType]++;
                
                // 实时写入单条日志
                if (currentSession->isLogOpen) {
                    currentSession->logFile << "[" << getCurrentTimeStr() << "] "
                                            << "Socket(" << clientSocket << ") "
                                            << "Req(" << reqId << ") " 
                                            << kernelType << "\n";
                }
            }
            // -----------------------------

            auto decision = makeDecision(kernelType);
            std::string response = createResponseMessage(reqId, decision.first, decision.second);
            
            if (send(clientSocket, response.c_str(), response.length(), MSG_NOSIGNAL) < 0) {
                 goto cleanup;
            }
        }
    }

cleanup:
    if (currentSession) {
        handleDisconnect(currentPid, currentSession);
    }
    close(clientSocket);
}

int main() {
    signal(SIGPIPE, SIG_IGN);
    mkdir("logs", 0777);

    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SCHEDULER_PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 10) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    std::cout << "[Scheduler] 多连接聚合统计模式运行中 (Port " << SCHEDULER_PORT << ")... " << std::endl;

    while (true) {
        int new_socket;
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }
        
        std::thread clientThread(serviceClient, new_socket);
        clientThread.detach(); 
    }
    
    close(server_fd);
    return 0;
}