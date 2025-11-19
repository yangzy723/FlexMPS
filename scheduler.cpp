#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <chrono>
#include <iomanip>
#include <ctime>

#include "IPCProtocol.h"

// ---------------- 全局变量 ----------------

std::mutex logMutex;           // 互斥锁：保护文件读写 和 文件切换
std::ofstream globalLogFile;   // 全局日志文件流
long long connectionCount = 0; // 连接计数器，用于判断是否需要切分文件

// ---------------- 日志辅助函数 ----------------

// 获取当前时间字符串
std::string getCurrentTimeStrForFile() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

// 切换日志文件 (非线程安全，需要在锁内调用)
void rotateLogFile() {
    // 1. 如果旧文件是打开的，先关闭它
    if (globalLogFile.is_open()) {
        globalLogFile.close();
        std::cout << "[Main] 上一轮日志已关闭。" << std::endl;
    }

    // 2. 生成新文件名
    std::string filename = "logs/" + getCurrentTimeStrForFile() + ".log";

    // 3. 打开新文件
    globalLogFile.open(filename, std::ios::out | std::ios::app);
    
    if (!globalLogFile.is_open()) {
        std::cerr << "[Main] 致命错误: 无法创建日志文件 " << filename << std::endl;
        // 这里可以选择不退出，而是打印错误，取决于业务容错要求
    } else {
        std::cout << "[Main] 新的一轮开始，日志文件已创建: " << filename << std::endl;
    }
}

// 线程安全的日志写入
void writeLog(const std::string& message) {
    // 这里加锁，既防止多线程写入冲突，也防止写入时 Main 线程突然把文件关了
    std::lock_guard<std::mutex> lock(logMutex); 
    
    if (globalLogFile.is_open()) {
        globalLogFile << message << std::endl;
        globalLogFile.flush(); 
    } else {
        // 如果文件没打开（极其罕见的情况），输出到控制台防止丢失
        std::cout << "[Log Lost]: " << message << std::endl;
    }
}

// ---------------- 业务逻辑 ----------------

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

void serviceClient(int clientSocket) {
    std::stringstream ss;
    ss << "[Scheduler] 收到连接 (Socket: " << clientSocket << ")";
    writeLog(ss.str()); 

    char buffer[1024] = {0};
    
    while (true) {
        memset(buffer, 0, 1024);
        ssize_t bytesRead = read(clientSocket, buffer, 1023);

        if (bytesRead <= 0) {
            ss.str(""); 
            if (bytesRead == 0) {
                ss << "[Scheduler] Socket " << clientSocket << " 已断开。";
            } else {
                ss << "[Scheduler] Socket " << clientSocket << " 读取错误。";
            }
            writeLog(ss.str());
            close(clientSocket);
            break;              
        }

        std::string message(buffer, bytesRead);
        if (!message.empty() && message.back() == '\n') message.pop_back();
        if (!message.empty() && message.back() == '\r') message.pop_back();

        ss.str("");
        ss << "[Scheduler] 收到请求: " << message << " (Socket: " << clientSocket << ")";
        writeLog(ss.str());

        auto parts = split(message, '|');
        if (parts.size() != 2) {
            writeLog("[Scheduler] 格式错误，断开。");
            close(clientSocket);
            break;
        }
        
        std::string reqId = parts[0];
        std::string kernelType = parts[1];

        auto decision = makeDecision(kernelType);
        bool allowed = decision.first;
        std::string reason = decision.second;

        ss.str("");
        ss << "[Scheduler] 决策 G (ID: " << reqId << "): " << (allowed ? "允许" : "拒绝");
        writeLog(ss.str());

        std::string response = createResponseMessage(reqId, allowed, reason);
        if (send(clientSocket, response.c_str(), response.length(), 0) < 0) {
             writeLog("[Scheduler] 发送响应失败，连接断开。");
             close(clientSocket);
             break;
        }
    }
}

int main() {
    // 程序启动时，只创建文件夹，不创建文件
    mkdir("logs", 0777);

    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 设置端口复用，防止重启时端口被占用
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

    std::cout << "[Scheduler] 服务端运行中 (Port " << SCHEDULER_PORT << ")... 等待每轮 2 个客户端连接" << std::endl;

    while (true) {
        int new_socket;
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }
        
        // --- 核心修改区域：在接收到连接时判断是否需要切分文件 ---
        {
            // 加锁，确保在切换文件时，没有其他线程正在写入旧文件
            std::lock_guard<std::mutex> lock(logMutex);
            
            // 逻辑：每 2 个连接算一轮。
            if (connectionCount % 2 == 0) {
                rotateLogFile();
            }
            
            connectionCount++;
        }
        // ----------------------------------------------------

        std::thread clientThread(serviceClient, new_socket);
        clientThread.detach(); 
    }

    if(globalLogFile.is_open()) globalLogFile.close();
    close(server_fd);
    return 0;
}