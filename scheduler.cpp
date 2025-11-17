#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <chrono>   // 用于获取高精度时间
#include <iomanip>  // 用于格式化时间

#include "IPCProtocol.h"

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
    // Your Policy
    return {true, "OK"};
}

void serviceClient(int clientSocket) {
    
    std::string logFilename = "client_socket_" + std::to_string(clientSocket) + ".log";
    std::ofstream logStream(logFilename);

    if (!logStream.is_open()) {
        std::cerr << "[Scheduler-Main] 致命错误: 无法为 Socket " << clientSocket 
                  << " 创建日志文件: " << logFilename << std::endl;
        close(clientSocket);
        return;
    }

    logStream << "\n[Scheduler] 收到一个新的连接。开始服务该连接 (Socket: " << clientSocket << ")" << std::endl;

    char buffer[1024] = {0};
    
    while (true) {
        memset(buffer, 0, 1024);
        ssize_t bytesRead = read(clientSocket, buffer, 1023);

        if (bytesRead <= 0) {
            if (bytesRead == 0) {
                logStream << "[Scheduler] 客户端 (Socket " << clientSocket << ") 已优雅断开连接。" << std::endl;
            } else {
                perror(("[Scheduler] Socket " + std::to_string(clientSocket) + " 读取错误").c_str());
                logStream << "[Scheduler] Socket " << clientSocket << " 读取错误，断开连接。" << std::endl;
            }
            close(clientSocket);
            break;              
        }

        std::string message(buffer, bytesRead);
        message.erase(message.find_last_not_of("\r\n") + 1); 

        logStream << "[Scheduler] 收到请求: " << message << std::endl;

        auto parts = split(message, '|');
        if (parts.size() != 2) {
            logStream << "[Scheduler] 请求格式错误: " << message << "，断开连接。" << std::endl;
            close(clientSocket);
            break;
        }
        
        std::string reqId = parts[0];
        std::string kernelType = parts[1];

        auto decision = makeDecision(kernelType);
        bool allowed = decision.first;
        std::string reason = decision.second;

        logStream << "[Scheduler] G (ID: " << reqId << "): " 
                  << (allowed ? "允许" : "拒绝") << std::endl;

        std::string response = createResponseMessage(reqId, allowed, reason);
        
        if (send(clientSocket, response.c_str(), response.length(), 0) < 0) {
             logStream << "[Scheduler] 发送响应失败，连接断开。" << std::endl;
             close(clientSocket);
             break;
        }
    }
    
    logStream << "[Scheduler] 线程 (Socket: " << clientSocket << ") 退出。" << std::endl;
    
    logStream.close();
}

int main() {
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

    std::cout << "[Scheduler] 进程已启动，正在 " << SCHEDULER_PORT << " 端口监听..." << std::endl;

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