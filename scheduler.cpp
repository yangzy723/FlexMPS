#include "IPCProtocol.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

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
    std::cout << "\n[Scheduler] 收到一个新的连接。开始服务该连接 (Socket: " << clientSocket << ")" << std::endl;

    char buffer[1024] = {0};
    
    while (true) {
        memset(buffer, 0, 1024);
        ssize_t bytesRead = read(clientSocket, buffer, 1023);

        if (bytesRead <= 0) {
            if (bytesRead == 0) {
                std::cout << "[Scheduler] 客户端 (Socket " << clientSocket << ") 已优雅断开连接。" << std::endl;
            } else {
                std::cerr << "[Scheduler] Socket " << clientSocket << " 读取错误，断开连接。" << std::endl;
            }
            close(clientSocket);
            break;              
        }

        std::string message(buffer, bytesRead);
        message.erase(message.find_last_not_of("\r\n") + 1); 

        std::cout << "[Scheduler] 收到请求: " << message << std::endl;

        auto parts = split(message, '|');
        if (parts.size() != 2) {
            std::cerr << "[Scheduler] 请求格式错误: " << message << "，断开连接。" << std::endl;
            close(clientSocket);
            break;
        }
        
        std::string reqId = parts[0];
        std::string kernelType = parts[1];

        auto decision = makeDecision(kernelType);
        bool allowed = decision.first;
        std::string reason = decision.second;

        std::cout << "[Scheduler] 决策 (ID: " << reqId << "): " 
                  << (allowed ? "允许" : "拒绝") << std::endl;

        std::string response = createResponseMessage(reqId, allowed, reason);
        
        if (send(clientSocket, response.c_str(), response.length(), 0) < 0) {
             std::cerr << "[Scheduler] 发送响应失败，连接断开。" << std::endl;
             close(clientSocket);
             break;
        }
    }
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

    if (listen(server_fd, 3) < 0) {
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
        serviceClient(new_socket); 
    }

    return 0;
}