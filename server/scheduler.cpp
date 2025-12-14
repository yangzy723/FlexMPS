#include "logger.h"
#include "scheduler.h"

#include <sstream>
#include <iostream>

Scheduler::Scheduler() {}

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::stop() {
    running = false;
    std::lock_guard<std::mutex> lock(threadsMutex);
    for (auto& t : workers) {
        if (t.joinable())
            t.join();
    }
    workers.clear();
}

size_t Scheduler::getActiveCount() {
    std::lock_guard<std::mutex> lock(threadsMutex);
    return workers.size(); 
}

std::pair<bool, std::string> Scheduler::makeDecision(const std::string& kernelType) {
    // 核心调度算法
    return {true, "OK"};
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

void Scheduler::onNewClient(std::unique_ptr<IChannel> channel) {
    LogManager::instance().sessionIdIncrement();
    // 转移 channel 所有权给线程
    std::lock_guard<std::mutex> lock(threadsMutex);
    workers.emplace_back(&Scheduler::clientHandler, this, std::move(channel));
}

void Scheduler::clientHandler(std::unique_ptr<IChannel> channel) {
    std::stringstream ss;
    long long sessionId = LogManager::instance().getSessionId();
    std::string clientKey = channel->getType() + ":" + channel->getId();
    ss << "[Scheduler] Session #" << sessionId << " started for " 
       << clientKey << " (SHM: " << channel->getName() << ")";
    std::cout << ss.str() << std::endl;

    channel->setReady();

    std::string message;
    std::string the_unique_id;
    while (running && channel->isConnected()) {
        // 阻塞接收 (底层实现忙等待)
        if (!channel->recvBlocking(message)) {
             continue; 
        }

        // 简单的协议解析
        while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
            message.pop_back();
        }

        auto parts = split(message, '|');
        if (parts.size() < 3) {
            continue;
        }

        std::string kernelType = parts[0];     
        std::string reqId = parts[1];  
        std::string client_id = parts[2];
        std::string unique_id = parts.size() >=4 ? parts[3] : client_id;
        if (the_unique_id.empty()) {
            the_unique_id = unique_id;
        }

        LogManager::instance().getLogger(unique_id)->kernelIdIncrement();
        long long kernelId = LogManager::instance().getLogger(unique_id)->getKernelId();
        LogManager::instance().getLogger(unique_id)->recordKernelStat(kernelType);

        ss.str("");
        ss << "Kernel " << kernelId << ": " << kernelType << " from " << client_id;
        LogManager::instance().getLogger(unique_id)->write(ss.str());

        // 决策
        auto decision = makeDecision(kernelType);
        
        // 构建响应
        std::string response = reqId + "|" + (decision.first ? "1" : "0") + "|" + decision.second + "\n";
        
        if (!channel->sendBlocking(response)) {
            LogManager::instance().getLogger(unique_id)->write("[Scheduler] Send timeout for " + clientKey);
        }
    }
    LogManager::instance().removeLogger(the_unique_id);
    ss.str("");
    ss << "[Scheduler] Session #" << sessionId << " ended (" << clientKey << ")";
    std::cout << ss.str() << std::endl;
}