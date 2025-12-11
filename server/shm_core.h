#pragma once

/**
 * @file shm_core.h
 * @brief 共享内存核心实现（服务端兼容层）
 * 
 * 此文件导入共享层的共享内存实现，为服务端代码提供向后兼容
 */

#include "ipc.h"
#include "config.h"
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <functional>

// ============================================================
//  ShmChannel - 服务端通道包装类
// ============================================================

class ShmChannel {
public:
    ShmChannel(std::unique_ptr<ipc::IChannel> channel)
        : channel_(std::move(channel)) {}
    
    ~ShmChannel() {
        if (channel_) {
            channel_->setServerReady(false);
        }
    }

    bool recvBlocking(std::string& outMsg) {
        // 使用超时轮询，以便能响应停止信号
        // 每 100ms 检查一次，如果队列为空则返回 false
        return channel_->getRequestQueue().receiveBlocking(outMsg, 100);
    }

    bool sendBlocking(const std::string& msg) {
        return channel_->getResponseQueue().sendBlocking(msg, 5000);
    }

    bool isConnected() {
        return channel_->isClientConnected();
    }

    void setReady() {
        channel_->setServerReady(true);
    }

    // 清理
    void unlink() {
        // 由工厂管理
    }
    
    std::string getId() const { return channel_->getUniqueId(); }
    std::string getType() const { return channel_->getClientType(); }
    std::string getName() const { return channel_->getName(); }

    ipc::IChannel* getChannel() { return channel_.get(); }

private:
    std::unique_ptr<ipc::IChannel> channel_;
};

// ============================================================
//  ShmServer - 服务端监听器
// ============================================================

class ShmServer {
public:
    ShmServer() : running_(false) {}
    
    ~ShmServer() {
        stop();
    }

    bool init() {
        listener_ = std::make_unique<ipc::shm::ShmServerListener>();
        return listener_->init();
    }

    void start(std::function<void(std::unique_ptr<ShmChannel>)> onNewClient) {
        userCallback_ = onNewClient;
        running_.store(true);
        
        listener_->start([this](std::unique_ptr<ipc::IChannel> channel) {
            // 包装成 ShmChannel 后交给用户回调
            auto shmChannel = std::make_unique<ShmChannel>(std::move(channel));
            if (userCallback_) {
                userCallback_(std::move(shmChannel));
            }
        });
    }

    void stop() {
        running_.store(false);
        if (listener_) {
            listener_->stop();
        }
    }

    bool isRunning() const { return running_.load(); }

private:
    std::atomic<bool> running_;
    std::unique_ptr<ipc::shm::ShmServerListener> listener_;
    std::function<void(std::unique_ptr<ShmChannel>)> userCallback_;
};
