#pragma once

/**
 * @file ipc.h
 * @brief IPC 接口（服务端兼容层）
 * 
 * 此文件导入共享层的定义，为服务端代码提供向后兼容
 */

// 引入 IPC 核心模块
#include "ipc_common.h"
#include "ipc_interface.h"
#include "shm_transport.h"

// ============================================================
//  服务端特定的简化接口（可选，向后兼容）
// ============================================================

/**
 * IChannel 的简化包装器
 * 为原有服务端代码提供简化接口
 */
class ServerChannel {
public:
    explicit ServerChannel(std::unique_ptr<ipc::IChannel> channel)
        : channel_(std::move(channel)) {}
    
    virtual ~ServerChannel() = default;

    // 阻塞接收消息
    bool recvBlocking(std::string& outMsg) {
        return channel_->getRequestQueue().receiveBlocking(outMsg, -1);
    }

    // 发送响应
    bool sendBlocking(const std::string& msg) {
        return channel_->getResponseQueue().sendBlocking(msg, 5000);
    }

    // 检查连接
    bool isConnected() {
        return channel_->isClientConnected();
    }

    // 标记就绪
    void setReady() {
        channel_->setServerReady(true);
    }

    // 获取元数据
    std::string getId() const { return channel_->getUniqueId(); }
    std::string getType() const { return channel_->getClientType(); }
    std::string getName() const { return channel_->getName(); }

    // 获取底层通道
    ipc::IChannel* getChannel() { return channel_.get(); }

private:
    std::unique_ptr<ipc::IChannel> channel_;
};

// 类型别名（向后兼容）
using IIPCServer = ipc::IServerListener;
