#pragma once
#include <string>
#include <functional>
#include <memory>

// 代表一个已连接的客户端通道
class IChannel {
public:
    virtual ~IChannel() = default;

    // 阻塞接收消息 (返回 true 表示收到，false 表示超时或错误)
    // 实现层应处理忙等待/CPU pause
    virtual bool recvBlocking(std::string& outMsg) = 0;

    // 发送响应
    virtual bool sendBlocking(const std::string& msg) = 0;

    // 检查连接是否仍然存活
    virtual bool isConnected() = 0;

    // 标记调度器已准备好服务此通道 (握手用)
    virtual void setReady() = 0;

    // 获取元数据
    virtual std::string getId() const = 0;
    virtual std::string getType() const = 0;
    virtual std::string getName() const = 0;
};

// 代表 IPC 服务端/监听器
class IIPCServer {
public:
    virtual ~IIPCServer() = default;

    // 初始化
    virtual bool init() = 0;

    // 开始监听新客户端
    // 当有新客户端连接时，调用 callback
    virtual void start(std::function<void(std::unique_ptr<IChannel>)> onNewClient) = 0;

    // 停止服务
    virtual void stop() = 0;
};