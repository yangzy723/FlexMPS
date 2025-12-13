#pragma once

/**
 * @file config.h
 * @brief 配置和数据结构定义（服务端兼容层）
 * 
 * 此文件导入共享层的定义，为服务端代码提供向后兼容
 */

#include "ipc_common.h"

// ============================================================
//  类型别名（向后兼容旧代码）
// ============================================================

using SPSCQueue = ipc::SPSCQueueData;
using ClientChannelStruct = ipc::ChannelData;
using ClientRegistryEntry = ipc::RegistryEntryData;
using ClientRegistry = ipc::RegistryData;

// ============================================================
//  常量别名
// ============================================================

constexpr size_t SPSC_QUEUE_SIZE = ipc::SPSC_QUEUE_SIZE;
constexpr size_t SPSC_MSG_SIZE = ipc::MAX_MSG_SIZE;
constexpr size_t CACHE_LINE_SIZE = ipc::CACHE_LINE_SIZE;
constexpr size_t MAX_REGISTERED_CLIENTS = ipc::MAX_CLIENTS;

// 网络设置
#define SCHEDULER_PORT 9999
#define LOCALHOST "127.0.0.1"
