#include <iostream>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>

// 定义基础类型
using uint8 = unsigned char;
using uint32 = unsigned int;
using uint64 = unsigned long long;

// 基础版本的SM3实现
class SM3_Base {
public:
    SM3_Base() { reset(); }

    void reset() {
        state[0] = 0x7380166F;
        state[1] = 0x4914B2B9;
        state[2] = 0x172442D7;
        state[3] = 0xDA8A0600;
        state[4] = 0xA96F30BC;
        state[5] = 0x163138AA;
        state[6] = 0xE38DEE4D;
        state[7] = 0xB0FB0E4E;
        memset(buffer, 0, 64);
        total_length = 0;
        buffer_length = 0;
    }

    void update(const uint8* data, size_t len) {
        total_length += len;
        while (len > 0) {
            size_t copy_size = std::min(len, static_cast<size_t>(64 - buffer_length));
            memcpy(buffer + buffer_length, data, copy_size);
            buffer_length += copy_size;
            data += copy_size;
            len -= copy_size;

            if (buffer_length == 64) {
                compress(buffer);
                buffer_length = 0;
            }
        }
    }

    void finalize(uint8 digest[32]) {
        uint64 bit_length = total_length * 8;
        buffer[buffer_length++] = 0x80;

        // 处理填充不足的情况
        if (buffer_length > 56) {
            memset(buffer + buffer_length, 0, 64 - buffer_length);
            compress(buffer);
            buffer_length = 0;
        }

        // 填充0直到长度56字节（448位）
        memset(buffer + buffer_length, 0, 56 - buffer_length);

        // 以大端序存储消息长度（高位在前）
        for (int i = 0; i < 8; i++) {
            buffer[56 + i] = (bit_length >> (56 - 8 * i)) & 0xFF;
        }

        compress(buffer);

        // 以大端序输出哈希值
        for (int i = 0; i < 8; i++) {
            digest[4 * i] = (state[i] >> 24) & 0xFF;
            digest[4 * i + 1] = (state[i] >> 16) & 0xFF;
            digest[4 * i + 2] = (state[i] >> 8) & 0xFF;
            digest[4 * i + 3] = state[i] & 0xFF;
        }
    }

protected:
    uint32 state[8];
    uint8 buffer[64];
    uint64 total_length;
    size_t buffer_length;

    static uint32 rol(uint32 x, int n) {
        return (x << n) | (x >> (32 - n));
    }

    static uint32 P0(uint32 x) {
        return x ^ rol(x, 9) ^ rol(x, 17);
    }

    static uint32 P1(uint32 x) {
        return x ^ rol(x, 15) ^ rol(x, 23);
    }

    static uint32 FF0(uint32 x, uint32 y, uint32 z) {
        return x ^ y ^ z;
    }

    static uint32 FF1(uint32 x, uint32 y, uint32 z) {
        return (x & y) | (x & z) | (y & z);
    }

    static uint32 GG0(uint32 x, uint32 y, uint32 z) {
        return x ^ y ^ z;
    }

    static uint32 GG1(uint32 x, uint32 y, uint32 z) {
        return (x & y) | (~x & z);
    }

    virtual void compress(const uint8 block[64]) {
        uint32 W[68];
        uint32 W1[64];
        uint32 A = state[0];
        uint32 B = state[1];
        uint32 C = state[2];
        uint32 D = state[3];
        uint32 E = state[4];
        uint32 F = state[5];
        uint32 G = state[6];
        uint32 H = state[7];

        // 消息扩展
        for (int i = 0; i < 16; i++) {
            W[i] = (block[4 * i] << 24) |
                (block[4 * i + 1] << 16) |
                (block[4 * i + 2] << 8) |
                block[4 * i + 3];
        }

        for (int i = 16; i < 68; i++) {
            W[i] = P1(W[i - 16] ^ W[i - 9] ^ rol(W[i - 3], 15)) ^ rol(W[i - 13], 7) ^ W[i - 6];
        }

        for (int i = 0; i < 64; i++) {
            W1[i] = W[i] ^ W[i + 4];
        }

        // 压缩函数
        for (int j = 0; j < 64; j++) {
            const uint32 T_val = (j < 16) ? 0x79CC4519 : 0x7A879D8A;
            const uint32 T_shift = rol(T_val, j % 32);
            const uint32 SS1 = rol(rol(A, 12) + E + T_shift, 7);
            const uint32 SS2 = SS1 ^ rol(A, 12);
            uint32 TT1, TT2;

            if (j < 16) {
                TT1 = FF0(A, B, C) + D + SS2 + W1[j];
                TT2 = GG0(E, F, G) + H + SS1 + W[j];
            }
            else {
                TT1 = FF1(A, B, C) + D + SS2 + W1[j];
                TT2 = GG1(E, F, G) + H + SS1 + W[j];
            }

            D = C;
            C = rol(B, 9);
            B = A;
            A = TT1;
            H = G;
            G = rol(F, 19);
            F = E;
            E = P0(TT2);
        }

        // 更新状态
        state[0] ^= A;
        state[1] ^= B;
        state[2] ^= C;
        state[3] ^= D;
        state[4] ^= E;
        state[5] ^= F;
        state[6] ^= G;
        state[7] ^= H;
    }
};

// 优化版本的SM3实现
// 优化版本的SM3实现（修正版）
class SM3_Optimized : public SM3_Base {
protected:
    // 重写compress函数以应用优化
    void compress(const uint8 block[64]) override {
        uint32 A = state[0];
        uint32 B = state[1];
        uint32 C = state[2];
        uint32 D = state[3];
        uint32 E = state[4];
        uint32 F = state[5];
        uint32 G = state[6];
        uint32 H = state[7];

        // 消息扩展
        uint32 W[68];
        for (int i = 0; i < 16; i++) {
            const uint8* b = block + 4 * i;
            W[i] = (static_cast<uint32>(b[0]) << 24) |
                (static_cast<uint32>(b[1]) << 16) |
                (static_cast<uint32>(b[2]) << 8) |
                static_cast<uint32>(b[3]);
        }

        // 修正消息扩展计算
        for (int i = 16; i < 68; i++) {
            W[i] = P1(W[i - 16] ^ W[i - 9] ^ rol(W[i - 3], 15)) ^ rol(W[i - 13], 7) ^ W[i - 6];
        }

        // 计算W1数组（W[i] ^ W[i+4]）
        uint32 W1[64];
        for (int i = 0; i < 64; i++) {
            W1[i] = W[i] ^ W[i + 4];
        }

        // 压缩函数
        for (int j = 0; j < 64; j++) {
            const uint32 T_val = (j < 16) ? 0x79CC4519 : 0x7A879D8A;
            const uint32 T_shift = rol(T_val, j % 32);
            const uint32 SS1 = rol(rol(A, 12) + E + T_shift, 7);
            const uint32 SS2 = SS1 ^ rol(A, 12);
            uint32 TT1, TT2;

            if (j < 16) {
                TT1 = FF0(A, B, C) + D + SS2 + W1[j];
                TT2 = GG0(E, F, G) + H + SS1 + W[j];
            }
            else {
                TT1 = FF1(A, B, C) + D + SS2 + W1[j];
                TT2 = GG1(E, F, G) + H + SS1 + W[j];
            }

            // 更新工作变量
            D = C;
            C = rol(B, 9);
            B = A;
            A = TT1;
            H = G;
            G = rol(F, 19);
            F = E;
            E = P0(TT2);
        }

        // 更新状态
        state[0] ^= A;
        state[1] ^= B;
        state[2] ^= C;
        state[3] ^= D;
        state[4] ^= E;
        state[5] ^= F;
        state[6] ^= G;
        state[7] ^= H;
    }
};

// 工具函数：将哈希转换为十六进制字符串
std::string hash_to_hex(const uint8 digest[32]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) {
        oss << std::setw(2) << static_cast<int>(digest[i]);
    }
    return oss.str();
}

// 工具函数：验证哈希值是否匹配
bool verify_hash(const uint8 digest[32], const std::string& expected) {
    std::string actual = hash_to_hex(digest);
    return actual == expected;
}

// 测试函数模板
template<typename SM3Type>
void test_sm3(const std::string& name) {
    // 测试用例1: "abc" (标准测试向量)
    {
        SM3Type sm3;
        uint8 digest[32];
        const std::string abc_expected = "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0";

        auto start = std::chrono::high_resolution_clock::now();
        sm3.update(reinterpret_cast<const uint8*>("abc"), 3);
        sm3.finalize(digest);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::string actual_abc = hash_to_hex(digest);
        bool abc_match = verify_hash(digest, abc_expected);

        std::cout << name << " - \"abc\"测试: " << std::endl;
        std::cout << "  耗时: " << duration.count() << " 微秒" << std::endl;
        std::cout << "  匹配: " << (abc_match ? "成功" : "失败") << std::endl;
    }

    // 测试用例2: 长消息 (1MB数据，性能测试)
    {
        SM3Type sm3;
        uint8 digest[32];
        std::vector<uint8> data(1024 * 1024, 'a'); // 1MB的'a'

        auto start = std::chrono::high_resolution_clock::now();
        sm3.update(data.data(), data.size());
        sm3.finalize(digest);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double speed = (1024 * 1024) / static_cast<double>(duration.count()) * 1000; // MB/s
        std::string long_hash = hash_to_hex(digest);

        std::cout << name << " - 1MB数据测试: " << std::endl;
        std::cout << "  耗时: " << duration.count() << " 微秒" << std::endl;
        std::cout << "  速度: " << std::fixed << std::setprecision(2) << speed << " MB/s" << std::endl;
    }
}

// 主函数
int main() {
    std::cout << "=== SM3哈希算法基准测试 ===" << std::endl;

    // 测试基础版本
    std::cout << "\n[基础版本]" << std::endl;
    test_sm3<SM3_Base>("SM3_Base");

    // 测试优化版本
    std::cout << "\n[优化版本]" << std::endl;
    test_sm3<SM3_Optimized>("SM3_Optimized");

    return 0;
}