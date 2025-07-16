#include <iostream>
#include <cstring>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

// 定义基础类型
using uint8 = unsigned char;
using uint32 = unsigned int;
using uint64 = unsigned long long;

// SM3基础实现（已添加状态和长度设置方法）
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

    // 设置内部状态（用于长度扩展攻击）
    void set_state(const uint32 new_state[8], uint64 new_total_len) {
        memcpy(state, new_state, 8 * sizeof(uint32));
        total_length = new_total_len;
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

        if (buffer_length > 56) {
            memset(buffer + buffer_length, 0, 64 - buffer_length);
            compress(buffer);
            buffer_length = 0;
        }

        memset(buffer + buffer_length, 0, 56 - buffer_length);
        for (int i = 0; i < 8; i++) {
            buffer[56 + i] = (bit_length >> (56 - 8 * i)) & 0xFF;
        }

        compress(buffer);

        for (int i = 0; i < 8; i++) {
            digest[4 * i] = (state[i] >> 24) & 0xFF;
            digest[4 * i + 1] = (state[i] >> 16) & 0xFF;
            digest[4 * i + 2] = (state[i] >> 8) & 0xFF;
            digest[4 * i + 3] = state[i] & 0xFF;
        }
    }

    static void digest_to_state(const uint8 digest[32], uint32 state[8]) {
        for (int i = 0; i < 8; i++) {
            state[i] = (digest[4 * i] << 24) |
                (digest[4 * i + 1] << 16) |
                (digest[4 * i + 2] << 8) |
                digest[4 * i + 3];
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

// 工具函数：将哈希转换为十六进制字符串
std::string hash_to_hex(const uint8 digest[32]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) {
        oss << std::setw(2) << static_cast<int>(digest[i]);
    }
    return oss.str();
}

// 计算原始消息的填充
std::vector<uint8> calculate_padding(uint64 original_len_bytes) {
    uint64 original_len_bits = original_len_bytes * 8;
    size_t padding_len = 64 - (original_len_bytes % 64);
    if (padding_len < 9) padding_len += 64; // 需要额外一个块
    padding_len = padding_len > 64 ? padding_len - 64 : padding_len;

    std::vector<uint8> padding(padding_len, 0);
    padding[0] = 0x80;

    // 最后8字节存储原始消息的比特长度（大端序）
    for (int i = 0; i < 8; i++) {
        padding[padding_len - 8 + i] = (original_len_bits >> (56 - 8 * i)) & 0xFF;
    }

    return padding;
}

// 执行长度扩展攻击
void length_extension_attack() {
    // ========== 第1部分：计算原始消息的哈希 ==========
    std::string secret = "secret";
    std::string public_data = "data";

    // 计算原始消息的哈希
    SM3_Base sm3_original;
    sm3_original.update(reinterpret_cast<const uint8*>(secret.data()), secret.size());
    uint8 orig_digest[32];
    sm3_original.finalize(orig_digest);

    std::cout << "原始消息: '" << secret << "'\n";
    std::cout << "原始哈希: " << hash_to_hex(orig_digest) << "\n";

    // ========== 第2部分：正常计算扩展后的哈希 ==========
    std::vector<uint8> full_message;
    full_message.insert(full_message.end(), secret.begin(), secret.end());

    // 添加原始消息的填充
    auto padding = calculate_padding(secret.size());
    full_message.insert(full_message.end(), padding.begin(), padding.end());

    // 添加扩展数据
    full_message.insert(full_message.end(), public_data.begin(), public_data.end());

    // 计算正确的新哈希
    SM3_Base sm3_correct;
    sm3_correct.update(full_message.data(), full_message.size());
    uint8 correct_digest[32];
    sm3_correct.finalize(correct_digest);

    std::cout << "\n正常计算扩展后消息的哈希:\n";
    std::cout << "消息长度: " << full_message.size() << " 字节\n";
    std::cout << "哈希值: " << hash_to_hex(correct_digest) << "\n";

    // ========== 第3部分：执行长度扩展攻击 ==========
    SM3_Base sm3_attack;

    // 设置攻击状态
    uint32 new_state[8];
    SM3_Base::digest_to_state(orig_digest, new_state);

    // 计算原始消息+填充的总长度（字节）
    uint64 total_len = secret.size() + padding.size();
    sm3_attack.set_state(new_state, total_len);

    // 附加攻击数据
    sm3_attack.update(reinterpret_cast<const uint8*>(public_data.data()), public_data.size());

    // 获取攻击结果
    uint8 attack_digest[32];
    sm3_attack.finalize(attack_digest);

    std::cout << "\n长度扩展攻击结果:\n";
    std::cout << "哈希值: " << hash_to_hex(attack_digest) << "\n";

    // ========== 验证攻击结果 ==========
    bool success = memcmp(correct_digest, attack_digest, 32) == 0;
    std::cout << "\n攻击结果: " << (success ? "成功" : "失败") << "\n";
    if (success) {
        std::cout << "长度扩展攻击验证成功!\n";
    }
    else {
        std::cout << "攻击结果与预期不一致!\n";
    }
}

int main() {
    std::cout << "=== SM3长度扩展攻击验证 ===\n";
    length_extension_attack();
    return 0;
}