#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <chrono>

// ======================= SM4基础实现 =======================
// 定义循环左移函数
inline uint32_t rol(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// S盒定义
static const uint8_t SBOX[256] = {
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f, 0xd5,
    0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51, 0x8d,
    0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8, 0x0a,
    0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0, 0x89,
    0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84, 0x18,
    0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48
};

// 系统参数FK
static const uint32_t FK[4] = {
    0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc
};

// 固定参数CK
static const uint32_t CK[32] = {
    0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
    0x70777e85, 0x8c939aa1, 0xa8afb6bd, 0xc4cbd2d9,
    0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
    0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
    0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
    0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
    0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
    0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279
};

// τ变换：32位输入通过S盒替换
uint32_t tau(uint32_t x) {
    uint32_t res = 0;
    res |= static_cast<uint32_t>(SBOX[(x >> 24) & 0xff]) << 24;
    res |= static_cast<uint32_t>(SBOX[(x >> 16) & 0xff]) << 16;
    res |= static_cast<uint32_t>(SBOX[(x >> 8) & 0xff]) << 8;
    res |= static_cast<uint32_t>(SBOX[x & 0xff]);
    return res;
}

// 线性变换L（用于轮函数）
uint32_t L(uint32_t x) {
    return x ^ rol(x, 2) ^ rol(x, 10) ^ rol(x, 18) ^ rol(x, 24);
}

// 线性变换L'（用于密钥扩展）
uint32_t L_prime(uint32_t x) {
    return x ^ rol(x, 13) ^ rol(x, 23);
}

// 合成置换T（用于轮函数）
uint32_t T(uint32_t x) {
    return L(tau(x));
}

// 合成置换T'（用于密钥扩展）
uint32_t T_prime(uint32_t x) {
    return L_prime(tau(x));
}

// 轮函数F
uint32_t F(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t rk) {
    return x0 ^ T(x1 ^ x2 ^ x3 ^ rk);
}

// 反序变换
void reverse_words(uint32_t& x0, uint32_t& x1, uint32_t& x2, uint32_t& x3) {
    std::swap(x0, x3);
    std::swap(x1, x2);
}

// 密钥扩展函数
std::vector<uint32_t> expand_key(const uint8_t key[16]) {
    std::vector<uint32_t> rk(32);
    uint32_t k[36];

    // 将字节数组转换为字
    for (int i = 0; i < 4; ++i) {
        k[i] = (static_cast<uint32_t>(key[4 * i]) << 24) |
            (static_cast<uint32_t>(key[4 * i + 1]) << 16) |
            (static_cast<uint32_t>(key[4 * i + 2]) << 8) |
            static_cast<uint32_t>(key[4 * i + 3]);
    }

    // 初始化K0~K3
    for (int i = 0; i < 4; ++i) {
        k[i] ^= FK[i];
    }

    // 生成轮密钥
    for (int i = 0; i < 32; ++i) {
        k[i + 4] = k[i] ^ T_prime(k[i + 1] ^ k[i + 2] ^ k[i + 3] ^ CK[i]);
        rk[i] = k[i + 4];
    }

    return rk;
}

// 加密函数
void sm4_encrypt_block(const uint8_t in[16], uint8_t out[16], const std::vector<uint32_t>& rk) {
    uint32_t x[4];

    // 将输入字节转换为字
    for (int i = 0; i < 4; ++i) {
        x[i] = (static_cast<uint32_t>(in[4 * i]) << 24) |
            (static_cast<uint32_t>(in[4 * i + 1]) << 16) |
            (static_cast<uint32_t>(in[4 * i + 2]) << 8) |
            static_cast<uint32_t>(in[4 * i + 3]);
    }

    // 32轮迭代
    for (int i = 0; i < 32; ++i) {
        uint32_t tmp = F(x[0], x[1], x[2], x[3], rk[i]);
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = tmp;
    }

    // 反序变换
    reverse_words(x[0], x[1], x[2], x[3]);

    // 将输出字转换为字节
    for (int i = 0; i < 4; ++i) {
        out[4 * i] = static_cast<uint8_t>(x[i] >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(x[i] >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(x[i] >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(x[i]);
    }
}

// ======================= SM4-GCM基础实现（未优化） =======================
class SM4_GCM_Unoptimized {
public:
    SM4_GCM_Unoptimized(const uint8_t key[16]) {
        // 扩展密钥
        rk_ = expand_key(key);

        // 计算加密零块作为GHASH的H
        uint8_t zero[16] = { 0 };
        sm4_encrypt_block(zero, H_, rk_);
    }

    // 设置IV（初始化向量）
    void set_iv(const uint8_t* iv, size_t iv_len) {
        if (iv_len == 12) {
            // 最常用的12字节IV
            memcpy(J0_, iv, 12);
            J0_[12] = J0_[13] = J0_[14] = 0;
            J0_[15] = 0x01;
        }
        else {
            // 对于非12字节IV，使用GHASH计算J0
            compute_J0_from_iv(iv, iv_len);
        }
    }

    // 认证加密
    void encrypt(const uint8_t* plain, size_t len,
        uint8_t* cipher, uint8_t tag[16],
        const uint8_t* aad, size_t aad_len) {
        // 复制J0作为初始计数器
        uint8_t ctr[16];
        memcpy(ctr, J0_, 16);
        increment_counter(ctr);

        // 清空GHASH状态
        memset(ghash_state_, 0, 16);
        ghash_len_ = 0;

        // 处理附加认证数据
        if (aad_len > 0) {
            ghash_update(aad, aad_len, true);
        }

        // 加密数据
        gctr(plain, len, cipher, ctr);

        // 处理密文
        if (len > 0) {
            ghash_update(cipher, len, false);
        }

        // 处理长度信息
        uint64_t aad_bits = static_cast<uint64_t>(aad_len) * 8;
        uint64_t cipher_bits = static_cast<uint64_t>(len) * 8;

        uint8_t len_block[16];
        memset(len_block, 0, 16);
        for (int i = 0; i < 8; ++i) {
            len_block[7 - i] = static_cast<uint8_t>(aad_bits >> (i * 8));
            len_block[15 - i] = static_cast<uint8_t>(cipher_bits >> (i * 8));
        }
        ghash_block(len_block);

        // 计算认证标签
        uint8_t s_block[16];
        sm4_encrypt_block(J0_, s_block, rk_);
        for (int i = 0; i < 16; ++i) {
            tag[i] = s_block[i] ^ ghash_state_[i];
        }
    }

    // 认证解密
    bool decrypt(const uint8_t* cipher, size_t len,
        const uint8_t tag[16], uint8_t* plain,
        const uint8_t* aad, size_t aad_len) {
        // 复制J0作为初始计数器
        uint8_t ctr[16];
        memcpy(ctr, J0_, 16);
        increment_counter(ctr);

        // 清空GHASH状态
        memset(ghash_state_, 0, 16);
        ghash_len_ = 0;

        // 处理附加认证数据
        if (aad_len > 0) {
            ghash_update(aad, aad_len, true);
        }

        // 处理密文
        if (len > 0) {
            ghash_update(cipher, len, false);
        }

        // 处理长度信息
        uint64_t aad_bits = static_cast<uint64_t>(aad_len) * 8;
        uint64_t cipher_bits = static_cast<uint64_t>(len) * 8;

        uint8_t len_block[16];
        memset(len_block, 0, 16);
        for (int i = 0; i < 8; ++i) {
            len_block[7 - i] = static_cast<uint8_t>(aad_bits >> (i * 8));
            len_block[15 - i] = static_cast<uint8_t>(cipher_bits >> (i * 8));
        }
        ghash_block(len_block);

        // 验证标签
        uint8_t s_block[16];
        sm4_encrypt_block(J0_, s_block, rk_);
        for (int i = 0; i < 16; ++i) {
            if ((s_block[i] ^ ghash_state_[i]) != tag[i]) {
                return false;
            }
        }

        // 标签验证通过，解密数据
        if (len > 0) {
            gctr(cipher, len, plain, ctr);
        }
        return true;
    }

private:
    // GF(2^128)乘法（无优化版本）
    void gf128_mul_basic(uint8_t* x, const uint8_t* y) {
        uint8_t v[16];
        memcpy(v, y, 16);

        uint8_t z[16] = { 0 };

        for (int i = 0; i < 16; i++) {
            uint8_t byte = x[i];
            for (int j = 0; j < 8; j++) {
                if (byte & 0x80) {
                    for (int k = 0; k < 16; k++) {
                        z[k] ^= v[k];
                    }
                }

                // 记录v的最高位
                uint8_t carry = v[0] & 0x80;

                // 左移v
                for (int k = 0; k < 15; k++) {
                    v[k] = (v[k] << 1) | ((v[k + 1] & 0x80) >> 7);
                }
                v[15] = v[15] << 1;

                // 如果最高位是1，则异或0x87
                if (carry) {
                    v[15] ^= 0x87;
                }

                byte <<= 1;
            }
        }

        memcpy(x, z, 16);
    }

    // 处理一个GHASH块（无优化版本）
    void ghash_block(const uint8_t block[16]) {
        for (int i = 0; i < 16; i++) {
            ghash_state_[i] ^= block[i];
        }
        gf128_mul_basic(ghash_state_, H_);
    }

    // 更新GHASH状态
    void ghash_update(const uint8_t* data, size_t len, bool is_aad) {
        size_t block_offset = ghash_len_ % 16;
        size_t processed = 0;

        // 处理不完整的块
        if (block_offset != 0) {
            size_t to_copy = std::min(16 - block_offset, len);
            for (size_t i = 0; i < to_copy; i++) {
                ghash_buffer_[block_offset + i] = data[i];
            }

            if (block_offset + to_copy == 16) {
                ghash_block(ghash_buffer_);
            }

            processed += to_copy;
            data += to_copy;
            len -= to_copy;
            ghash_len_ += to_copy;
        }

        // 批量处理完整块
        while (len >= 16) {
            ghash_block(data);
            data += 16;
            len -= 16;
            processed += 16;
            ghash_len_ += 16;
        }

        // 存储余数
        if (len > 0) {
            memcpy(ghash_buffer_, data, len);
            memset(ghash_buffer_ + len, 0, 16 - len);
            ghash_len_ += len;
        }
    }

    // GCTR模式加密/解密
    void gctr(const uint8_t* in, size_t len, uint8_t* out, uint8_t counter[16]) {
        uint8_t block[16];

        while (len >= 16) {
            // 生成密钥流
            sm4_encrypt_block(counter, block, rk_);
            increment_counter(counter);

            // 异或加密
            for (int i = 0; i < 16; i++) {
                out[i] = in[i] ^ block[i];
            }

            in += 16;
            out += 16;
            len -= 16;
        }

        // 处理尾部
        if (len > 0) {
            sm4_encrypt_block(counter, block, rk_);
            increment_counter(counter);

            for (size_t i = 0; i < len; i++) {
                out[i] = in[i] ^ block[i];
            }
        }
    }

    // 计数器增量
    void increment_counter(uint8_t ctr[16]) {
        for (int i = 15; i >= 12; i--) {
            if (++ctr[i] != 0) {
                break;
            }
        }
    }

    // 计算J0（用于非12字节IV的情况）
    void compute_J0_from_iv(const uint8_t* iv, size_t iv_len) {
        // 清空GHASH状态
        memset(ghash_state_, 0, 16);
        ghash_len_ = 0;

        // 处理IV（无填充）
        if (iv_len > 0) {
            ghash_update(iv, iv_len, false);
        }

        // 处理长度信息
        uint64_t iv_bits = static_cast<uint64_t>(iv_len) * 8;
        uint8_t len_block[16];
        memset(len_block, 0, 16);
        for (int i = 0; i < 8; ++i) {
            len_block[15 - i] = static_cast<uint8_t>(iv_bits >> (i * 8));
        }
        ghash_block(len_block);

        // J0 = GHASH结果
        memcpy(J0_, ghash_state_, 16);
    }

    // 内部状态
    std::vector<uint32_t> rk_;      // 轮密钥
    uint8_t H_[16] = { 0 };           // GHASH的H值
    uint8_t J0_[16] = { 0 };          // 初始计数器
    uint8_t ghash_state_[16] = { 0 }; // GHASH当前状态
    size_t ghash_len_ = 0;          // GHASH已处理字节数
    uint8_t ghash_buffer_[16] = { 0 };// 未处理的部分块
};

// ======================= SM4-GCM优化实现 =======================
class SM4_GCM_Optimized {
public:
    SM4_GCM_Optimized(const uint8_t key[16]) {
        // 扩展密钥
        rk_ = expand_key(key);

        // 计算加密零块作为GHASH的H
        uint8_t zero[16] = { 0 };
        sm4_encrypt_block(zero, H_, rk_);

        // 预计算GHASH乘法表
        precompute_ghash_table();
    }

    // 设置IV（初始化向量）
    void set_iv(const uint8_t* iv, size_t iv_len) {
        if (iv_len == 12) {
            // 最常用的12字节IV
            memcpy(J0_, iv, 12);
            J0_[12] = J0_[13] = J0_[14] = 0;
            J0_[15] = 0x01;
        }
        else {
            // 对于非12字节IV，使用GHASH计算J0
            compute_J0_from_iv(iv, iv_len);
        }
    }

    // 认证加密
    void encrypt(const uint8_t* plain, size_t len,
        uint8_t* cipher, uint8_t tag[16],
        const uint8_t* aad, size_t aad_len) {
        // 复制J0作为初始计数器
        uint8_t ctr[16];
        memcpy(ctr, J0_, 16);
        increment_counter(ctr);

        // 清空GHASH状态
        memset(ghash_state_, 0, 16);
        ghash_len_ = 0;

        // 处理附加认证数据
        if (aad_len > 0) {
            ghash_update(aad, aad_len, true);
        }

        // 加密数据
        gctr(plain, len, cipher, ctr);

        // 处理密文
        if (len > 0) {
            ghash_update(cipher, len, false);
        }

        // 处理长度信息
        uint64_t aad_bits = static_cast<uint64_t>(aad_len) * 8;
        uint64_t cipher_bits = static_cast<uint64_t>(len) * 8;

        uint8_t len_block[16];
        memset(len_block, 0, 16);
        for (int i = 0; i < 8; ++i) {
            len_block[7 - i] = static_cast<uint8_t>(aad_bits >> (i * 8));
            len_block[15 - i] = static_cast<uint8_t>(cipher_bits >> (i * 8));
        }
        ghash_block(len_block);

        // 计算认证标签
        uint8_t s_block[16];
        sm4_encrypt_block(J0_, s_block, rk_);
        for (int i = 0; i < 16; ++i) {
            tag[i] = s_block[i] ^ ghash_state_[i];
        }
    }

    // 认证解密
    bool decrypt(const uint8_t* cipher, size_t len,
        const uint8_t tag[16], uint8_t* plain,
        const uint8_t* aad, size_t aad_len) {
        // 复制J0作为初始计数器
        uint8_t ctr[16];
        memcpy(ctr, J0_, 16);
        increment_counter(ctr);

        // 清空GHASH状态
        memset(ghash_state_, 0, 16);
        ghash_len_ = 0;

        // 处理附加认证数据
        if (aad_len > 0) {
            ghash_update(aad, aad_len, true);
        }

        // 处理密文
        if (len > 0) {
            ghash_update(cipher, len, false);
        }

        // 处理长度信息
        uint64_t aad_bits = static_cast<uint64_t>(aad_len) * 8;
        uint64_t cipher_bits = static_cast<uint64_t>(len) * 8;

        uint8_t len_block[16];
        memset(len_block, 0, 16);
        for (int i = 0; i < 8; ++i) {
            len_block[7 - i] = static_cast<uint8_t>(aad_bits >> (i * 8));
            len_block[15 - i] = static_cast<uint8_t>(cipher_bits >> (i * 8));
        }
        ghash_block(len_block);

        // 验证标签
        uint8_t s_block[16];
        sm4_encrypt_block(J0_, s_block, rk_);
        for (int i = 0; i < 16; ++i) {
            if ((s_block[i] ^ ghash_state_[i]) != tag[i]) {
                return false;
            }
        }

        // 标签验证通过，解密数据
        if (len > 0) {
            gctr(cipher, len, plain, ctr);
        }
        return true;
    }

private:
    // 预计算GHASH乘法表（8位表）
    void precompute_ghash_table() {
        memset(ghash_table_, 0, 256 * 16);

        // 基本表项：T[0] = 0, T[1] = H
        ghash_table_[0] = 0;
        for (int i = 0; i < 16; i++) {
            ghash_table_[1 * 16 + i] = H_[i];
        }

        // 计算所有表项：T[i] = (i*H) mod P
        for (int i = 2; i < 256; i++) {
            if (i % 2 == 0) {
                // 偶数：T[i] = 2 * T[i/2]
                const uint8_t* prev = &ghash_table_[(i / 2) * 16];
                uint8_t* current = &ghash_table_[i * 16];
                uint8_t carry = prev[0] & 0x80;
                for (int j = 0; j < 15; j++) {
                    current[j] = (prev[j] << 1) | ((prev[j + 1] & 0x80) >> 7);
                }
                current[15] = (prev[15] << 1);
                if (carry) {
                    current[15] ^= 0x87;
                }
            }
            else {
                // 奇数：T[i] = T[i-1] XOR H
                const uint8_t* prev = &ghash_table_[(i - 1) * 16];
                uint8_t* current = &ghash_table_[i * 16];
                for (int j = 0; j < 16; j++) {
                    current[j] = prev[j] ^ H_[j];
                }
            }
        }
    }

    // 处理一个GHASH块（优化版本）
    void ghash_block(const uint8_t block[16]) {
        // 状态块与输入块异或
        for (int i = 0; i < 16; i++) {
            ghash_state_[i] ^= block[i];
        }

        // 使用查表法计算GHASH乘法
        uint8_t result[16] = { 0 };

        // 一次性处理整个块（16字节）
        for (int i = 0; i < 16; i++) {
            const uint8_t* mul_table = &ghash_table_[ghash_state_[i] * 16];
            for (int j = 0; j < 16; j++) {
                result[j] ^= mul_table[j];
            }
        }

        memcpy(ghash_state_, result, 16);
    }

    // 更新GHASH状态
    void ghash_update(const uint8_t* data, size_t len, bool is_aad) {
        size_t block_offset = ghash_len_ % 16;
        size_t processed = 0;

        // 处理不完整的块
        if (block_offset != 0) {
            size_t to_copy = std::min(16 - block_offset, len);
            for (size_t i = 0; i < to_copy; i++) {
                ghash_buffer_[block_offset + i] = data[i];
            }

            if (block_offset + to_copy == 16) {
                ghash_block(ghash_buffer_);
            }

            processed += to_copy;
            data += to_copy;
            len -= to_copy;
            ghash_len_ += to_copy;
        }

        // 批量处理完整块
        for (; len >= 16; len -= 16) {
            ghash_block(data);
            data += 16;
            processed += 16;
            ghash_len_ += 16;
        }

        // 存储余数
        if (len > 0) {
            memcpy(ghash_buffer_, data, len);
            memset(ghash_buffer_ + len, 0, 16 - len);
            ghash_len_ += len;
        }
    }

    // GCTR模式加密/解密
    void gctr(const uint8_t* in, size_t len, uint8_t* out, uint8_t counter[16]) {
        uint8_t block[16];

        // 使用块填充计数器
        while (len >= 16) {
            // 生成密钥流
            sm4_encrypt_block(counter, block, rk_);
            increment_counter(counter);

            // 异或加密
            for (int i = 0; i < 16; i++) {
                out[i] = in[i] ^ block[i];
            }

            in += 16;
            out += 16;
            len -= 16;
        }

        // 处理尾部
        if (len > 0) {
            sm4_encrypt_block(counter, block, rk_);
            increment_counter(counter);

            for (size_t i = 0; i < len; i++) {
                out[i] = in[i] ^ block[i];
            }
        }
    }

    // 计数器增量
    void increment_counter(uint8_t ctr[16]) {
        // 只递增最后4字节（优化：32位计数器）
        if (++ctr[15] == 0) {
            if (++ctr[14] == 0) {
                if (++ctr[13] == 0) {
                    ++ctr[12];
                }
            }
        }
    }

    // 计算J0（用于非12字节IV的情况）
    void compute_J0_from_iv(const uint8_t* iv, size_t iv_len) {
        // 清空GHASH状态
        memset(ghash_state_, 0, 16);
        ghash_len_ = 0;

        // 处理IV（无填充）
        if (iv_len > 0) {
            ghash_update(iv, iv_len, false);
        }

        // 处理长度信息
        uint64_t iv_bits = static_cast<uint64_t>(iv_len) * 8;
        uint8_t len_block[16];
        memset(len_block, 0, 16);
        for (int i = 0; i < 8; ++i) {
            len_block[15 - i] = static_cast<uint8_t>(iv_bits >> (i * 8));
        }
        ghash_block(len_block);

        // J0 = GHASH结果
        memcpy(J0_, ghash_state_, 16);
    }

    // 内部状态
    std::vector<uint32_t> rk_;      // 轮密钥
    uint8_t H_[16] = { 0 };           // GHASH的H值
    uint8_t J0_[16] = { 0 };          // 初始计数器
    uint8_t ghash_state_[16] = { 0 }; // GHASH当前状态
    size_t ghash_len_ = 0;          // GHASH已处理字节数
    uint8_t ghash_buffer_[16] = { 0 };// 未处理的部分块

    // GHASH预计算表（256个表项 * 16字节 = 4096字节）
    uint8_t ghash_table_[256 * 16];
};

// 辅助函数：打印十六进制数据
void print_hex(const char* label, const uint8_t* data, size_t len) {
    std::cout << label << ": ";
    for (size_t i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(data[i]);
    }
    std::cout << std::dec << std::endl;
}

// 生成测试数据
void generate_test_data(uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<uint8_t>(i);
    }
}

// 测试函数
int main() {
    // 测试密钥
    uint8_t key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    // 测试IV（12字节）
    uint8_t iv[12] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b
    };

    // 测试AAD
    uint8_t aad[64];
    generate_test_data(aad, sizeof(aad));

    // 测试数据大小（1MB）
    const size_t data_size = 1 * 1024 * 1024;
    uint8_t* plaintext = new uint8_t[data_size];
    uint8_t* ciphertext = new uint8_t[data_size];
    uint8_t* decrypted = new uint8_t[data_size];
    uint8_t tag[16];

    // 生成测试数据
    generate_test_data(plaintext, data_size);

    // 实例化未优化和优化版本
    SM4_GCM_Unoptimized unoptimized(key);
    SM4_GCM_Optimized optimized(key);

    // 设置初始化向量
    unoptimized.set_iv(iv, 12);
    optimized.set_iv(iv, 12);

    // 性能测试
    using namespace std::chrono;

    // ========= 未优化版本测试 =========
    auto start_unopt = high_resolution_clock::now();
    unoptimized.encrypt(plaintext, data_size, ciphertext, tag, aad, sizeof(aad));
    auto end_unopt = high_resolution_clock::now();

    auto start_unopt_dec = high_resolution_clock::now();
    bool success_unopt = unoptimized.decrypt(ciphertext, data_size, tag, decrypted, aad, sizeof(aad));
    auto end_unopt_dec = high_resolution_clock::now();

    auto unopt_enc_time = duration_cast<microseconds>(end_unopt - start_unopt).count();
    auto unopt_dec_time = duration_cast<microseconds>(end_unopt_dec - start_unopt_dec).count();

    // ========= 优化版本测试 =========
    auto start_opt = high_resolution_clock::now();
    optimized.encrypt(plaintext, data_size, ciphertext, tag, aad, sizeof(aad));
    auto end_opt = high_resolution_clock::now();

    auto start_opt_dec = high_resolution_clock::now();
    bool success_opt = optimized.decrypt(ciphertext, data_size, tag, decrypted, aad, sizeof(aad));
    auto end_opt_dec = high_resolution_clock::now();

    auto opt_enc_time = duration_cast<microseconds>(end_opt - start_opt).count();
    auto opt_dec_time = duration_cast<microseconds>(end_opt_dec - start_opt_dec).count();

    // 验证解密结果
    bool match = memcmp(plaintext, decrypted, data_size) == 0;

    // 打印结果
    std::cout << "===== SM4-GCM 性能测试 =====" << std::endl;
    std::cout << "数据大小: " << data_size / 1024 << " KB" << std::endl;
    std::cout << "AAD大小: " << sizeof(aad) << " 字节" << std::endl;
    std::cout << "解密验证: " << (match ? "成功" : "失败") << std::endl;
    std::cout << std::endl;

    // 性能对比表
    std::cout << "          | 加密时间(us) | 解密时间(us) | 速度提升" << std::endl;
    std::cout << "----------|--------------|--------------|-----------" << std::endl;
    printf("未优化版本 | %12lld | %12lld | 1.00x\n", unopt_enc_time, unopt_dec_time);
    printf("优化版本   | %12lld | %12lld | %.2fx\n",
        opt_enc_time, opt_dec_time,
        static_cast<double>(unopt_enc_time) / opt_enc_time);

    std::cout << std::endl;

    // 详细时间统计
    std::cout << "===== 详细时间统计 =====" << std::endl;
    std::cout << "未优化版本加密: " << unopt_enc_time << " μs ("
        << data_size / (unopt_enc_time / 1000.0) << " KB/s)" << std::endl;
    std::cout << "未优化版本解密: " << unopt_dec_time << " μs ("
        << data_size / (unopt_dec_time / 1000.0) << " KB/s)" << std::endl;
    std::cout << "优化版本加密  : " << opt_enc_time << " μs ("
        << data_size / (opt_enc_time / 1000.0) << " KB/s)" << std::endl;
    std::cout << "优化版本解密  : " << opt_dec_time << " μs ("
        << data_size / (opt_dec_time / 1000.0) << " KB/s)" << std::endl;
    std::cout << "GHASH加速比   : "
        << static_cast<double>(unopt_enc_time - unopt_dec_time) / (opt_enc_time - opt_dec_time)
        << "x" << std::endl;

    // 清理
    delete[] plaintext;
    delete[] ciphertext;
    delete[] decrypted;

    return 0;
}