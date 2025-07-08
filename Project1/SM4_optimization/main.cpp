#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <immintrin.h>

// 定义循环左移函数
inline uint32_t rol(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// SSE/AVX专用的32位循环左移函数
inline __m128i _mm_rool_epi32(__m128i x, int n) {
    return _mm_or_si128(_mm_slli_epi32(x, n), _mm_srli_epi32(x, 32 - n));
}

// S盒定义（固定值）
static const uint8_t SBOX[256] = {
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x0a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48
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

// 组合查表法优化所需表
uint32_t T0[256], T1[256], T2[256], T3[256];
uint32_t T_prime0[256], T_prime1[256], T_prime2[256], T_prime3[256];

// 线性变换L（用于轮函数）
inline uint32_t L(uint32_t x) {
    return x ^ rol(x, 2) ^ rol(x, 10) ^ rol(x, 18) ^ rol(x, 24);
}

// 线性变换L'（用于密钥扩展）
inline uint32_t L_prime(uint32_t x) {
    return x ^ rol(x, 13) ^ rol(x, 23);
}

// τ变换：32位输入通过S盒替换
inline uint32_t tau(uint32_t x) {
    uint32_t res = 0;
    res |= static_cast<uint32_t>(SBOX[(x >> 24) & 0xff]) << 24;
    res |= static_cast<uint32_t>(SBOX[(x >> 16) & 0xff]) << 16;
    res |= static_cast<uint32_t>(SBOX[(x >> 8) & 0xff]) << 8;
    res |= static_cast<uint32_t>(SBOX[x & 0xff]);
    return res;
}

// 合成置换T（用于轮函数）
inline uint32_t T(uint32_t x) {
    return L(tau(x));
}

// 合成置换T'（用于密钥扩展）
inline uint32_t T_prime(uint32_t x) {
    return L_prime(tau(x));
}

// 初始化组合查表
void init_tables() {
    for (int i = 0; i < 256; i++) {
        uint32_t y0 = static_cast<uint32_t>(SBOX[i]) << 24;
        uint32_t y1 = static_cast<uint32_t>(SBOX[i]) << 16;
        uint32_t y2 = static_cast<uint32_t>(SBOX[i]) << 8;
        uint32_t y3 = static_cast<uint32_t>(SBOX[i]);

        T0[i] = L(y0);
        T1[i] = L(y1);
        T2[i] = L(y2);
        T3[i] = L(y3);

        T_prime0[i] = L_prime(y0);
        T_prime1[i] = L_prime(y1);
        T_prime2[i] = L_prime(y2);
        T_prime3[i] = L_prime(y3);
    }
}

// 优化后的T函数（组合查表法）
inline uint32_t T_combined(uint32_t x) {
    return T0[(x >> 24) & 0xFF] ^
        T1[(x >> 16) & 0xFF] ^
        T2[(x >> 8) & 0xFF] ^
        T3[x & 0xFF];
}

// 优化后的T_prime函数（组合查表法）
inline uint32_t T_prime_combined(uint32_t x) {
    return T_prime0[(x >> 24) & 0xFF] ^
        T_prime1[(x >> 16) & 0xFF] ^
        T_prime2[(x >> 8) & 0xFF] ^
        T_prime3[x & 0xFF];
}

// AESNI优化的轮函数
inline __m128i aesni_round_function(__m128i data, __m128i rk) {
    __m128i t = _mm_xor_si128(data, rk);
    t = _mm_aesenc_si128(t, _mm_setzero_si128());
    return _mm_xor_si128(
        t,
        _mm_xor_si128(
            _mm_rool_epi32(t, 2),
            _mm_xor_si128(
                _mm_rool_epi32(t, 10),
                _mm_xor_si128(
                    _mm_rool_epi32(t, 18),
                    _mm_rool_epi32(t, 24)
                )
            )
        )
    );
}

// 轮函数F（原始版本）
inline uint32_t F_base(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t rk) {
    return x0 ^ T(x1 ^ x2 ^ x3 ^ rk);
}

// 轮函数F（AESNI优化版）
inline uint32_t F_optimized(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t rk) {
    __m128i data = _mm_set_epi32(x3, x2, x1, x0);
    __m128i key = _mm_set1_epi32(rk);
    __m128i result = aesni_round_function(data, key);
    return _mm_extract_epi32(result, 0);
}

// 反序变换
void reverse_words(uint32_t& x0, uint32_t& x1, uint32_t& x2, uint32_t& x3) {
    std::swap(x0, x3);
    std::swap(x1, x2);
}

// 密钥扩展函数（原始版本）
std::vector<uint32_t> expand_key(const uint32_t key[4]) {
    std::vector<uint32_t> rk(32);
    uint32_t k[36];

    // 初始化K0~K3
    for (int i = 0; i < 4; ++i) {
        k[i] = key[i] ^ FK[i];
    }

    // 生成轮密钥
    for (int i = 0; i < 32; ++i) {
        k[i + 4] = k[i] ^ T_prime(k[i + 1] ^ k[i + 2] ^ k[i + 3] ^ CK[i]);
        rk[i] = k[i + 4];
    }

    return rk;
}

// 密钥扩展函数（优化版）
std::vector<uint32_t> expand_key_optimized(const uint32_t key[4]) {
    std::vector<uint32_t> rk(32);
    uint32_t k[36];

    for (int i = 0; i < 4; ++i) {
        k[i] = key[i] ^ FK[i];
    }

    for (int i = 0; i < 32; ++i) {
        k[i + 4] = k[i] ^ T_prime_combined(k[i + 1] ^ k[i + 2] ^ k[i + 3] ^ CK[i]);
        rk[i] = k[i + 4];
    }

    return rk;
}

// 加密函数（原始版本）
void sm4_encrypt(const uint32_t in[4], uint32_t out[4], const std::vector<uint32_t>& rk) {
    uint32_t x[4] = { in[0], in[1], in[2], in[3] };

    for (int i = 0; i < 32; ++i) {
        uint32_t tmp = F_base(x[0], x[1], x[2], x[3], rk[i]);
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = tmp;
    }

    reverse_words(x[0], x[1], x[2], x[3]);

    for (int i = 0; i < 4; ++i) {
        out[i] = x[i];
    }
}

// 加密函数（优化版）
void sm4_encrypt_optimized(const uint32_t in[4], uint32_t out[4], const std::vector<uint32_t>& rk) {
    uint32_t x[4] = { in[0], in[1], in[2], in[3] };

    for (int i = 0; i < 32; ++i) {
        uint32_t tmp = F_optimized(x[0], x[1], x[2], x[3], rk[i]);
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = tmp;
    }

    reverse_words(x[0], x[1], x[2], x[3]);

    for (int i = 0; i < 4; ++i) {
        out[i] = x[i];
    }
}

// 解密函数（原始版本）
void sm4_decrypt(const uint32_t in[4], uint32_t out[4], const std::vector<uint32_t>& rk) {
    uint32_t x[4] = { in[0], in[1], in[2], in[3] };

    for (int i = 0; i < 32; ++i) {
        uint32_t tmp = F_base(x[0], x[1], x[2], x[3], rk[31 - i]);
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = tmp;
    }

    reverse_words(x[0], x[1], x[2], x[3]);

    for (int i = 0; i < 4; ++i) {
        out[i] = x[i];
    }
}

// 辅助函数：将字节数组转换为字（大端序）
uint32_t bytes_to_word(const uint8_t bytes[4]) {
    return (static_cast<uint32_t>(bytes[0]) << 24) |
        (static_cast<uint32_t>(bytes[1]) << 16) |
        (static_cast<uint32_t>(bytes[2]) << 8) |
        static_cast<uint32_t>(bytes[3]);
}

// 辅助函数：将字转换为字节数组（大端序）
void word_to_bytes(uint32_t w, uint8_t bytes[4]) {
    bytes[0] = static_cast<uint8_t>(w >> 24);
    bytes[1] = static_cast<uint8_t>(w >> 16);
    bytes[2] = static_cast<uint8_t>(w >> 8);
    bytes[3] = static_cast<uint8_t>(w);
}

// 性能测试函数
void performance_test() {
    // 测试数据初始化
    uint8_t key_bytes[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };
    uint8_t plaintext_bytes[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint32_t key[4], plaintext[4];
    for (int i = 0; i < 4; ++i) {
        key[i] = bytes_to_word(key_bytes + 4 * i);
        plaintext[i] = bytes_to_word(plaintext_bytes + 4 * i);
    }

    // 密钥扩展
    auto rk_original = expand_key(key);
    auto rk_optimized = expand_key_optimized(key);

    const int iterations = 10000;
    uint32_t ciphertext[4];

    // 原始算法性能测试
    auto start_orig = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sm4_encrypt(plaintext, ciphertext, rk_original);
    }
    auto end_orig = std::chrono::high_resolution_clock::now();

    // 优化算法性能测试
    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sm4_encrypt_optimized(plaintext, ciphertext, rk_optimized);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();

    // 输出结果
    auto orig_time = std::chrono::duration_cast<std::chrono::microseconds>(end_orig - start_orig);
    auto opt_time = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt);

    std::cout << "原始算法平均时间: " << orig_time.count() / (double)iterations << " μs/次\n";
    std::cout << "优化算法平均时间: " << opt_time.count() / (double)iterations << " μs/次\n";
    std::cout << "性能提升: " << (1 - (double)opt_time.count() / orig_time.count()) * 100 << "%\n";
}

int main() {
    init_tables(); // 初始化组合表

    // 测试向量
    uint8_t key_bytes[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };
    uint8_t plaintext_bytes[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    // 将密钥和明文转换为字数组
    uint32_t key[4], plaintext[4], ciphertext[4], decrypted[4];
    for (int i = 0; i < 4; ++i) {
        key[i] = bytes_to_word(key_bytes + 4 * i);
        plaintext[i] = bytes_to_word(plaintext_bytes + 4 * i);
    }

    // 密钥扩展
    std::vector<uint32_t> rk = expand_key(key);

    // 加密测试
    sm4_encrypt(plaintext, ciphertext, rk);
    std::cout << "加密测试完成\n";

    // 解密测试
    sm4_decrypt(ciphertext, decrypted, rk);

    // 验证解密结果
    bool success = true;
    for (int i = 0; i < 4; ++i) {
        if (decrypted[i] != plaintext[i]) {
            success = false;
            break;
        }
    }
    std::cout << "解密验证: " << (success ? "成功" : "失败") << std::endl;

    // 运行性能测试
    performance_test();

    return 0;
}
