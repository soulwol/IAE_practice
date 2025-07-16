#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <map>
#include <cassert>

using namespace std;

// 定义基础类型
using uint8 = unsigned char;
using uint32 = unsigned int;
using uint64 = unsigned long long;

// SM3哈希算法实现
class SM3 {
public:
    SM3() { reset(); }

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
            size_t copy_size = min(len, static_cast<size_t>(64 - buffer_length));
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

    void update(const string& data) {
        update(reinterpret_cast<const uint8*>(data.data()), data.size());
    }

    string finalize() {
        uint8 digest[32];
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

        // 以小端序存储哈希值
        for (int i = 0; i < 8; i++) {
            digest[4 * i] = (state[i] >> 24) & 0xFF;
            digest[4 * i + 1] = (state[i] >> 16) & 0xFF;
            digest[4 * i + 2] = (state[i] >> 8) & 0xFF;
            digest[4 * i + 3] = state[i] & 0xFF;
        }

        return string(reinterpret_cast<char*>(digest), 32);
    }

    static string hash(const string& data) {
        SM3 sm3;
        sm3.update(data);
        return sm3.finalize();
    }

private:
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

    void compress(const uint8 block[64]) {
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

// Merkle树节点
struct MerkleNode {
    string hash;
    MerkleNode* left = nullptr;
    MerkleNode* right = nullptr;

    MerkleNode(string h) : hash(h) {}
    MerkleNode(MerkleNode* l, MerkleNode* r) : left(l), right(r) {
        // 按照RFC6962，内部节点的哈希为：H(0x01 || left_hash || right_hash)
        string data(1, 0x01);  // 前缀0x01
        data += l->hash;
        data += r->hash;
        hash = SM3::hash(data);
    }
};

// Merkle树实现
class MerkleTree {
public:
    MerkleNode* root = nullptr;
    vector<MerkleNode*> leaves;
    size_t leaf_count = 0;
    vector<string> leaf_values;

    // 构建Merkle树
    void build(const vector<string>& items) {
        leaf_values = items;
        leaf_count = items.size();

        // 创建叶子节点
        for (const auto& item : items) {
            // RFC6962：叶子节点的哈希为 H(0x00 || data)
            string leaf_data(1, 0x00);
            leaf_data += item;
            leaves.push_back(new MerkleNode(SM3::hash(leaf_data)));
        }

        // 构建内部节点
        vector<MerkleNode*> current_level = leaves;
        while (current_level.size() > 1) {
            vector<MerkleNode*> next_level;

            for (size_t i = 0; i < current_level.size(); i += 2) {
                MerkleNode* left = current_level[i];
                MerkleNode* right = (i + 1 < current_level.size()) ? current_level[i + 1] : left;
                next_level.push_back(new MerkleNode(left, right));
            }

            current_level = next_level;
        }

        root = current_level[0];
    }

    // 获取Merkle根哈希
    string get_root_hash() const {
        return root->hash;
    }

    // 生成存在性证明
    vector<string> generate_proof(size_t index) {
        vector<string> proof;
        vector<MerkleNode*> path;
        get_path_to_root(leaves[index], path);

        for (MerkleNode* node : path) {
            if (node->left && node->right) {
                // 如果是左节点，兄弟是右节点，反之亦然
                if (node->left == leaves[index]) {
                    proof.push_back(node->right->hash);
                }
                else if (node->right == leaves[index]) {
                    proof.push_back(node->left->hash);
                }
            }
            leaves[index] = node; // 移动到父节点
        }

        return proof;
    }

    // 验证存在性证明
    static bool verify_proof(const string& leaf_hash, const string& root_hash,
        const vector<string>& proof, size_t index, size_t total_leaves) {
        string current = leaf_hash;

        // RFC6962：叶子节点的哈希为 H(0x00 || data)
        string leaf_data(1, 0x00);
        leaf_data += leaf_hash;
        current = SM3::hash(leaf_data);

        size_t tree_index = index;

        for (const auto& sibling_hash : proof) {
            string combined;
            // 计算父节点哈希：H(0x01 || left_hash || right_hash)
            if (tree_index % 2 == 0) {
                // 当前节点是左节点
                combined = string(1, 0x01) + current + sibling_hash;
            }
            else {
                // 当前节点是右节点
                combined = string(1, 0x01) + sibling_hash + current;
            }

            current = SM3::hash(combined);
            tree_index /= 2;
        }

        return current == root_hash;
    }

    // 生成不存在性证明
    vector<string> generate_absence_proof(const string& non_leaf_value) {
        // 查找非叶子值应该在的位置
        auto it = lower_bound(leaf_values.begin(), leaf_values.end(), non_leaf_value);
        size_t index = distance(leaf_values.begin(), it);

        // 获取左右叶子节点的证明
        vector<string> proof;

        if (index > 0) {
            vector<string> left_proof = generate_proof(index - 1);
            proof.insert(proof.end(), left_proof.begin(), left_proof.end());
        }

        if (index < leaf_count) {
            vector<string> right_proof = generate_proof(index);
            proof.insert(proof.end(), right_proof.begin(), right_proof.end());
        }

        return proof;
    }

    // 验证不存在性证明
    bool verify_absence_proof(const string& non_leaf_value, const vector<string>& proof,
        const string& root_hash) {
        // 查找非叶子值应该在的位置
        auto it = lower_bound(leaf_values.begin(), leaf_values.end(), non_leaf_value);
        size_t index = distance(leaf_values.begin(), it);

        // 验证左右叶子节点
        if (index > 0) {
            // 验证左叶子节点
            string left_leaf = leaf_values[index - 1];
            string left_leaf_hash = left_leaf;

            vector<string> left_proof(proof.begin(), proof.begin() + leaf_count);
            if (!verify_proof(left_leaf_hash, root_hash, left_proof, index - 1, leaf_count)) {
                return false;
            }
        }

        if (index < leaf_count) {
            // 验证右叶子节点
            string right_leaf = leaf_values[index];
            string right_leaf_hash = right_leaf;

            vector<string> right_proof(proof.begin() + leaf_count, proof.end());
            if (!verify_proof(right_leaf_hash, root_hash, right_proof, index, leaf_count)) {
                return false;
            }
        }

        // 确认非叶子值确实不存在
        return it == leaf_values.end() || *it != non_leaf_value;
    }

private:
    // 获取从叶子节点到根节点的路径
    void get_path_to_root(MerkleNode* node, vector<MerkleNode*>& path) {
        MerkleNode* current = root;
        while (current != nullptr) {
            path.push_back(current);

            if (current->left == node || current->right == node) {
                return;
            }

            if (node->hash < current->left->hash) {
                current = current->left;
            }
            else {
                current = current->right;
            }
        }
    }
};

// 工具函数：将哈希值转换为十六进制字符串
string to_hex(const string& hash) {
    stringstream ss;
    ss << hex << setfill('0');
    for (uint8_t byte : hash) {
        ss << setw(2) << static_cast<int>(static_cast<uint8_t>(byte));
    }
    return ss.str();
}

int main() {
    const size_t LEAF_COUNT = 100000;  // 10万个叶子节点

    // 1. 生成10万个叶子节点
    vector<string> leaf_values;
    for (size_t i = 0; i < LEAF_COUNT; ++i) {
        leaf_values.push_back("Leaf " + to_string(i));
    }

    // 2. 构建Merkle树
    MerkleTree tree;
    tree.build(leaf_values);

    cout << "=== Merkle树构建完成 ===" << endl;
    cout << "叶子节点数量: " << LEAF_COUNT << endl;
    cout << "根哈希: " << to_hex(tree.get_root_hash()) << endl << endl;

    // 3. 生成并验证存在性证明
    size_t test_index = 50000;
    vector<string> existence_proof = tree.generate_proof(test_index);

    cout << "=== 存在性证明 ===" << endl;
    cout << "目标叶子: \"" << leaf_values[test_index] << "\" (索引 " << test_index << ")" << endl;
    cout << "证明路径长度: " << existence_proof.size() << endl;

    bool exists = MerkleTree::verify_proof(
        leaf_values[test_index],
        tree.get_root_hash(),
        existence_proof,
        test_index,
        LEAF_COUNT
    );

    cout << "验证结果: " << (exists ? "成功" : "失败") << endl << endl;

    // 4. 生成并验证不存在性证明
    string non_existent_value = "Leaf 1000000";  // 肯定不存在
    vector<string> absence_proof = tree.generate_absence_proof(non_existent_value);

    cout << "=== 不存在性证明 ===" << endl;
    cout << "目标值: \"" << non_existent_value << "\"" << endl;
    cout << "证明路径长度: " << absence_proof.size() << endl;

    bool absent = tree.verify_absence_proof(
        non_existent_value,
        absence_proof,
        tree.get_root_hash()
    );

    cout << "验证结果: " << (absent ? "成功" : "失败") << endl;

    // 5. 验证边缘情况
    // 测试一个刚好在范围内的值不存在
    string middle_value = "Leaf " + to_string(LEAF_COUNT / 2);
    vector<string> middle_proof = tree.generate_absence_proof(middle_value);
    bool middle_absent = tree.verify_absence_proof(
        middle_value,
        middle_proof,
        tree.get_root_hash()
    );

    // 由于这个值实际存在，验证应该失败
    cout << "\n边缘情况测试:" << endl;
    cout << "验证存在的叶子值："
        << (middle_absent ? "错误地报告不存在" : "正确地检测到存在")
        << endl;

    return 0;
}