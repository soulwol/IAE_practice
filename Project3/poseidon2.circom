// poseidon2.circom
pragma circom 2.0.0;

include "node_modules/circomlib/circuits/poseidon.circom";

template Poseidon2OneBlock() {
    signal input private_in;   // 隐私输入: 哈希原象 (1 个域元素)
    signal output public_hash; // 公开输出: 哈希结果 (1 个域元素)

    // 实例化 Poseidon 哈希组件 (t=2)
    component hasher = Poseidon(1); // 输入个数 = t - 1 = 1
    hasher.inputs[0] <== private_in;
    
    // 设置公开输出
    public_hash <== hasher.out;
}

// 主组件声明公开信号
component main { public [public_hash] } = Poseidon2OneBlock();