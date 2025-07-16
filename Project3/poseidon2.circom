pragma circom 2.1.4;

include "node_modules/circomlib/circuits/bitify.circom";
include "node_modules/circomlib/circuits/poseidon.circom";

template Poseidon2Hash() {
    signal input preimage;
    signal output hash;

    // 将整数转换为256位二进制
    component n2b = Num2Bits(256);
    n2b.in <== preimage;

    // 分割为两个128位块
    component b2n1 = Bits2Num(128);
    component b2n2 = Bits2Num(128);
    
    // 低128位
    for (var i = 0; i < 128; i++) {
        b2n1.in[i] <== n2b.out[i];
    }
    
    // 高128位
    for (var i = 0; i < 128; i++) {
        b2n2.in[i] <== n2b.out[128 + i];
    }

    // 调用Poseidon哈希
    component hasher = Poseidon(2);
    hasher.inputs[0] <== b2n1.out;
    hasher.inputs[1] <== b2n2.out;
    
    hash <== hasher.out;
}

template Main() {
    // 隐私输入：原像
    signal input preimage;
    // 输出哈希值
    signal output hash;

    component poseidon = Poseidon2Hash();
    poseidon.preimage <== preimage;
    hash <== poseidon.hash;
}

component main = Main();