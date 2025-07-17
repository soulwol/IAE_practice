import hashlib
import random


# 注意：这是简化的教学示例，真实ECDSA算法无法被攻破
# 比特币使用secp256k1椭圆曲线，本示例使用简化数学模型说明原理

class ToyECDSA:
    def __init__(self):
        # 模拟椭圆曲线参数（真实场景使用256位素数）
        self.p = 101  # 小素数用于演示
        self.G = (2, 4)  # 生成元
        self.n = 17  # 阶

    def add_points(self, P, Q):
        """椭圆曲线点加法（简化版）"""
        if P == "O": return Q
        if Q == "O": return P
        if P[0] == Q[0] and P[1] != Q[1]:
            return "O"  # 无穷远点

        if P != Q:
            lam = (Q[1] - P[1]) * pow(Q[0] - P[0], -1, self.p) % self.p
        else:
            lam = (3 * P[0] * P[0]) * pow(2 * P[1], -1, self.p) % self.p

        x = (lam * lam - P[0] - Q[0]) % self.p
        y = (lam * (P[0] - x) - P[1]) % self.p
        return (x, y)

    def scalar_mult(self, k, P):
        """椭圆曲线标量乘法"""
        R = "O"  # 无穷远点
        while k:
            if k & 1:
                R = self.add_points(R, P)
            P = self.add_points(P, P)
            k >>= 1
        return R


def forge_signature():
    """伪造签名演示"""
    # 初始化简化的ECDSA系统
    ecdsa = ToyECDSA()

    # 模拟中本聪密钥对（私钥d，公钥Q）
    d = random.randint(1, ecdsa.n - 1)  # 真实中本聪私钥（无人知晓）
    Q = ecdsa.scalar_mult(d, ecdsa.G)  # 公钥

    print(f"模拟中本聪公钥: {Q}\n")

    # 攻击者伪造签名过程（无需知道d）
    # 选择两个随机数u, v
    u = random.randint(1, ecdsa.n - 1)
    v = random.randint(1, ecdsa.n - 1)

    # 计算伪造的公钥点
    R = ecdsa.add_points(
        ecdsa.scalar_mult(u, ecdsa.G),
        ecdsa.scalar_mult(v, Q)
    )

    # 计算伪造的签名(r, s)
    r = R[0] % ecdsa.n
    s = r * pow(v, -1, ecdsa.n) % ecdsa.n
    forged_sig = (r, s)

    # 伪造消息的哈希（任意消息都可以被"验证"）
    forged_msg_hash = u * s % ecdsa.n

    print("伪造成功！以下签名将被错误地验证为中本聪签名：")
    print(f"伪造消息哈希: {forged_msg_hash}")
    print(f"伪造签名(r, s): {forged_sig}\n")

    # 伪实验证过程
    w = pow(s, -1, ecdsa.n)
    u1 = (forged_msg_hash * w) % ecdsa.n
    u2 = (r * w) % ecdsa.n

    P = ecdsa.add_points(
        ecdsa.scalar_mult(u1, ecdsa.G),
        ecdsa.scalar_mult(u2, Q)
    )

    verify_r = P[0] % ecdsa.n
    is_valid = verify_r == r

    print("验证伪造签名结果: " + ("成功" if is_valid else "失败"))
    print("注意：这是数学模型演示，真实ECDSA不会出现此漏洞！")


if __name__ == "__main__":
    print("=" * 60)
    print("中本聪签名伪造教学演示（基于数学原理简化模型）")
    print("=" * 60)
    print("警告：真实ECDSA签名不可伪造，此演示仅用于教学目的\n")

    forge_signature()

    print("\n" + "=" * 60)
    print("核心漏洞说明：")
    print("本例通过构造 (r,s) 使得: r = (uG + vQ)_x, s = r/v")
    print("伪造消息哈希: e = u·s mod n")
    print("验证时计算： (e/s)G + (r/s)Q = (u + vd)G")
    print("通过选择 u = e/s, v = r/s 即可通过验证")
    print("=" * 60)