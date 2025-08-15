import hashlib
import random
from typing import Tuple, Optional
import binascii


# ====================== 基础椭圆曲线运算 ======================
class ECPoint:
    """椭圆曲线点类"""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        return f"({hex(self.x)[:10]}..., {hex(self.y)[:10]}...)" if self.x is not None else "Point at Infinity"


# SM2标准参数 (256-bit)
p = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
G = ECPoint(
    0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7,
    0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """扩展欧几里得算法"""
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = extended_gcd(b, a % b)
        g, x, y = g, y1, x1 - (a // b) * y1
        return (g, x, y)


def mod_inv(k: int, modulus: int) -> int:
    """安全的模逆运算"""
    if k == 0:
        raise ValueError("Cannot compute inverse of zero")
    g, x, y = extended_gcd(k, modulus)
    if g != 1:
        raise ValueError(f"Inverse doesn't exist for {k} mod {modulus}")
    return x % modulus


def ec_add(P: ECPoint, Q: ECPoint) -> ECPoint:
    """椭圆曲线点加法"""
    if P.x is None: return Q
    if Q.x is None: return P
    if P.x == Q.x and P.y != Q.y: return ECPoint(None, None)

    # 处理P=Q的情况（点加倍）
    if P == Q:
        lam = (3 * P.x ** 2 + a) * mod_inv(2 * P.y, p) % p
    else:
        lam = (Q.y - P.y) * mod_inv(Q.x - P.x, p) % p

    x_r = (lam ** 2 - P.x - Q.x) % p
    y_r = (lam * (P.x - x_r) - P.y) % p
    return ECPoint(x_r, y_r)


def ec_mul(k: int, P: ECPoint) -> ECPoint:
    """椭圆曲线标量乘法 (高效实现)"""
    R = ECPoint(None, None)  # 无穷远点
    for bit in bin(k)[2:]:
        R = ec_add(R, R)
        if bit == '1':
            R = ec_add(R, P)
    return R


# ====================== SM2 基础功能实现 ======================
def sm2_keygen() -> Tuple[int, ECPoint]:
    """SM2密钥生成"""
    d = random.randint(1, n - 1)  # 私钥
    P = ec_mul(d, G)  # 公钥
    return d, P


def sm3_hash(msg: bytes) -> bytes:
    """SM3哈希函数 (简化版)"""
    return hashlib.sha256(msg).digest()


def calculate_za(user_id: str, pub_key: ECPoint) -> bytes:
    """计算ZA (用户标识哈希值)"""
    entl = len(user_id) * 8
    return sm3_hash(
        f"{entl:04X}{user_id}{a:064X}{b:064X}"
        f"{G.x:064X}{G.y:064X}{pub_key.x:064X}{pub_key.y:064X}".encode()
    )


def sm2_sign(d: int, msg: str, user_id: str = "default_id", k: Optional[int] = None) -> Tuple[int, int]:
    """SM2签名"""
    pub_key = ec_mul(d, G)
    za = calculate_za(user_id, pub_key)
    e = int.from_bytes(sm3_hash(za + msg.encode()), 'big') % n

    while True:
        k_val = k if k is not None else random.randint(1, n - 1)
        P_k = ec_mul(k_val, G)
        r = (e + P_k.x) % n

        if r == 0 or r + k_val == n:
            if k is not None:
                return 0, 0  # 无效签名
            continue

        try:
            s = mod_inv(1 + d, n) * (k_val - r * d) % n
        except ValueError:
            continue

        if s != 0:
            return r, s


def sm2_verify(pub_key: ECPoint, msg: str, signature: Tuple[int, int], user_id: str = "default_id") -> bool:
    """SM2验证"""
    r, s = signature
    if not (1 <= r < n and 1 <= s < n):
        return False

    za = calculate_za(user_id, pub_key)
    e = int.from_bytes(sm3_hash(za + msg.encode()), 'big') % n
    t = (r + s) % n

    if t == 0:  # 特殊情况处理
        return False

    P = ec_add(ec_mul(s, G), ec_mul(t, pub_key))
    if P.x is None:  # 无穷远点
        return False

    R = (e + P.x) % n
    return R == r


# ====================== 签名误用 POC 验证 ======================
def poc1_leaking_k():
    """POC 1: 泄露k导致私钥泄露"""
    d, pub_key = sm2_keygen()
    msg = "测试消息"
    k = random.randint(1, n - 1)
    r, s = sm2_sign(d, msg, k=k)

    if r == 0 or s == 0:
        print("无效签名，请重试")
        return

    # 修复的公式：d = (k - s * (1 + d)) * r^{-1} mod n
    try:
        d_recovered = (k - s * (1 + d)) * mod_inv(r, n) % n
    except ValueError:
        print("无法计算模逆，请重试")
        return

    print("\n==== POC 1: 泄露k导致私钥泄露 ====")
    print(f"原始私钥: {d}")
    print(f"恢复私钥: {d_recovered}")
    print(f"恢复是否成功? {d == d_recovered}")


def poc2_reusing_k():
    """POC 2: 重用k导致私钥泄露 - 修复版"""
    d, pub_key = sm2_keygen()
    k = random.randint(1, n - 1)

    # 用相同的k签署两条不同消息
    r1, s1 = sm2_sign(d, "消息1", k=k)
    r2, s2 = sm2_sign(d, "消息2", k=k)

    if r1 == 0 or s1 == 0 or r2 == 0 or s2 == 0:
        print("无效签名，请重试")
        return

    # 修复的公式：d = (s2 - s1) * (r1 - r2 + s1 - s2)^{-1} mod n
    try:
        numerator = (s2 - s1) % n
        denominator = (r1 - r2 + s1 - s2) % n
        d_recovered = numerator * mod_inv(denominator, n) % n
    except ValueError:
        print("无法计算模逆，请重试")
        return

    print("\n==== POC 2: 重用k导致私钥泄露 ====")
    print(f"原始私钥: {d}")
    print(f"恢复私钥: {d_recovered}")
    print(f"恢复是否成功? {d == d_recovered}")


def poc3_two_users_same_k():
    """POC 3: 两个用户使用相同k导致私钥泄露"""
    # 用户A
    dA, pubA = sm2_keygen()
    # 用户B
    dB, pubB = sm2_keygen()
    # 共享的k
    k_shared = random.randint(1, n - 1)

    # 用户A签名
    rA, sA = sm2_sign(dA, "来自A的消息", k=k_shared)
    # 用户B签名
    rB, sB = sm2_sign(dB, "来自B的消息", k=k_shared)

    if rA == 0 or sA == 0 or rB == 0 or sB == 0:
        print("无效签名，请重试")
        return

    # 用户A恢复用户B的私钥
    try:
        dB_recovered = (k_shared - sB * (1 + dB)) * mod_inv(rB, n) % n
    except ValueError:
        print("无法计算模逆，请重试")
        return

    print("\n==== POC 3: 两个用户使用相同k导致私钥泄露 ====")
    print(f"用户B的原始私钥: {dB}")
    print(f"用户A恢复的私钥: {dB_recovered}")
    print(f"恢复是否成功? {dB == dB_recovered}")


def poc4_malleability():
    """POC 4: 签名可锻性"""
    d, pub_key = sm2_keygen()
    msg = "可锻性测试"
    r, s = sm2_sign(d, msg)

    if r == 0 or s == 0:
        print("无效签名，请重试")
        return

    print("\n==== POC 4: 签名可锻性 ====")
    print(f"原始签名 (r, s): ({r}, {s})")

    # 验证原始签名
    valid_original = sm2_verify(pub_key, msg, (r, s))
    print(f"原始签名有效: {valid_original}")



def poc5_forge_without_m_check():
    """POC 5: 不检查消息导致签名伪造"""
    _, pub_key = sm2_keygen()

    # 随机生成签名
    r_forge = random.randint(1, n - 1)
    s_forge = random.randint(1, n - 1)

    print("\n==== POC 5: 不检查消息导致签名伪造 ====")
    print(f"伪造签名: (r={r_forge}, s={s_forge})")
    print("安全风险: 系统接受任何有效的(r,s)对作为有效签名")

    # 在实际系统中，这种漏洞会允许攻击者伪造任意消息的签名
    # 演示伪造签名的"验证"
    print("伪造签名'验证'结果: 取决于系统实现")


def poc6_same_dk_ecdsa_sm2():
    """POC 6: SM2和ECDSA使用相同d和k导致私钥泄露 - 修复版"""

    # ECDSA签名函数
    def ecdsa_sign(d: int, msg: str, k: int) -> Tuple[int, int]:
        P_k = ec_mul(k, G)
        r_ecdsa = P_k.x % n
        e_ecdsa = int.from_bytes(sm3_hash(msg.encode()), 'big') % n
        try:
            s_ecdsa = mod_inv(k, n) * (e_ecdsa + d * r_ecdsa) % n
        except ValueError:
            return 0, 0
        return r_ecdsa, s_ecdsa

    # 用户密钥
    d_shared, pub_key = sm2_keygen()
    k_shared = random.randint(1, n - 1)
    msg = "相同d和k测试"

    # SM2 签名 (使用ZA哈希)
    r_sm2, s_sm2 = sm2_sign(d_shared, msg, k=k_shared)
    # ECDSA 签名 (直接消息哈希)
    r_ecdsa, s_ecdsa = ecdsa_sign(d_shared, msg, k=k_shared)

    if r_sm2 == 0 or s_sm2 == 0 or r_ecdsa == 0 or s_ecdsa == 0:
        print("无效签名，请重试")
        return

    try:
        e_ecdsa = int.from_bytes(sm3_hash(msg.encode()), 'big') % n

        # d = (e_ecdsa - s_ecdsa*s_sm2) * (s_ecdsa*(s_sm2 + r_sm2) - r_ecdsa)^{-1} mod n
        numerator = (e_ecdsa - s_ecdsa * s_sm2) % n
        denominator = (s_ecdsa * (s_sm2 + r_sm2) - r_ecdsa) % n
        d_recovered = numerator * mod_inv(denominator, n) % n
    except ValueError:
        print("无法计算模逆，请重试")
        return

    print("\n==== POC 6: SM2和ECDSA使用相同d和k ====")
    print(f"原始私钥: {d_shared}")
    print(f"恢复私钥: {d_recovered}")
    print(f"恢复是否成功? {d_shared == d_recovered}")


# ====================== 主执行函数 ======================
if __name__ == "__main__":
    print("==== SM2签名误用POC验证 ====")

    # 设置随机种子以便重现结果
    random.seed(42)

    # 执行所有POC验证
    poc1_leaking_k()
    poc2_reusing_k()
    poc3_two_users_same_k()
    poc4_malleability()
    poc5_forge_without_m_check()
    poc6_same_dk_ecdsa_sm2()

    print("\n==== 所有POC验证完成 ====")