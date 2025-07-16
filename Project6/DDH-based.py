from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from phe import paillier
import os
import random
import hashlib
import math


class DDHPrivateIntersectionSum:
    # P-256 曲线标准参数（SECP256R1/NIST P-256）
    P256_PARAMS = {
        'p': 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF,
        'a': 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC,
        'b': 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B,
        'order': 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    }

    def __init__(self, curve=ec.SECP256R1()):
        self.curve = curve
        self.params = self.P256_PARAMS
        self.p = self.params['p']
        self.a = self.params['a']
        self.b = self.params['b']
        self.order = self.params['order']

    # --------------------------
    # 椭圆曲线点操作实现
    # --------------------------
    # 点加倍算法修正
    def _point_double(self, x, y):
        if y == 0:  # 处理无穷远点
            return (0, 0)

        # 计算斜率λ = (3x² + a)/(2y) mod p
        numerator = (3 * pow(x, 2, self.p) + self.a) % self.p
        denominator = (2 * y) % self.p
        inv_denom = pow(denominator, self.p - 2, self.p)  # 费马小定理求逆
        lam = (numerator * inv_denom) % self.p

        # 计算新点坐标
        x3 = (pow(lam, 2, self.p) - 2 * x) % self.p
        y3 = (lam * (x - x3) - y) % self.p
        return x3, y3

    # 点加法算法修正
    def _point_add(self, x1, y1, x2, y2):
        # 处理无穷远点情况
        if x1 == 0 and y1 == 0:
            return (x2, y2)
        if x2 == 0 and y2 == 0:
            return (x1, y1)
        if x1 == x2 and y1 != y2:  # 垂直切线
            return (0, 0)

        # 计算斜率
        if x1 == x2 and y1 == y2:
            return self._point_double(x1, y1)

        numerator = (y2 - y1) % self.p
        denominator = (x2 - x1) % self.p
        inv_denom = pow(denominator, self.p - 2, self.p)
        lam = (numerator * inv_denom) % self.p

        # 计算新点坐标
        x3 = (pow(lam, 2, self.p) - x1 - x2) % self.p
        y3 = (lam * (x1 - x3) - y1) % self.p
        return x3, y3

    def _scalar_multiply(self, x, y, scalar):
        # 处理边界情况
        if scalar == 0:
            return (0, 0)
        if scalar == 1:
            return (x, y)

        result_x, result_y = 0, 0
        current_x, current_y = x, y

        # 增强的double-and-add算法
        while scalar:
            if scalar & 1:
                if result_x == 0 and result_y == 0:
                    result_x, result_y = current_x, current_y
                else:
                    result_x, result_y = self._point_add(result_x, result_y, current_x, current_y)

            # 点加倍
            current_x, current_y = self._point_double(current_x, current_y)
            scalar >>= 1

        return result_x, result_y

    def _mod_inverse(self, a, m):
        """计算模逆元（扩展欧几里得算法）"""
        # 扩展欧几里得算法
        g, x, _ = self._extended_gcd(a, m)
        if g != 1:
            raise ValueError(f"{a} has no inverse modulo {m}")
        return x % m

    def _extended_gcd(self, a, b):
        """扩展欧几里得算法实现"""
        if b == 0:
            return (a, 1, 0)
        else:
            g, x, y = self._extended_gcd(b, a % b)
            return (g, y, x - (a // b) * y)

    # --------------------------
    # 协议实现
    # --------------------------
    def p1_round1(self, v_set: list[str], k1: int) -> list[bytes]:
        """P1计算 H(v_i)^k1 并随机排列"""
        hashed_exponents = []
        for v in v_set:
            # 哈希到曲线点
            x, y = self._hash_to_point(v)

            # 点乘法
            result_x, result_y = self._scalar_multiply(x, y, k1)

            # 序列化点 (x坐标)
            # 实际应用中应使用完整点序列化，此处简化
            point_bytes = result_x.to_bytes(32, 'big')
            hashed_exponents.append(point_bytes)

        random.shuffle(hashed_exponents)
        return hashed_exponents

    def p2_round2(self, received: list[bytes], w_pairs: list[tuple], k2: int, pk: paillier.PaillierPublicKey) -> tuple:
        """P2计算 Z 和加密对"""
        # 计算 Z = {H(v_i)^{k1 k2}}
        Z = []
        for point_bytes in received:
            # 反序列化点
            x = int.from_bytes(point_bytes, 'big')

            # 点乘法
            result_x, _ = self._scalar_multiply(x, self._calculate_y(x), k2)

            # 序列化结果点
            z_bytes = result_x.to_bytes(32, 'big')
            Z.append(z_bytes)

        random.shuffle(Z)

        # 计算 {(H(w_j)^k2, Enc(t_j))}
        encrypted_pairs = []
        for w, t in w_pairs:
            # 哈希到曲线点
            x, y = self._hash_to_point(w)

            # 点乘法
            result_x, _ = self._scalar_multiply(x, y, k2)

            # 序列化点
            point_bytes = result_x.to_bytes(32, 'big')

            # Paillier加密值
            enc_t = pk.encrypt(t)
            encrypted_pairs.append((point_bytes, enc_t))

        random.shuffle(encrypted_pairs)
        return Z, encrypted_pairs

    def p1_round3(self, Z: list[bytes], encrypted_pairs: list, k1: int,
                  pk: paillier.PaillierPublicKey) -> paillier.EncryptedNumber:
        """P1检测交集并同态求和"""
        # 先计算Z集的字节表示
        Z_set = set(Z)

        # 同态求和初始化
        sum_cipher = pk.encrypt(0)

        # 交集计数
        intersection_count = 0

        for w_point_bytes, enc_t in encrypted_pairs:
            # 反序列化点
            x = int.from_bytes(w_point_bytes, 'big')

            # 点乘法
            result_x, _ = self._scalar_multiply(x, self._calculate_y(x), k1)

            # 序列化结果点
            test_bytes = result_x.to_bytes(32, 'big')

            # 检测交集
            if test_bytes in Z_set:
                sum_cipher = sum_cipher + enc_t  # 同态加法
                intersection_count += 1

        print(f"检测到 {intersection_count} 个交集元素")
        return sum_cipher

    # --------------------------
    # 辅助函数
    # --------------------------
    def _hash_to_point(self, identifier: str) -> tuple:
        # 使用HKDF确保均匀分布
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"EC_Point_Mapping",
            backend=default_backend()
        )
        key_bytes = hkdf.derive(identifier.encode())

        # 循环直到找到有效点
        attempts = 0
        while attempts < 10:  # 防止无限循环
            value = int.from_bytes(key_bytes, 'big') % self.p
            y_squared = (pow(value, 3, self.p) + self.a * value + self.b) % self.p

            # 尝试Tonelli-Shanks算法
            if self._is_quadratic_residue(y_squared, self.p):
                y = self._modular_sqrt(y_squared, self.p)
                return (value, y)

            # 下一次尝试
            key_bytes = hashlib.sha256(key_bytes).digest()
            attempts += 1

        raise ValueError(f"无法为标识符找到有效点: {identifier}")
    def _calculate_y(self, x):
        """根据x坐标计算y坐标 (y² = x³ + ax + b mod p)"""
        # 计算 y² = x³ + a*x + b mod p
        y_squared = (x * x * x + self.a * x + self.b) % self.p

        # 求平方根（模p）
        return self._modular_sqrt(y_squared, self.p)

    def _modular_sqrt(self, a, p):
        """Tonelli-Shanks算法计算模平方根"""
        # 简单情况：p ≡ 3 mod 4
        if p % 4 == 3:
            return pow(a, (p + 1) // 4, p)

        # 通用实现
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1

        # 找二次非剩余
        z = 2
        while self._is_quadratic_residue(z, p):
            z += 1

        # Tonelli-Shanks算法
        c = pow(z, q, p)
        r = pow(a, (q + 1) // 2, p)
        t = pow(a, q, p)
        m = s

        while t != 1:
            i = 0
            temp = t
            while temp != 1:
                temp = (temp * temp) % p
                i += 1

            b = pow(c, 2 ** (m - i - 1), p)
            r = (r * b) % p
            t = (t * b * b) % p
            c = (b * b) % p
            m = i

        return r

    def _is_quadratic_residue(self, a, p):
        """检查a是否是模p的二次剩余"""
        return pow(a, (p - 1) // 2, p) == 1


# ================================================
# 协议演示执行流程
# ===============================================
if __name__ == "__main__":
    # 初始化协议
    protocol = DDHPrivateIntersectionSum()

    # 打印曲线参数
    print("曲线参数:")
    print(f"模数 (p): {hex(protocol.p)[:20]}...")
    print(f"阶数 (order): {hex(protocol.order)[:20]}...")

    # 生成Paillier密钥对
    print("\n生成Paillier密钥对...")
    paillier_pubkey, paillier_privkey = paillier.generate_paillier_keypair(n_length=768)

    # 生成安全的随机指数
    print("生成随机指数...")
    k1 = random.SystemRandom().randint(1, protocol.order - 1)
    k2 = random.SystemRandom().randint(1, protocol.order - 1)
    print(f"k1: {hex(k1)[:20]}...")
    print(f"k2: {hex(k2)[:20]}...")

    # 模拟数据集
    p1_data = ["user1", "user2", "user3"]
    p2_data = [("user1", 100), ("user2", 200), ("user3", 300), ("user4", 400)]

    print("\n开始协议执行...")
    print(f"P1数据: {p1_data}")
    print(f"P2数据: {p2_data}")
    print(f"期望交集和: 600 (user1+user2+user3)")

    try:
        # Round 1: P1 -> P2
        print("\n=== Round 1: P1 → P2 ===")
        p1_to_p2 = protocol.p1_round1(p1_data, k1)
        print(f"P1发送了 {len(p1_to_p2)} 个点的列表")

        # Round 2: P2 -> P1
        print("\n=== Round 2: P2 → P1 ===")
        Z, p2_to_p1 = protocol.p2_round2(p1_to_p2, p2_data, k2, paillier_pubkey)
        print(f"P2发送了Z（{len(Z)}点）和 {len(p2_to_p1)} 个加密对")

        # Round 3: P1 -> P2
        print("\n=== Round 3: P1 → P2 ===")
        encrypted_sum = protocol.p1_round3(Z, p2_to_p1, k1, paillier_pubkey)
        print(f"P1检测到交集并计算加密和")

        # P2解密获得交集和
        print("\nP2解密结果...")
        sum_value = paillier_privkey.decrypt(encrypted_sum)

        # 输出结果
        print(f"\n=== 结果 ===")
        print(f"计算交集和: {sum_value}")
        print(f"期望交集和: 600")

        if sum_value == 600:
            print("✅ 测试成功！协议正确执行")
        else:
            print(f"❌ 测试失败！差值: {abs(sum_value - 600)}")
            print("可能原因: 点操作实现问题或哈希冲突")

    except Exception as e:
        print(f"\n❌ 协议执行错误: {str(e)}")
        import traceback

        traceback.print_exc()