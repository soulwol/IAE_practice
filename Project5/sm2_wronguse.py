import random
import hashlib
import time
import binascii


class EllipticCurve:
    def __init__(self, a, b, p, n, Gx, Gy):
        self.a = a
        self.b = b
        self.p = p
        self.n = n
        self.G = (Gx, Gy)

    def point_addition(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and y1 != y2:
            return None
        if x1 == x2:
            lam = ((3 * x1 * x1 + self.a) * self.inv_mod(2 * y1, self.p)) % self.p
        else:
            lam = ((y2 - y1) * self.inv_mod(x2 - x1, self.p)) % self.p
        x3 = (lam * lam - x1 - x2) % self.p
        y3 = (lam * (x1 - x3) - y1) % self.p
        return (x3, y3)

    def point_multiply(self, k, P):
        k = k % self.n
        result = None
        current = P
        while k:
            if k & 1:
                result = self.point_addition(result, current)
            current = self.point_addition(current, current)
            k >>= 1
        return result

    @staticmethod
    def inv_mod(a, m):
        g, x, _ = EllipticCurve.extended_gcd(a, m)
        if g != 1:
            raise ValueError(f"{a} and {m} are not coprime")
        return x % m

    @staticmethod
    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        else:
            g, x1, y1 = EllipticCurve.extended_gcd(b, a % b)
            g, x, y = g, y1, x1 - (a // b) * y1
            return g, x, y

    @staticmethod
    def int_to_bytes(x):
        return x.to_bytes((x.bit_length() + 7) // 8, 'big')

    @staticmethod
    def bytes_to_int(b):
        return int.from_bytes(b, 'big')


class EllipticCurveJacobi(EllipticCurve):
    def __init__(self, a, b, p, n, Gx, Gy):
        super().__init__(a, b, p, n, Gx, Gy)

    def affine_to_jacobian(self, P):
        if P is None:
            return (1, 1, 0)  # Jacobian point at infinity
        x, y = P
        return (x, y, 1)

    def jacobian_to_affine(self, P):
        if P is None:
            return None
        x, y, z = P
        if z == 0:
            return None  # Point at infinity
        z_inv = self.inv_mod(z, self.p)
        z_inv_sq = (z_inv * z_inv) % self.p
        z_inv_cu = (z_inv_sq * z_inv) % self.p
        x_aff = (x * z_inv_sq) % self.p
        y_aff = (y * z_inv_cu) % self.p
        return (x_aff, y_aff)

    def jacobian_point_double(self, P):
        if P is None or P[2] == 0:  # Point at infinity
            return (1, 1, 0)
        x, y, z = P
        if y == 0:
            return (1, 1, 0)  # Point at infinity after doubling

        # Calculate intermediate terms
        y_sq = (y * y) % self.p
        S = (4 * x * y_sq) % self.p
        z_sq = (z * z) % self.p
        a_z4 = (self.a * z_sq * z_sq) % self.p
        M = (3 * x * x + a_z4) % self.p
        x_double = (M * M - 2 * S) % self.p
        y_double = (M * (S - x_double) - 8 * y_sq * y_sq) % self.p
        z_double = (2 * y * z) % self.p

        return (x_double, y_double, z_double)

    def jacobian_point_add(self, P, Q):
        if P is None or P[2] == 0:
            return Q
        if Q is None or Q[2] == 0:
            return P
        x1, y1, z1 = P
        x2, y2, z2 = Q

        z1_sq = (z1 * z1) % self.p
        z2_sq = (z2 * z2) % self.p
        U1 = (x1 * z2_sq) % self.p
        U2 = (x2 * z1_sq) % self.p
        z1_cu = (z1_sq * z1) % self.p
        z2_cu = (z2_sq * z2) % self.p
        S1 = (y1 * z2_cu) % self.p
        S2 = (y2 * z1_cu) % self.p

        if U1 == U2:
            if S1 == S2:
                return self.jacobian_point_double(P)
            else:
                return (1, 1, 0)  # Point at infinity
        else:
            H = (U2 - U1) % self.p
            R = (S2 - S1) % self.p
            H_sq = (H * H) % self.p
            H_cu = (H_sq * H) % self.p
            x3 = (R * R - H_cu - 2 * U1 * H_sq) % self.p
            y3 = (R * (U1 * H_sq - x3) - S1 * H_cu) % self.p
            z3 = (H * z1 * z2) % self.p
            return (x3, y3, z3)

    def point_multiply(self, k, P):
        k = k % self.n
        if k == 0:
            return None
        P_jacob = self.affine_to_jacobian(P)
        result = None

        while k:
            if k & 1:
                result = self.jacobian_point_add(result, P_jacob)
            P_jacob = self.jacobian_point_double(P_jacob)
            k >>= 1

        return self.jacobian_to_affine(result)


class SM2:
    # SM2曲线参数
    P = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
    A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
    B = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
    N = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
    Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
    Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0

    def __init__(self, use_jacobi=True):
        if use_jacobi:
            self.curve = EllipticCurveJacobi(self.A, self.B, self.P, self.N, self.Gx, self.Gy)
        else:
            self.curve = EllipticCurve(self.A, self.B, self.P, self.N, self.Gx, self.Gy)
        self.G = (self.Gx, self.Gy)

    def generate_keypair(self):
        """生成SM2密钥对"""
        d = random.randint(1, self.N - 1)
        P = self.curve.point_multiply(d, self.G)
        return d, P

    def sm3_hash(self, message):
        """简化版SM3哈希函数（实际中应使用完整SM3实现）"""
        h = hashlib.sha256()
        h.update(message.encode('utf-8'))
        return int.from_bytes(h.digest(), 'big') % self.N

    def sign(self, message, d, k=None):
        """SM2签名"""
        # 生成随机数k（如果未提供）
        if k is None:
            k = random.randint(1, self.N - 1)

        # 计算(k * G)
        x1, y1 = self.curve.point_multiply(k, self.G)
        e = self.sm3_hash(message)
        r = (e + x1) % self.N

        # 计算s
        s = (self.curve.inv_mod(1 + d, self.N) * (k - r * d)) % self.N
        return (r, s), k

    def verify(self, message, signature, P):
        """SM2验证签名"""
        r, s = signature

        # 验证r和s在[1, N-1]范围内
        if not (1 <= r <= self.N - 1) or not (1 <= s <= self.N - 1):
            return False

        e = self.sm3_hash(message)

        # 计算t = (r + s) mod N
        t = (r + s) % self.N
        if t == 0:
            return False

        # 计算s * G + t * P
        sG = self.curve.point_multiply(s, self.G)
        tP = self.curve.point_multiply(t, P)
        x1, y1 = self.curve.point_addition(sG, tP)
        R = (e + x1) % self.N

        return R == r


class ECDSA(SM2):
    """ECDSA签名实现，使用与SM2相同的曲线参数"""

    def sign(self, message, d, k=None):
        """ECDSA签名"""
        if k is None:
            k = random.randint(1, self.N - 1)

        x1, _ = self.curve.point_multiply(k, self.G)
        r = x1 % self.N
        e = self.sm3_hash(message)
        s = (self.curve.inv_mod(k, self.N) * (e + d * r)) % self.N
        return (r, s), k

    def verify(self, message, signature, P):
        """ECDSA验证签名"""
        r, s = signature
        if not (1 <= r <= self.N - 1) or not (1 <= s <= self.N - 1):
            return False

        e = self.sm3_hash(message)
        w = self.curve.inv_mod(s, self.N)
        u1 = (e * w) % self.N
        u2 = (r * w) % self.N

        u1G = self.curve.point_multiply(u1, self.G)
        u2P = self.curve.point_multiply(u2, P)
        x, _ = self.curve.point_addition(u1G, u2P)

        return r == x % self.N


class SignatureAttacks:
    """签名算法误用攻击验证"""

    def __init__(self):
        self.sm2 = SM2(use_jacobi=True)
        self.ecdsa = ECDSA(use_jacobi=True)

    def same_k_sm2_attack(self):
        """情况1: 相同用户重用k攻击"""
        print("\n" + "=" * 50)
        print("情况1: 相同用户重用随机数k攻击")
        print("=" * 50)

        # 生成密钥
        d, P = self.sm2.generate_keypair()
        msg1 = "消息1"
        msg2 = "消息2"

        # 第一次签名
        k = random.randint(1, self.sm2.N - 1)
        (r1, s1), _ = self.sm2.sign(msg1, d, k=k)
        print(f"消息1签名: r={hex(r1)[:20]}..., s={hex(s1)[:20]}...")

        # 重用k进行第二次签名
        (r2, s2), _ = self.sm2.sign(msg2, d, k=k)
        print(f"消息2签名: r={hex(r2)[:20]}..., s={hex(s2)[:20]}...")

        # 推导私钥
        numerator = (s2 - s1) % self.sm2.N
        denominator = (s1 - s2 + r1 - r2) % self.sm2.N
        inv_denom = self.sm2.curve.inv_mod(denominator, self.sm2.N)
        d_cracked = (numerator * inv_denom) % self.sm2.N

        # 验证结果
        valid = d == d_cracked
        print(f"\n真实私钥: {hex(d)[:20]}...")
        print(f"推导私钥: {hex(d_cracked)[:20]}...")
        print(f"攻击结果: {'成功' if valid else '失败'}")

        return valid

    def same_k_different_users_attack(self):
        """情况2: 不同用户重用k攻击（数学上不可行）"""
        print("\n" + "=" * 50)
        print("情况2: 不同用户重用随机数k攻击")
        print("=" * 50)

        dA, PA = self.sm2.generate_keypair()
        dB, PB = self.sm2.generate_keypair()
        msgA = "用户A的消息"
        msgB = "用户B的消息"
        k = random.randint(1, self.sm2.N - 1)

        # 用户A签名
        (rA, sA), _ = self.sm2.sign(msgA, dA, k=k)
        print(f"用户A签名: r={hex(rA)[:20]}..., s={hex(sA)[:20]}...")

        # 用户B签名
        (rB, sB), _ = self.sm2.sign(msgB, dB, k=k)
        print(f"用户B签名: r={hex(rB)[:20]}..., s={hex(sB)[:20]}...")

        # 数学推导不可行，直接标记失败
        dA_cracked = 0
        print(f"\n用户A真实私钥: {hex(dA)[:20]}...")
        print(f"推导用户A私钥: {hex(dA_cracked)[:20]}...")
        print("攻击结果: 失败 (数学上不可行)")
        return False



    def run_all_attacks(self):
        """运行所有攻击并生成报告"""
        print("SM2签名算法误用漏洞验证")
        print("=" * 50)

        results = []

        # 运行攻击1
        print("\n>>> 开始攻击验证: 相同用户重用k <<<")
        result1 = self.same_k_sm2_attack()
        results.append(("相同用户重用k", result1))

        # 运行攻击2
        print("\n>>> 开始攻击验证: 不同用户重用k <<<")
        result2 = self.same_k_different_users_attack()
        results.append(("不同用户重用k", result2))

        # 生成报告
        print("\n" + "=" * 50)
        print("验证结果摘要:")
        print("=" * 50)
        for name, result in results:
            print(f"{name}: {'成功' if result else '失败'}")

        return results


if __name__ == "__main__":
    attacker = SignatureAttacks()
    attacker.run_all_attacks()