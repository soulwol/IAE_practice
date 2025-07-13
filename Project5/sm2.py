import random
import hashlib
import time


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
        start_time = time.time()
        d = random.randint(1, self.N - 1)
        P = self.curve.point_multiply(d, self.G)
        end_time = time.time()
        gen_time = (end_time - start_time) * 1000
        return d, P, gen_time

    def sm3_hash(self, message, Z=None):
        """简化版SM3哈希函数"""
        if Z is not None:
            message = Z + message
        h = hashlib.sha256()  # 实际SM3更复杂，这里用SHA256简化
        h.update(message.encode('utf-8') if isinstance(message, str) else message)
        return int.from_bytes(h.digest(), 'big') % self.N

    def sign(self, message, d):
        """SM2签名"""
        start_time = time.time()
        # 生成随机数k
        k = random.randint(1, self.N - 1)
        # 计算(k * G)
        x1, y1 = self.curve.point_multiply(k, self.G)
        e = self.sm3_hash(message)
        r = (e + x1) % self.N
        # 如果r为0或r+k为N，重新选择k
        while r == 0 or r + k == self.N:
            k = random.randint(1, self.N - 1)
            x1, y1 = self.curve.point_multiply(k, self.G)
            e = self.sm3_hash(message)
            r = (e + x1) % self.N
        # 计算s
        s = (self.curve.inv_mod(1 + d, self.N) * (k - r * d)) % self.N
        if s == 0:
            return self.sign(message, d)  # 重新生成签名
        end_time = time.time()
        sign_time = (end_time - start_time) * 1000
        return (r, s), sign_time

    def verify(self, message, signature, P):
        """SM2验证签名"""
        start_time = time.time()
        r, s = signature
        # 验证r和s在[1, N-1]范围内
        if not (1 <= r <= self.N - 1) or not (1 <= s <= self.N - 1):
            return False, 0.0
        e = self.sm3_hash(message)
        # 计算t = (r + s) mod N
        t = (r + s) % self.N
        if t == 0:
            return False, 0.0
        # 计算s * G + t * P
        sG = self.curve.point_multiply(s, self.G)
        tP = self.curve.point_multiply(t, P)
        x1, y1 = self.curve.point_addition(sG, tP)
        R = (e + x1) % self.N
        end_time = time.time()
        verify_time = (end_time - start_time) * 1000
        return R == r, verify_time


def test_sm2(use_jacobi):
    sm2 = SM2(use_jacobi=use_jacobi)
    method = "优化版（雅可比坐标）" if use_jacobi else "原始版（仿射坐标）"

    # 生成密钥对
    print(f"{method}生成密钥对...")
    private_key, public_key, gen_time = sm2.generate_keypair()
    print(f"密钥生成时间: {gen_time:.4f} ms")

    # 消息
    message = "Hello, SM2 digital signature!"
    print(f"原始消息: {message}")

    # 签名
    print(f"\n{method}生成签名...")
    signature, sign_time = sm2.sign(message, private_key)
    print(f"签名结果: r={hex(signature[0])}, s={hex(signature[1])}")
    print(f"签名时间: {sign_time:.4f} ms")

    # 验证签名
    print(f"\n{method}验证签名...")
    valid, verify_time = sm2.verify(message, signature, public_key)
    print(f"验证结果: {'成功' if valid else '失败'}")
    print(f"验证时间: {verify_time:.4f} ms")

    return gen_time, sign_time, verify_time


if __name__ == "__main__":
    # 测试原始版本
    print("=" * 50)
    print("测试原始版本（仿射坐标）")
    orig_gen, orig_sign, orig_verify = test_sm2(use_jacobi=False)

    print("\n" + "=" * 50)
    print("测试优化版本（雅可比坐标）")
    opt_gen, opt_sign, opt_verify = test_sm2(use_jacobi=True)

    print("\n" + "=" * 50)
    print("性能对比：")
    print(f"密钥生成加速: {orig_gen / opt_gen:.2f}x ({orig_gen:.2f} ms -> {opt_gen:.2f} ms)")
    print(f"签名加速: {orig_sign / opt_sign:.2f}x ({orig_sign:.2f} ms -> {opt_sign:.2f} ms)")
    print(f"验证加速: {orig_verify / opt_verify:.2f}x ({orig_verify:.2f} ms -> {opt_verify:.2f} ms)")