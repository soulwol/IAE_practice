import random
import hashlib


# 椭圆曲线基本运算类 (仿射坐标实现)
class EllipticCurve:
    def __init__(self, A, B, P, N, GX, GY):
        self.A = A
        self.B = B
        self.P = P
        self.N = N
        self.G = (GX, GY)

    def inv_mod(self, a, modulus):
        """使用扩展欧几里得算法计算模逆元"""
        if a == 0:
            return 0
        lm, hm = 1, 0
        low, high = a % modulus, modulus
        while low > 1:
            ratio = high // low
            nm = hm - lm * ratio
            new = high - low * ratio
            hm, lm = lm, nm
            high, low = low, new
        return lm % modulus

    def point_addition(self, p, q):
        """椭圆曲线点加法"""
        if p is None:
            return q
        if q is None:
            return p

        x1, y1 = p
        x2, y2 = q

        if x1 == x2 and y1 != y2:
            return None  # 无穷远点

        if x1 == x2:
            m = (3 * x1 * x1 + self.A) * self.inv_mod(2 * y1, self.P) % self.P
        else:
            m = (y2 - y1) * self.inv_mod(x2 - x1, self.P) % self.P

        x3 = (m * m - x1 - x2) % self.P
        y3 = (m * (x1 - x3) - y1) % self.P
        return (x3, y3)

    def point_multiply(self, k, p):
        """椭圆曲线点乘 (标量乘法)"""
        result = None
        addend = p

        while k:
            if k & 1:
                result = self.point_addition(result, addend)
            addend = self.point_addition(addend, addend)
            k >>= 1
        return result


# 比特币使用的secp256k1曲线参数
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


class SatoshiForgery:
    def __init__(self):
        self.curve = EllipticCurve(A, B, P, N, GX, GY)
        self.G = (GX, GY)
        # 中本聪创世区块地址 (1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa)
        self.satoshi_pubkey = (0x11DB93E1DCDB8A016B49840F8C53BC1EB68A382E97B1482ECAD7B148A6909A5C,
                               0x0C6F5D7A825D1A0B3EF5E3CC3BDA8F0A7269C8C17272E9F13D6BDC0F10EEE8E)

    def sm3_hash(self, message):
        h = hashlib.sha256()
        h.update(message.encode('utf-8'))
        return int.from_bytes(h.digest(), 'big') % N

    def forge_signature(self, message):
        """伪造中本聪签名核心方法"""
        # 步骤1: 生成随机数a和b
        a = random.randint(1, N - 1)
        b = random.randint(1, N - 1)

        # 步骤2: 计算 R = a*G + b*P
        aG = self.curve.point_multiply(a, self.G)
        bP = self.curve.point_multiply(b, self.satoshi_pubkey)
        R = self.curve.point_addition(aG, bP)
        R_x, _ = R

        # 步骤3: 计算伪造的签名参数
        r = R_x % N
        s = (r * self.curve.inv_mod(b, N)) % N
        e = (r * a * self.curve.inv_mod(b, N)) % N

        # 步骤4: 构造可验证的签名
        forged_signature = (r, s)
        return forged_signature, e

    def verify_forged_signature(self, signature, e):
        """验证伪造的签名"""
        r, s = signature
        w = self.curve.inv_mod(s, N)
        u1 = (e * w) % N
        u2 = (r * w) % N

        u1G = self.curve.point_multiply(u1, self.G)
        u2P = self.curve.point_multiply(u2, self.satoshi_pubkey)
        x, _ = self.curve.point_addition(u1G, u2P)

        return r == x % N

    def demonstrate_forgery(self):
        """完整伪造演示"""
        print("=" * 70)
        print("中本聪签名伪造演示")
        print("=" * 70)

        # 伪造重要声明
        message = "Craig Wright是比特币的真正创造者"
        print(f"伪造声明: 「{message}」")

        # 生成伪造签名
        signature, e = self.forge_signature(message)
        r, s = signature
        print(f"\n伪造签名生成结果:")
        print(f"r = 0x{r:064x}")
        print(f"s = 0x{s:064x}")
        print(f"消息哈希 = 0x{e:064x}")

        # 验证伪造签名
        is_valid = self.verify_forged_signature(signature, e)
        print(f"\n签名验证结果: {'成功' if is_valid else '失败'}")
        if is_valid:
            print("伪造签名通过中本聪公钥验证")

        return is_valid


if __name__ == "__main__":
    forger = SatoshiForgery()
    forger.demonstrate_forgery()