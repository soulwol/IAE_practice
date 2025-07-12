from blind_watermark import WaterMark
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os


# ====== 1. 嵌入水印 ======
def embed_watermark(input_path='test_image.jpg', output_path='embedded.png', wm_text='SECRET'):
    bwm = WaterMark(password_img=1, password_wm=1)
    bwm.read_img(input_path)
    bwm.read_wm(wm_text, mode='str')
    bwm.embed(output_path)
    len_wm = len(bwm.wm_bit)
    print(f'水印长度: {len_wm}')
    return len_wm


# ====== 2. 提取水印 ======
def extract_watermark(img_path, wm_length, mode='str'):
    bwm = WaterMark(password_img=1, password_wm=1)
    return bwm.extract(img_path, wm_shape=wm_length, mode=mode)


# ====== 3. 图像攻击测试函数 ======
def test_robustness(original_path, wm_length, output_dir='attacked_images'):
    os.makedirs(output_dir, exist_ok=True)

    # 记录结果
    results = []

    # ====== 攻击函数组 ======
    def add_attack(name, func):
        nonlocal results
        attacked_path = f"{output_dir}/{name}.png"
        func(original_path, attacked_path)

        try:
            extracted = extract_watermark(attacked_path, wm_length)
            results.append((name, extracted))
            print(f"{name}: 提取结果 -> {extracted}")
        except Exception as e:
            results.append((name, f"提取失败: {str(e)}"))
            print(f"{name}: 提取失败 - {str(e)}")

    # ====== 具体攻击实现 ======

    # 水平翻转
    def flip_horizontal(in_path, out_path):
        img = Image.open(in_path)
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flipped.save(out_path)

    add_attack("水平翻转", flip_horizontal)

    # 垂直翻转
    def flip_vertical(in_path, out_path):
        img = Image.open(in_path)
        img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_flipped.save(out_path)

    add_attack("垂直翻转", flip_vertical)

    # 平移
    def shift_image(in_path, out_path, dx=50, dy=30):
        img = Image.open(in_path)
        img_np = np.array(img)
        shifted = np.roll(img_np, (dy, dx), axis=(0, 1))
        Image.fromarray(shifted).save(out_path)

    add_attack("平移攻击", shift_image)

    # 裁剪
    def crop_image(in_path, out_path, ratio=0.8):
        img = Image.open(in_path)
        w, h = img.size
        new_w, new_h = int(w * ratio), int(h * ratio)
        img_cropped = img.crop((0, 0, new_w, new_h))
        # 填充回原始尺寸
        img_padded = Image.new(img.mode, (w, h), color=(0, 0, 0))
        img_padded.paste(img_cropped, (0, 0))
        img_padded.save(out_path)

    add_attack("裁剪攻击", crop_image)

    # 对比度增强
    def enhance_contrast(in_path, out_path, factor=2.0):
        img = Image.open(in_path)
        enhancer = ImageEnhance.Contrast(img)
        img_contrast = enhancer.enhance(factor)
        img_contrast.save(out_path)

    add_attack("对比度增强", enhance_contrast)

    # 对比度减弱
    def reduce_contrast(in_path, out_path, factor=0.5):
        img = Image.open(in_path)
        enhancer = ImageEnhance.Contrast(img)
        img_contrast = enhancer.enhance(factor)
        img_contrast.save(out_path)

    add_attack("对比度减弱", reduce_contrast)

    # 亮度调整
    def adjust_brightness(in_path, out_path, factor=0.7):
        img = Image.open(in_path)
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(factor)
        img_bright.save(out_path)

    add_attack("亮度降低", adjust_brightness)

    # 颜色平衡破坏
    def color_shift(in_path, out_path):
        img = Image.open(in_path)
        r, g, b = img.split()
        # 增加红色通道强度
        r = r.point(lambda x: min(x * 1.5, 255))
        img_shifted = Image.merge('RGB', (r, g, b))
        img_shifted.save(out_path)

    add_attack("颜色偏移", color_shift)

    return results


# ====== 主程序 ======
if __name__ == "__main__":
    # 嵌入水印
    wm_text = 'SECRET'
    len_wm = embed_watermark(output_path='embedded.png', wm_text=wm_text)

    # 直接提取测试
    print("\n=== 直接提取测试 ===")
    extracted = extract_watermark('embedded.png', len_wm)
    print(f"原始水印: {wm_text}")
    print(f"提取结果: {extracted}")

    # 进行鲁棒性测试
    print("\n=== 鲁棒性测试开始 ===")
    test_results = test_robustness('embedded.png', len_wm)

    # 打印最终结果
    print("\n=== 最终测试结果 ===")
    for attack, result in test_results:
        print(f"{attack:<12} -> {result}")