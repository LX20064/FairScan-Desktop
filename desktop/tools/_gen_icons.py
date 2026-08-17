"""生成 ① 16x16 齿轮 SVG path（设置按钮图标）② 应用图标 icon.ico/icon.png。
用法: python tools/_gen_icons.py
"""
import math
from PIL import Image, ImageDraw

CX, CY = 8.0, 8.0
R, r = 6.2, 4.3          # 齿顶半径 / 齿根半径
A1, A2 = 9.0, 14.0       # 齿顶半角 / 齿根半角（度）
N = 8                     # 齿数


def pt(angle_deg, radius):
    a = math.radians(angle_deg)
    return (CX + radius * math.cos(a), CY + radius * math.sin(a))


parts = []
start = (pt(22.5 - A1, R))
parts.append(f"M {start[0]:.2f} {start[1]:.2f}")
for k in range(N):
    th = 22.5 + 45 * k
    p1 = pt(th + A1, R)
    p2 = pt(th + A2, r)
    p3 = pt(th + 45 - A2, r)
    p4 = pt(th + 45 - A1, R)
    parts.append(f"A {R} {R} 0 0 1 {p1[0]:.2f} {p1[1]:.2f}")      # 齿顶弧
    parts.append(f"L {p2[0]:.2f} {p2[1]:.2f}")                    # 下降
    parts.append(f"A {r} {r} 0 0 1 {p3[0]:.2f} {p3[1]:.2f}")      # 齿根弧
    parts.append(f"L {p4[0]:.2f} {p4[1]:.2f}")                    # 上升
# 中心孔（evenodd 挖空）
parts.append("M 6.50 8.00 a 1.50 1.50 0 1 0 3.00 0 a 1.50 1.50 0 1 0 -3.00 0 z")
gear_path = " ".join(parts)
print("GEAR_PATH=")
print(gear_path)
print()

# ---- 应用图标：蓝底圆角方块 + 白色文档页 + 扫描线 ----
S = 256
BG = (0, 95, 184)          # --accent #005fb8
BG_TOP = (26, 115, 211)    # 顶部微亮
PAGE = (255, 255, 255)
SCAN = (30, 110, 200)


def draw_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 背景圆角方块（带一点垂直渐变：上亮下暗）
    radius = size * 0.22
    steps = 8
    for i in range(steps):
        y0 = size * i // steps
        y1 = size * (i + 1) // steps
        t = i / (steps - 1)
        col = tuple(int(BG_TOP[c] + (BG[c] - BG_TOP[c]) * t) for c in range(3))
        d.rounded_rectangle([0, y0, size, y1], radius=radius, fill=col + (255,))
    # 文档页：brand 字形 (1,2)(15,2)(15,13)(9,13)(7,15)(5,13)(1,13) 缩放
    s = size / 16.0
    ox, oy = 0.5 * s, 0.6 * s  # 页面居中（16 格内 0.5/0.6 偏移）
    poly = [(ox + 1 * s, oy + 2 * s), (ox + 15 * s, oy + 2 * s),
            (ox + 15 * s, oy + 13 * s), (ox + 9 * s, oy + 13 * s),
            (ox + 7 * s, oy + 15 * s), (ox + 5 * s, oy + 13 * s),
            (ox + 1 * s, oy + 13 * s)]
    d.polygon(poly, fill=PAGE + (255,))
    # 折角内线（微调出页角折痕）
    d.line([(ox + 9 * s, oy + 13 * s), (ox + 9 * s, oy + 15 * s),
            (ox + 7 * s, oy + 15 * s)], fill=BG + (255,), width=max(1, size // 96))
    # 扫描线（横贯页面的亮色横条）
    lw = max(1, size // 32)
    d.rounded_rectangle([ox + 2.5 * s, oy + 7.6 * s, ox + 13.5 * s, oy + 7.6 * s + lw],
                        radius=lw, fill=SCAN + (255,))
    return img


master = draw_icon(S)
master.save("src/icon.png")
master.save("src/icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("saved src/icon.png and src/icon.ico")
