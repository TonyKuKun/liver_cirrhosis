from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT = Path(r"E:\pycharm_code\liver_cirrhosis\geometry_feature_extract\outputs\clinical_segmentation_flowchart_simple.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 1600, 2440
img = Image.new("RGB", (W, H), "#f7fafc")
d = ImageDraw.Draw(img)

FONT = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


TITLE = font(48, True)
SUB = font(27)
BOX_TITLE = font(31, True)
TEXT = font(24)
SMALL = font(22)
DECISION = font(24, True)


def text_w(text, f):
    box = d.textbbox((0, 0), text, font=f)
    return box[2] - box[0]


def wrap(text, max_width, f):
    lines = []
    for para in text.split("\n"):
        current = ""
        for ch in para:
            trial = current + ch
            if text_w(trial, f) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def box(x, y, w, h, title, body, fill="#ffffff", outline="#cbd5e1", title_fill="#0f172a"):
    d.rounded_rectangle((x + 7, y + 9, x + w + 7, y + h + 9), radius=20, fill="#dbe3ee")
    d.rounded_rectangle((x, y, x + w, y + h), radius=20, fill=fill, outline=outline, width=3)
    d.text((x + 30, y + 22), title, font=BOX_TITLE, fill=title_fill)
    yy = y + 76
    for line in wrap(body, w - 60, TEXT):
        d.text((x + 30, yy), line, font=TEXT, fill="#1f2937")
        yy += 35


def diamond(x, y, w, h, text):
    cx, cy = x + w / 2, y + h / 2
    pts = [(cx, y), (x + w, cy), (cx, y + h), (x, cy)]
    d.polygon([(px + 7, py + 9) for px, py in pts], fill="#dbe3ee")
    d.polygon(pts, fill="#fff7ed", outline="#fb923c")
    d.line(pts + [pts[0]], fill="#fb923c", width=3)
    lines = wrap(text, w - 130, DECISION)
    total_h = len(lines) * 34
    yy = cy - total_h / 2
    for line in lines:
        d.text((cx - text_w(line, DECISION) / 2, yy), line, font=DECISION, fill="#7c2d12")
        yy += 34


def arrow(x1, y1, x2, y2, label=None):
    color = "#334155"
    d.line((x1, y1, x2, y2), fill=color, width=5)
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 20
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - math.pi / 6), y2 - head * math.sin(ang - math.pi / 6)),
        (x2 - head * math.cos(ang + math.pi / 6), y2 - head * math.sin(ang + math.pi / 6)),
    ]
    d.polygon(pts, fill=color)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        tw = text_w(label, SMALL)
        d.rounded_rectangle((mx - tw / 2 - 16, my - 24, mx + tw / 2 + 16, my + 14), radius=11, fill="#ffffff", outline="#cbd5e1")
        d.text((mx - tw / 2, my - 20), label, font=SMALL, fill=color)


def pill(x, y, text, fill, outline):
    tw = text_w(text, SMALL)
    d.rounded_rectangle((x, y, x + tw + 34, y + 42), radius=21, fill=fill, outline=outline, width=2)
    d.text((x + 17, y + 8), text, font=SMALL, fill="#0f172a")


def section_label(x, y, text, color):
    d.rounded_rectangle((x, y, x + 14, y + 42), radius=7, fill=color)
    d.text((x + 25, y + 4), text, font=SMALL, fill="#334155")


d.text((80, 54), "门静脉解剖分段：临床判断流程", font=TITLE, fill="#12335f")
d.text((82, 124), "直白版：先确定 MPV 的临床终点，再判断远端血管归属；TIPS 术后不把支架入口附近误延长为 MPV。", font=SUB, fill="#475569")

pill(80, 178, "核心边界：SV/SMV 汇合点到肝门首次左右门静脉分叉 = MPV", "#dbeafe", "#60a5fa")
pill(905, 178, "TIPS 术后：LPV/RPV/TIPS 交汇区不归入 MPV", "#ccfbf1", "#14b8a6")

cx = W // 2
left = 150
wide = 1300

box(
    left, 255, wide, 145,
    "1. 输入资料",
    "门静脉中心线树、患者坐标方向、SV/SMV 入口、是否存在 TIPS 支架。",
    "#eef6ff", "#93c5fd",
)
arrow(cx, 400, cx, 470)

box(
    left, 470, wide, 185,
    "2. 先定 MPV 起止点",
    "起点：脾静脉 SV 与肠系膜上静脉 SMV 汇合处。\n终点：进入肝门后，第一次明确分向左门静脉和右门静脉的位置。",
    "#ffffff", "#cbd5e1",
)
arrow(cx, 655, cx, 725)

diamond(cx - 330, 725, 660, 160, "病例是否存在 TIPS 支架？")
arrow(cx - 330, 805, 360, 965, "否")
arrow(cx + 330, 805, 1240, 965, "是")

box(
    80, 965, 560, 260,
    "3A. 常规门静脉路径",
    "按临床解剖分叉判断：\nMPV 只到左、右门静脉第一次分叉处。\n分叉之后，向患者左侧连续走行为 LPV，向患者右侧连续走行为 RPV。",
    "#f8fafc", "#94a3b8",
)

box(
    960, 965, 560, 260,
    "3B. TIPS 术后路径",
    "先单独识别人工支架 TIPS。\n再找支架门静脉端与肝门分叉交汇区。\nMPV 不能跨过交汇区；\n不追入 LPV/RPV 或支架入口。",
    "#e7fffb", "#14b8a6",
)

arrow(360, 1225, 360, 1325)
arrow(1240, 1225, 1240, 1325)

box(
    80, 1325, 560, 230,
    "4A. 普通病例的远端归属",
    "在 MPV 终点之后：\n左向、左肝内连续分支标为 LPV。\n右向、右肝内连续分支标为 RPV。",
    "#f0fdf4", "#22c55e",
)

box(
    960, 1325, 560, 230,
    "4B. TIPS 病例的远端归属",
    "若短段已超过 MPV 终点，\n且靠近支架入口或左右分支交汇区：\n不标 MPV；按走向归 LPV/RPV。\n支架本体 = TIPS。",
    "#fff1f2", "#fb7185",
    title_fill="#991b1b",
)

arrow(360, 1555, cx, 1665)
arrow(1240, 1555, cx, 1665)

diamond(cx - 380, 1665, 760, 185, "待判断血管段是否已超过 MPV 临床终点？")
arrow(cx - 380, 1758, 360, 1940, "否")
arrow(cx + 380, 1758, 1240, 1940, "是")

box(
    80, 1940, 560, 205,
    "5A. 仍属于 MPV",
    "血管段仍位于 SV/SMV 汇合点到肝门首次左右分叉之间。\n此时保留为 MPV。",
    "#eef6ff", "#60a5fa",
)

box(
    960, 1940, 560, 205,
    "5B. 不属于 MPV",
    "血管段已进入肝内左/右分支，\n或进入 TIPS 入口交汇区。\n按走向归 LPV/RPV；支架本体 = TIPS。",
    "#fef3c7", "#f59e0b",
)

arrow(360, 2145, cx, 2230)
arrow(1240, 2145, cx, 2230)

box(
    left, 2230, wide, 150,
    "6. 输出与质控",
    "输出 MPV、SV、SMV、LPV、RPV、TIPS。\n若左右分叉不清、支架附着异常或拓扑不典型，保留人工复核标志。",
    "#ffffff", "#cbd5e1",
)

section_label(80, 440, "临床边界优先，不按最长主干或视觉延长线决定 MPV", "#2563eb")
section_label(80, 1595, "红箭头这类肝门后段：超过 MPV 终点后，应归 LPV/RPV 或 TIPS", "#dc2626")

img.save(OUT, quality=95)
print(OUT)
