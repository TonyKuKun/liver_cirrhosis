from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_architecture() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 4200, 2350
    image = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(image)

    f_title = font(72, True)
    f_sub = font(34)
    f_panel = font(36, True)
    f_box = font(30, True)
    f_text = font(25)
    f_small = font(21)
    f_tiny = font(18)

    colors = {
        "ink": "#162033",
        "muted": "#5d6878",
        "line": "#778397",
        "navy": "#24476f",
        "blue": "#dceafe",
        "blue_line": "#3b73b9",
        "cyan": "#ddf4f6",
        "cyan_line": "#2a9cab",
        "green": "#e4f4e7",
        "green_line": "#4f9a5e",
        "yellow": "#fff1cf",
        "yellow_line": "#c48a1c",
        "pink": "#fbe3ea",
        "pink_line": "#bd5571",
        "violet": "#ece7fb",
        "violet_line": "#7560b3",
        "white": "#ffffff",
    }

    def rounded(x0, y0, x1, y1, fill, outline, radius=28, width=4):
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=width)

    def shadow_box(x, y, w, h, fill, outline, title, body, title_color=None, wrap=18):
        draw.rounded_rectangle([x + 8, y + 10, x + w + 8, y + h + 10], radius=30, fill="#dce2ec")
        rounded(x, y, x + w, y + h, fill, outline, radius=30, width=4)
        draw.text((x + 28, y + 24), title, font=f_box, fill=title_color or colors["ink"])
        yy = y + 82
        for raw_line in body.split("\n"):
            lines = textwrap.wrap(raw_line, width=wrap) if raw_line else [""]
            for line in lines:
                draw.text((x + 28, yy), line, font=f_text, fill=colors["muted"])
                yy += 34
            yy += 5

    def arrow(x0, y0, x1, y1, color=None, width_line=6):
        color = color or colors["line"]
        draw.line([x0, y0, x1, y1], fill=color, width=width_line)
        angle = math.atan2(y1 - y0, x1 - x0)
        size = 28
        spread = 0.55
        p1 = (x1 - size * math.cos(angle - spread), y1 - size * math.sin(angle - spread))
        p2 = (x1 - size * math.cos(angle + spread), y1 - size * math.sin(angle + spread))
        draw.polygon([(x1, y1), p1, p2], fill=color)

    def label(x, y, text, fnt=f_small, fill=None):
        draw.text((x, y), text, font=fnt, fill=fill or colors["muted"])

    # Background panels.
    draw.rectangle([0, 0, width, 310], fill="#eef3f8")
    draw.text((140, 70), "PVP Predictor 最终模型架构", font=f_title, fill=colors["ink"])
    draw.text(
        (145, 165),
        "单任务门静脉压力回归：8 条血管分支几何 + 可学习血流动力学代理 + 脾肝全局状态 + 单一 PVP 预测头",
        font=f_sub,
        fill=colors["muted"],
    )
    draw.text((145, 226), "Final 5-fold MAE 2.685 | RMSE 3.605 | R2 0.643", font=f_sub, fill=colors["navy"])

    # Section ribbons.
    ribbon_y = 350
    sections = [
        (145, "输入与质控", "#b8d4f5"),
        (820, "几何与物理编码", "#f5d58d"),
        (1760, "全局校正与图传播", "#aee2dc"),
        (2770, "单头预测与约束", "#e8bad1"),
    ]
    for x, txt, fill in sections:
        rounded(x, ribbon_y, x + 560, ribbon_y + 72, fill, "#ffffff", radius=22, width=2)
        tw = draw.textlength(txt, font=f_panel)
        draw.text((x + 280 - tw / 2, ribbon_y + 15), txt, font=f_panel, fill=colors["ink"])

    # Main boxes.
    y_main = 505
    shadow_box(
        120,
        y_main,
        530,
        430,
        colors["blue"],
        colors["blue_line"],
        "多分支血管输入",
        "MPV / SV / SMV\nLPV / RPV / TIPS\nLGV / PGV\n\n每条分支包含截面序列、弧长、中心线点与血管存在掩码。",
        wrap=18,
    )
    shadow_box(
        120,
        1045,
        530,
        300,
        colors["cyan"],
        colors["cyan_line"],
        "脾肝全局状态",
        "脾体积、肝体积、脾肝体积比。\n由 STL 最大连通域计算，用作患者级上下文。",
        wrap=18,
    )
    shadow_box(
        780,
        y_main,
        530,
        520,
        "#eef1f5",
        "#8b98aa",
        "可靠几何筛选",
        "默认仅保留稳定截面特征：面积、水力直径、内切半径、曲率、实心度、圆度、归一化 dA/ds。\n\n排除噪声较大的原始长度、扭率和连通域计数。",
        wrap=18,
    )
    shadow_box(
        1440,
        y_main,
        580,
        650,
        colors["yellow"],
        colors["yellow_line"],
        "可学习血流动力学代理",
        "根据几何估计有效半径、相对流量、截面速度、壁面剪切、Reynolds、Dean、阻力与压降等物理代理量。\n\n黏度缩放、半径指数和压力缩放可学习。",
        wrap=19,
    )
    shadow_box(
        2170,
        y_main,
        560,
        540,
        colors["cyan"],
        colors["cyan_line"],
        "GlobalFlowCorrector",
        "融合分支 embedding、物理特征、全局几何与脾肝状态，校正中间流量表征。\n\n脾肝特征作为上下文，不作为硬性流量比例。",
        wrap=18,
    )
    shadow_box(
        2860,
        y_main,
        530,
        500,
        colors["violet"],
        colors["violet_line"],
        "FlowGraphRefiner",
        "在解剖连接图上传播分支信息，并保留中心线图结构接口。\n\n消融：去掉该模块后 MAE 从 2.685 升至 3.171。",
        wrap=18,
    )
    shadow_box(
        3530,
        y_main,
        520,
        430,
        colors["green"],
        colors["green_line"],
        "单一 PVP 预测头",
        "聚合校正后的流量、物理代理、血管 mask、患者全局状态和物理基线。\n\n最终只输出一个门静脉压力值。",
        wrap=18,
    )

    # Bottom objective block.
    rounded(2145, 1195, 4050, 1535, colors["pink"], colors["pink_line"], radius=34, width=5)
    draw.text((2185, 1235), "训练目标：L2 PVP loss + 轻量 core_confluence 分流 loss", font=f_panel, fill=colors["ink"])
    label(2188, 1308, "L = MSE(PVP_pred, PVP_label) + 0.005 * ||Q_MPV - Q_SMV - Q_SV||^2", fnt=f_text, fill=colors["navy"])
    label(2188, 1363, "约束只用于训练；模型仍是单任务、单预测头。", fnt=f_text)
    label(2188, 1418, "PhysicsResidualNet 默认关闭：开启后 MAE=2.809，弱于最终模型。", fnt=f_text)

    # Flow arrows.
    arrow(650, 720, 780, 720)
    arrow(1310, 720, 1440, 720)
    arrow(2020, 720, 2170, 720)
    arrow(2730, 720, 2860, 720)
    arrow(3390, 720, 3530, 720)
    arrow(650, 1190, 2170, 960, color=colors["cyan_line"], width_line=5)
    arrow(3790, 935, 3790, 1195, color=colors["pink_line"], width_line=5)

    # Ablation strip.
    rounded(120, 1675, 4050, 2135, colors["white"], "#d2dae7", radius=34, width=3)
    draw.text((170, 1718), "关键实验结论", font=f_panel, fill=colors["ink"])
    ablation_items = [
        ("最终模型", "MAE 2.685", colors["green_line"]),
        ("纯 L2", "MAE 2.700", colors["blue_line"]),
        ("开启 residual", "MAE 2.809", colors["yellow_line"]),
        ("去掉 dropout 正则", "MAE 2.942", colors["violet_line"]),
        ("无脾肝全局特征", "MAE 3.104", colors["cyan_line"]),
        ("无全局流校正", "MAE 3.479", colors["pink_line"]),
        ("最佳传统 baseline", "MAE 3.420", colors["line"]),
    ]
    x = 170
    for title, metric, c in ablation_items:
        rounded(x, 1800, x + 500, 2050, "#fbfcfe", c, radius=26, width=4)
        draw.text((x + 28, 1840), title, font=f_box, fill=colors["ink"])
        draw.text((x + 28, 1912), metric, font=f_panel, fill=c)
        x += 550

    draw.text(
        (145, 2265),
        "图 1 | PVP Predictor 中文架构总览。结果来自 2026-06-10 subject-level 5-fold 实验。",
        font=f_tiny,
        fill=colors["muted"],
    )

    out = OUT_DIR / "model_architecture.png"
    image.save(out, quality=95)
    return out


if __name__ == "__main__":
    print(draw_architecture())
