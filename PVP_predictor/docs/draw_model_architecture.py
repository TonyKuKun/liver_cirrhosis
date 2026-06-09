from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_architecture() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 3400, 1900
    image = Image.new("RGB", (width, height), "#fbfcfd")
    draw = ImageDraw.Draw(image)

    f_title = font(56, True)
    f_sub = font(31)
    f_box = font(29, True)
    f_small = font(22)
    f_tiny = font(19)

    colors = {
        "ink": "#1b2633",
        "muted": "#5f6c7b",
        "blue": "#dcecff",
        "blue_line": "#4c84c4",
        "teal": "#dff5f2",
        "teal_line": "#35a89b",
        "green": "#e8f6df",
        "green_line": "#69a845",
        "amber": "#fff2d8",
        "amber_line": "#c49127",
        "rose": "#fde8ec",
        "rose_line": "#c55a6a",
        "gray": "#eef2f6",
        "gray_line": "#8a98a8",
    }

    def round_rect(x0, y0, x1, y1, fill, outline, radius=32, line_width=4):
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=radius,
            fill=fill,
            outline=outline,
            width=line_width,
        )

    def text_left(x, y, text, fnt, fill=colors["ink"], spacing=7):
        for line in text.split("\n"):
            draw.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + spacing

    def arrow(x0, y0, x1, y1, color="#536476", line_width=5):
        draw.line([x0, y0, x1, y1], fill=color, width=line_width)
        angle = math.atan2(y1 - y0, x1 - x0)
        length = 22
        spread = 0.55
        p1 = (
            x1 - length * math.cos(angle - spread),
            y1 - length * math.sin(angle - spread),
        )
        p2 = (
            x1 - length * math.cos(angle + spread),
            y1 - length * math.sin(angle + spread),
        )
        draw.polygon([(x1, y1), p1, p2], fill=color)

    def add_box(x, y, w, h, fill, outline, title, body):
        round_rect(x, y, x + w, y + h, fill, outline)
        draw.text((x + 26, y + 24), title, font=f_box, fill=colors["ink"])
        text_left(x + 26, y + 78, body, f_small, fill=colors["muted"], spacing=8)

    draw.text((120, 78), "PVP Predictor: current final architecture", font=f_title, fill=colors["ink"])
    draw.text(
        (122, 150),
        "8-vessel cross-section geometry + learnable hemodynamics + organ global context + one PVP head",
        font=f_sub,
        fill=colors["muted"],
    )

    columns = [120, 575, 1030, 1485, 1940, 2395, 2850]
    headers = [
        "Input",
        "Geometry\nfilter",
        "Learnable\nphysics",
        "Global flow\ncorrection",
        "Flow graph\nrefiner",
        "PVP head",
        "Training\nloss",
    ]
    for x, header in zip(columns, headers):
        text_left(x, 245, header, f_box, fill=colors["ink"], spacing=2)

    add_box(
        120,
        340,
        385,
        440,
        colors["blue"],
        colors["blue_line"],
        "Vessel sequences",
        "8 vessel branches\nMPV, SV, SMV\nLPV, RPV, TIPS\nLGV, PGV\n\nsegment mask\nCenterlinePoints-ready",
    )
    add_box(
        120,
        860,
        385,
        290,
        colors["teal"],
        colors["teal_line"],
        "Organ global state",
        "spleen volume\nliver volume\nspleen/liver ratio\n\nfrom largest connected\nSTL components",
    )
    add_box(
        575,
        450,
        385,
        560,
        colors["gray"],
        colors["gray_line"],
        "Reliable geometry",
        "area\nhydraulic diameter\ninscribed radius\ncurvature\nsolidity\ncircularity\ndA/ds norm\n\nDefault excludes noisy\nraw length/torsion\nand component counts",
    )
    add_box(
        1030,
        380,
        385,
        670,
        colors["amber"],
        colors["amber_line"],
        "Physics proxy layer",
        "effective radius\nrelative flow Q\nvelocity = Q / area\nwall shear proxy\nReynolds proxy\nDean proxy\nresistance proxy\npressure-drop proxy\n\nviscosity, radius exponent\nand pressure scale are\nlearnable unless ablated",
    )
    add_box(
        1485,
        410,
        385,
        590,
        colors["teal"],
        colors["teal_line"],
        "GlobalFlowCorrector",
        "uses filtered global\ngeometry and organ state\n\ntunes intermediate\nflow features\n\norgan volumes are context,\nnot hard Q constraints\n\npresence flags are excluded",
    )
    add_box(
        1940,
        435,
        385,
        540,
        colors["blue"],
        colors["blue_line"],
        "FlowGraphRefiner",
        "anatomical message passing\nbetween vessel branches\n\nkeeps the interface for\nCenterlinePoints graph data\n\ncurrent ablation shows\nsmall but positive effect\nfor the reference model",
    )
    add_box(
        2395,
        435,
        385,
        540,
        colors["green"],
        colors["green_line"],
        "Single PVP head",
        "aggregates corrected\nflow and physics features\nsegment mask\norgan global context\nphysics baseline state\n\noutputs portal venous\npressure in mmHg",
    )
    add_box(
        2850,
        380,
        385,
        660,
        colors["rose"],
        colors["rose_line"],
        "Objective",
        "L2 / MSE PVP loss\n+\noptional shunt loss\n\ncore_confluence:\nMPV ~= SMV + SV\n\nsingle-task PVP regression\none prediction head\nno extra auxiliary objectives",
    )

    arrow(505, 560, 575, 590)
    arrow(960, 720, 1030, 720)
    arrow(1415, 720, 1485, 720)
    arrow(1870, 720, 1940, 720)
    arrow(2325, 720, 2395, 720)
    arrow(2780, 720, 2850, 720)
    arrow(505, 1000, 1485, 850, color=colors["teal_line"])

    draw.line([315, 780, 315, 1260, 2465, 1260, 2465, 975], fill="#9aa7b5", width=4)
    arrow(2465, 1260, 2465, 975, color="#9aa7b5", line_width=4)
    draw.text(
        (690, 1288),
        "segment_mask blocks missing vessels; organ context calibrates patient-level flow state.",
        font=f_small,
        fill=colors["muted"],
    )

    round_rect(120, 1405, 1020, 1660, "#ffffff", colors["gray_line"], radius=28, line_width=3)
    draw.text((150, 1435), "Why this design", font=f_box, fill=colors["ink"])
    text_left(
        150,
        1495,
        "Use reliable cross-section geometry first.\nKeep uncertain raw vessel length out of hard formulas.\nUse organ volumes as global patient context instead of direct Q rescaling.",
        f_small,
        fill=colors["muted"],
    )

    round_rect(1090, 1405, 1990, 1660, "#ffffff", colors["gray_line"], radius=28, line_width=3)
    draw.text((1120, 1435), "2026-06-09 reference", font=f_box, fill=colors["ink"])
    text_left(
        1120,
        1495,
        "8 vessels + organ global features\nL2 + core_confluence shunt loss\nSubject-level 5-fold:\nMAE 3.153, RMSE 3.986, R2 0.585\nOOF bias -0.002 mmHg",
        f_small,
        fill=colors["muted"],
    )

    round_rect(2060, 1405, 3280, 1660, "#ffffff", colors["gray_line"], radius=28, line_width=3)
    draw.text((2090, 1435), "2026-06-09 ablation takeaways", font=f_box, fill=colors["ink"])
    text_left(
        2090,
        1495,
        "GlobalFlowCorrector and FlowGraphRefiner help the reference.\nFixed physics parameters, L2-only, and full shunt loss score better in this rerun.\nBest baseline: geometry/extra_trees, MAE 3.685, RMSE 4.550, R2 0.472.",
        f_small,
        fill=colors["muted"],
    )

    draw.text(
        (120, 1785),
        "PVP_predictor architecture figure | regenerated from 2026-06-09 experiments",
        font=f_tiny,
        fill="#8994a3",
    )

    out = OUT_DIR / "model_architecture.png"
    image.save(out, quality=95)
    return out


if __name__ == "__main__":
    print(draw_architecture())
