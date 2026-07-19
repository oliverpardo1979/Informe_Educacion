from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Paper" / "tables" / "remuneracion_educacion_comparable_summary.csv"
OUTPUT = ROOT / "Paper" / "figures" / "fig_remuneracion_educacion_niveles_2025.png"

WIDTH = 1700
HEIGHT = 1120

TEXT = "#111111"
MUTED = "#555555"
GRID = "#e5e7eb"
AXIS = "#444444"
YEAR_2010 = "#6b7280"
GROUP_COLORS = {
    "Primaria o menos": "#8b3ff2",
    "Secundaria o media": "#087db5",
    "Superior o universitaria": "#0a9d73",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fill: str,
    size: int,
    bold: bool = False,
    anchor: str = "la",
) -> None:
    draw.text(xy, text, fill=fill, font=font(size, bold), anchor=anchor)


def comma(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def read_data() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with DATA.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "group": row["grupo_educativo"],
                    "order": float(row["grupo_orden"]),
                    "hour_2010": float(row["rem_hora_2010"]) / 1000,
                    "hour_2025": float(row["rem_hora_2025"]) / 1000,
                    "worker_2010": float(row["rem_trabajador_2010"]) / 1_000_000,
                    "worker_2025": float(row["rem_trabajador_2025"]) / 1_000_000,
                }
            )
    return sorted(rows, key=lambda item: float(item["order"]))


def x_pos(value: float, left: int, right: int, xmin: float, xmax: float) -> float:
    return left + (value - xmin) / (xmax - xmin) * (right - left)


def draw_axis(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    top: int,
    bottom: int,
    ticks: list[float],
    xmin: float,
    xmax: float,
    tick_digits: int,
) -> None:
    for tick in ticks:
        x = x_pos(tick, left, right, xmin, xmax)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = str(int(tick)) if tick_digits == 0 else comma(tick, tick_digits)
        draw_text(draw, (x, bottom + 24), label, MUTED, 24, anchor="mt")
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)


def draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, filled: bool) -> None:
    radius = 10
    box = (x - radius, y - radius, x + radius, y + radius)
    if filled:
        draw.ellipse(box, fill=color, outline=color, width=3)
    else:
        draw.ellipse(box, fill="white", outline=color, width=4)


def draw_value_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: float,
    color: str,
    side: str,
) -> None:
    if side == "left":
        draw_text(draw, (x - 18, y - 4), comma(value), color, 28, True, anchor="rm")
    else:
        draw_text(draw, (x + 18, y - 4), comma(value), color, 28, True, anchor="lm")


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, float | str]],
    panel_top: int,
    title: str,
    subtitle: str,
    fields: tuple[str, str],
    ticks: list[float],
    xmax: float,
    tick_digits: int,
) -> None:
    left = 415
    right = 1530
    top = panel_top + 95
    bottom = panel_top + 410
    row_gap = 105
    first_y = top + 70

    draw_text(draw, (80, panel_top), title, TEXT, 38, True)
    draw_text(draw, (80, panel_top + 46), subtitle, MUTED, 28)
    draw_axis(draw, left, right, top, bottom, ticks, 0, xmax, tick_digits)

    for idx, row in enumerate(rows):
        group = str(row["group"])
        y = first_y + idx * row_gap
        color = GROUP_COLORS[group]
        value_2010 = float(row[fields[0]])
        value_2025 = float(row[fields[1]])
        x_2010 = x_pos(value_2010, left, right, 0, xmax)
        x_2025 = x_pos(value_2025, left, right, 0, xmax)
        draw_text(draw, (left - 26, y), group, TEXT, 28, True, anchor="rm")
        draw.line((x_2010, y, x_2025, y), fill=color, width=5)
        draw_point(draw, x_2010, y, YEAR_2010, False)
        draw_point(draw, x_2025, y, color, True)
        draw_value_label(draw, x_2010, y, value_2010, YEAR_2010, "left")
        draw_value_label(draw, x_2025, y, value_2025, color, "right")


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    y = 1045
    x0 = 1060
    draw_point(draw, x0, y, YEAR_2010, False)
    draw_text(draw, (x0 + 24, y + 1), "2010", TEXT, 25, anchor="lm")
    x1 = 1195
    draw_point(draw, x1, y, "#111111", True)
    draw_text(draw, (x1 + 24, y + 1), "2025", TEXT, 25, anchor="lm")


def main() -> None:
    rows = read_data()
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    draw_panel(
        draw,
        rows,
        40,
        "Panel A. Remuneración mensual por trabajador",
        "Millones de pesos mensuales de 2025",
        ("worker_2010", "worker_2025"),
        [0, 1, 2, 3, 4],
        4,
        0,
    )
    draw_panel(
        draw,
        rows,
        585,
        "Panel B. Remuneración por hora trabajada",
        "Miles de pesos de 2025 por hora",
        ("hour_2010", "hour_2025"),
        [0, 5, 10, 15, 20],
        20,
        0,
    )
    draw_text(draw, (80, 1045), "Fuente: cálculos propios con GEIH del DANE.", MUTED, 25)
    draw_legend(draw)
    img.save(OUTPUT, quality=95)


if __name__ == "__main__":
    main()
