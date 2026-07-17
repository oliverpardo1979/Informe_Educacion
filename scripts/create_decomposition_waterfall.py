from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Paper" / "tables" / "remuneracion_educacion_descomposicion.csv"
OUTPUT = ROOT / "Paper" / "figures" / "fig_descomposicion_remuneracion_educacion_cascada.png"

WIDTH = 1700
HEIGHT = 1100

BLUE = "#3479a8"
ORANGE = "#c97834"
DARK = "#333333"
GRID = "#e6e6e6"
AXIS = "#444444"
TEXT = "#111111"
MUTED = "#555555"


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


def text_bbox(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=6)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fill: str,
    size: int,
    bold: bool = False,
    anchor: str = "la",
    align: str = "left",
) -> None:
    fnt = font(size, bold)
    x, y = xy
    if "\n" in text and anchor != "la":
        width, height = text_bbox(draw, text, fnt)
        if anchor in {"mm", "mt"}:
            x -= width / 2
        if anchor == "mm":
            y -= height / 2
        elif anchor == "rm":
            x -= width
            y -= height / 2
        draw.multiline_text((x, y), text, fill=fill, font=fnt, align=align, spacing=6)
        return
    draw.multiline_text(
        (x, y),
        text,
        fill=fill,
        font=fnt,
        anchor=anchor,
        align=align,
        spacing=6,
    )


def comma(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def dot_thousands(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def signed_number(value: float, mode: str) -> str:
    sign = "+" if value >= 0 else "-"
    absolute = abs(value)
    if mode == "worker":
        rounded = round(absolute / 10) * 10
        return f"{sign}{int(rounded)}"
    return f"{sign}{dot_thousands(absolute)}"


def read_data() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with DATA.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["indicador_key"]
            rows[key] = {
                "total": float(row["cambio_total"]),
                "productivity": float(row["cambio_niveles"]),
                "education": float(row["cambio_composicion"]),
                "share_productivity": float(row["participacion_niveles"]),
                "share_education": float(row["participacion_composicion"]),
            }
    return rows


def y_pos(value: float, top: int, bottom: int, ymax: float) -> float:
    return bottom - (value / ymax) * (bottom - top)


def draw_axis(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    top: int,
    bottom: int,
    ymax: float,
    ticks: list[float],
    tick_formatter,
) -> None:
    for tick in ticks:
        y = y_pos(tick, top, bottom, ymax)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw_text(draw, (left - 18, y), tick_formatter(tick), MUTED, 27, anchor="rm")
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    base: float,
    value: float,
    top: int,
    bottom: int,
    ymax: float,
    color: str,
) -> tuple[float, float]:
    y0 = y_pos(base, top, bottom, ymax)
    y1 = y_pos(base + value, top, bottom, ymax)
    draw.rectangle((x0, min(y0, y1), x1, max(y0, y1)), fill=color)
    return y0, y1


def draw_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: float,
    share: float | None,
    mode: str,
) -> None:
    line1 = signed_number(value, mode)
    if share is None:
        label = line1
    else:
        label = f"{line1}\n({comma(share * 100)}%)"
    draw_text(draw, (x, y), label, TEXT, 30, True, anchor="mm", align="center")


def draw_panel(
    draw: ImageDraw.ImageDraw,
    panel_top: int,
    title: str,
    subtitle: str,
    values: dict[str, float],
    mode: str,
    ymax: float,
    ticks: list[float],
    tick_formatter,
) -> None:
    left = 195
    right = 1550
    top = panel_top + 96
    bottom = panel_top + 390
    bar_w = 225
    xs = [500, 920, 1340]

    draw_text(draw, (130, panel_top), title, TEXT, 39, True)
    draw_text(draw, (130, panel_top + 45), subtitle, MUTED, 29)
    draw_axis(draw, left, right, top, bottom, ymax, ticks, tick_formatter)

    productivity = values["productivity"]
    education = values["education"]
    total = values["total"]

    first_top = draw_bar(draw, xs[0] - bar_w // 2, xs[0] + bar_w // 2, 0, productivity, top, bottom, ymax, BLUE)[1]
    second_base = productivity
    second_top = draw_bar(draw, xs[1] - bar_w // 2, xs[1] + bar_w // 2, second_base, education, top, bottom, ymax, ORANGE)[1]
    total_top = draw_bar(draw, xs[2] - bar_w // 2, xs[2] + bar_w // 2, 0, total, top, bottom, ymax, DARK)[1]

    draw.line(
        (
            xs[0] + bar_w // 2,
            y_pos(productivity, top, bottom, ymax),
            xs[1] - bar_w // 2,
            y_pos(productivity, top, bottom, ymax),
        ),
        fill="#9a9a9a",
        width=2,
    )
    draw.line(
        (
            xs[1] + bar_w // 2,
            y_pos(total, top, bottom, ymax),
            xs[2] - bar_w // 2,
            y_pos(total, top, bottom, ymax),
        ),
        fill="#9a9a9a",
        width=2,
    )

    draw_label(draw, xs[0], first_top - 48, productivity, values["share_productivity"], mode)
    draw_label(draw, xs[1], second_top - 48, education, values["share_education"], mode)
    draw_label(draw, xs[2], total_top - 46, total, None, mode)

    labels = [
        "Mayor productividad\nde cada logro",
        "Mayor logro educativo\nde la poblaci\u00f3n ocupada",
        "Cambio total",
    ]
    for x, label in zip(xs, labels):
        draw_text(draw, (x, bottom + 31), label, DARK, 29, anchor="mt", align="center")


def main() -> None:
    data = read_data()
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    worker = data["trabajador"].copy()
    worker = {key: value * 1000 if key in {"total", "productivity", "education"} else value for key, value in worker.items()}

    hour = data["hora"].copy()
    hour = {key: value * 1000 if key in {"total", "productivity", "education"} else value for key, value in hour.items()}

    draw_panel(
        draw,
        35,
        "Panel A. Remuneraci\u00f3n mensual por trabajador",
        "Variaci\u00f3n en miles de pesos mensuales de 2025",
        worker,
        "worker",
        500,
        [0, 100, 200, 300, 400, 500],
        lambda value: str(int(value)),
    )

    draw_panel(
        draw,
        555,
        "Panel B. Remuneraci\u00f3n por hora trabajada",
        "Variaci\u00f3n en pesos de 2025 por hora",
        hour,
        "hour",
        3200,
        [0, 800, 1600, 2400, 3200],
        dot_thousands,
    )

    draw_text(draw, (80, HEIGHT - 54), "Fuente: c\u00e1lculos propios con GEIH del DANE.", MUTED, 26)
    img.save(OUTPUT)


if __name__ == "__main__":
    main()
