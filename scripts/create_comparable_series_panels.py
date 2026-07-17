from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Paper" / "tables" / "remuneracion_educacion_comparable_series.csv"
OUTPUT = ROOT / "Paper" / "figures" / "fig_remuneracion_educacion_series_comparables.png"

START_YEAR = 2010
END_YEAR = 2025
MONTHS_PER_WEEK = 52.0 / 12.0

SMLMV_NOMINAL = {
    2010: 515_000,
    2011: 535_600,
    2012: 566_700,
    2013: 589_500,
    2014: 616_000,
    2015: 644_350,
    2016: 689_455,
    2017: 737_717,
    2018: 781_242,
    2019: 828_116,
    2020: 877_803,
    2021: 908_526,
    2022: 1_000_000,
    2023: 1_160_000,
    2024: 1_300_000,
    2025: 1_423_500,
}

IPC_DIC = {
    2010: 73.45,
    2011: 76.19,
    2012: 78.05,
    2013: 79.56,
    2014: 82.47,
    2015: 88.05,
    2016: 93.11,
    2017: 96.92,
    2018: 100.00,
    2019: 103.80,
    2020: 105.48,
    2021: 111.41,
    2022: 126.03,
    2023: 137.72,
    2024: 144.88,
    2025: 152.27,
}

GROUPS = [
    ("Total", "#222222"),
    ("Primaria o menos", "#8a3ffc"),
    ("Secundaria", "#0072b2"),
    ("Universitaria o superior", "#009e73"),
]

WORKER_COLOR = "#0072b2"
HOUR_COLOR = "#d55e00"
SMLMV_COLOR = "#555555"
TEXT = "#111111"
MUTED = "#555555"
GRID = "#e7e7e7"
AXIS = "#222222"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
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
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, fill=fill, font=font(size, bold), anchor=anchor)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int = 3,
    dash: int = 18,
    gap: int = 10,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux = dx / length
    uy = dy / length
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        draw.line(
            (
                x0 + ux * distance,
                y0 + uy * distance,
                x0 + ux * segment_end,
                y0 + uy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        distance += dash + gap


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int = 3,
) -> None:
    for start, end in zip(points[:-1], points[1:]):
        draw_dashed_line(draw, start, end, fill=fill, width=width, dash=18, gap=10)


def fmt_decimal(value: float, digits: int = 1) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def nice_step(span: float, target_ticks: int = 4) -> float:
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if raw <= step:
            return step
    return 10 * magnitude


def axis_bounds(values: pd.Series, target_ticks: int = 4) -> tuple[float, float, float]:
    min_value = float(values.min())
    max_value = float(values.max())
    span = max_value - min_value
    padding = max(span * 0.14, abs(max_value) * 0.03)
    step = nice_step(span + 2 * padding, target_ticks)
    lower = math.floor((min_value - padding) / step) * step
    upper = math.ceil((max_value + padding) / step) * step
    if lower == upper:
        upper = lower + step
    return lower, upper, step


def load_series() -> pd.DataFrame:
    series = pd.read_csv(DATA)
    total = (
        series.groupby("anio", as_index=False)[
            ["ocupados", "rem_total_mensual", "horas_mensuales"]
        ]
        .sum()
        .assign(
            grupo_educativo="Total",
            grupo_orden=0,
        )
    )
    total["rem_por_trabajador"] = total["rem_total_mensual"] / total["ocupados"]
    total["rem_por_hora"] = total["rem_total_mensual"] / total["horas_mensuales"]
    total["horas_semanales"] = (
        total["horas_mensuales"] / total["ocupados"] / MONTHS_PER_WEEK
    )
    total["participacion_empleo"] = 1.0
    return pd.concat([total, series], ignore_index=True)


def add_scaled_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["rem_trabajador_millones"] = data["rem_por_trabajador"] / 1e6
    data["rem_hora_miles"] = data["rem_por_hora"] / 1e3
    data["smlmv_millones"] = data["anio"].map(
        lambda year: SMLMV_NOMINAL[int(year)] * IPC_DIC[END_YEAR] / IPC_DIC[int(year)] / 1e6
    )
    return data


def draw_panel(
    draw: ImageDraw.ImageDraw,
    data: pd.DataFrame,
    box: tuple[int, int, int, int],
    group: str,
    group_color: str,
    panel_label: str,
) -> None:
    left, top, right, bottom = box
    plot_top = top + 78
    plot_bottom = bottom - 72
    plot_left = left + 76
    plot_right = right - 92

    worker_min, worker_max, worker_step = axis_bounds(
        pd.concat([data["rem_trabajador_millones"], data["smlmv_millones"]])
    )
    hour_min, hour_max, hour_step = axis_bounds(data["rem_hora_miles"])
    years = sorted(data["anio"].astype(int).tolist())
    tick_years = [2010, 2015, 2019, 2021, 2025]

    draw_text(draw, (left, top), panel_label, group_color, 28, True)
    draw_text(draw, (left + 58, top), group, TEXT, 28, True)
    draw.line((left, top + 38, right, top + 38), fill=group_color, width=4)

    tick = worker_min
    while tick <= worker_max + worker_step / 2:
        y = plot_bottom - (tick - worker_min) / (worker_max - worker_min) * (
            plot_bottom - plot_top
        )
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw_text(draw, (plot_left - 10, y), fmt_decimal(tick, 1), WORKER_COLOR, 18, anchor="rm")
        tick += worker_step

    tick = hour_min
    while tick <= hour_max + hour_step / 2:
        y = plot_bottom - (tick - hour_min) / (hour_max - hour_min) * (
            plot_bottom - plot_top
        )
        draw_text(draw, (plot_right + 10, y), fmt_decimal(tick, 1), HOUR_COLOR, 18, anchor="lm")
        tick += hour_step

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=AXIS, width=2)

    def x_pos(year: int) -> float:
        return plot_left + (year - START_YEAR) / (END_YEAR - START_YEAR) * (
            plot_right - plot_left
        )

    def y_worker(value: float) -> float:
        return plot_bottom - (value - worker_min) / (worker_max - worker_min) * (
            plot_bottom - plot_top
        )

    def y_hour(value: float) -> float:
        return plot_bottom - (value - hour_min) / (hour_max - hour_min) * (
            plot_bottom - plot_top
        )

    for year in tick_years:
        x = x_pos(year)
        draw.line((x, plot_bottom, x, plot_bottom + 7), fill=AXIS, width=2)
        draw_text(draw, (x, plot_bottom + 17), str(year), MUTED, 17, anchor="mt")

    worker_points = [
        (x_pos(int(row.anio)), y_worker(float(row.rem_trabajador_millones)))
        for row in data.itertuples(index=False)
    ]
    hour_points = [
        (x_pos(int(row.anio)), y_hour(float(row.rem_hora_miles)))
        for row in data.itertuples(index=False)
    ]
    smlmv_points = [
        (x_pos(int(row.anio)), y_worker(float(row.smlmv_millones)))
        for row in data.itertuples(index=False)
    ]

    draw.line(worker_points, fill=WORKER_COLOR, width=5)
    draw.line(hour_points, fill=HOUR_COLOR, width=5)
    draw_dashed_polyline(draw, smlmv_points, fill=SMLMV_COLOR, width=3)

    for x, y in worker_points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=WORKER_COLOR)
    for x, y in hour_points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=HOUR_COLOR)

    last = data.iloc[-1]
    draw_text(
        draw,
        (plot_right - 6, worker_points[-1][1] - 13),
        f"{fmt_decimal(float(last['rem_trabajador_millones']), 1)}",
        WORKER_COLOR,
        20,
        True,
        anchor="rm",
    )
    draw_text(
        draw,
        (plot_right - 6, hour_points[-1][1] + 20),
        f"{fmt_decimal(float(last['rem_hora_miles']), 1)}",
        HOUR_COLOR,
        20,
        True,
        anchor="rm",
    )


def main() -> None:
    series = add_scaled_columns(load_series())
    width = 1900
    height = 1480
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw_text(draw, (80, 48), "Remuneración laboral por logro educativo", TEXT, 48, True)
    draw_text(
        draw,
        (80, 105),
        "2010--2025. Pesos constantes de 2025. La serie excluye 2020.",
        MUTED,
        27,
    )
    draw_text(
        draw,
        (80, 144),
        "Eje izquierdo: remuneración mensual y SMLMV en millones. Eje derecho: remuneración por hora en miles.",
        MUTED,
        25,
    )

    boxes = [
        (80, 220, 910, 720),
        (1030, 220, 1860, 720),
        (80, 805, 910, 1305),
        (1030, 805, 1860, 1305),
    ]
    panel_labels = ["A.", "B.", "C.", "D."]
    for (group, color), box, label in zip(GROUPS, boxes, panel_labels):
        data = series[series["grupo_educativo"] == group].sort_values("anio")
        draw_panel(draw, data, box, group, color, label)

    legend_y = 1376
    x = 190
    draw.line((x, legend_y, x + 68, legend_y), fill=WORKER_COLOR, width=7)
    draw_text(draw, (x + 84, legend_y), "Remuneración mensual por trabajador", TEXT, 25, anchor="lm")
    x = 705
    draw_dashed_line(draw, (x, legend_y), (x + 68, legend_y), fill=SMLMV_COLOR, width=5, dash=18, gap=8)
    draw_text(draw, (x + 84, legend_y), "SMLMV", TEXT, 25, anchor="lm")
    x = 925
    draw.line((x, legend_y, x + 68, legend_y), fill=HOUR_COLOR, width=7)
    draw_text(draw, (x + 84, legend_y), "Remuneración por hora trabajada", TEXT, 25, anchor="lm")

    draw_text(
        draw,
        (80, 1440),
        "Fuente: cálculos propios con GEIH del DANE y decretos del Gobierno nacional para SMLMV.",
        MUTED,
        22,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, quality=95)
    print(f"Figura guardada en {OUTPUT}")


if __name__ == "__main__":
    main()
