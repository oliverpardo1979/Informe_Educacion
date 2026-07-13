from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    REPO_ROOT.parent
    / "CJC-Monitor"
    / "Datos"
    / "Processed"
    / "Paper-GEIH_base_modelo_personas_2008_2025.dta"
)
TABLE_DIR = REPO_ROOT / "Paper" / "tables"
SECTION_DIR = REPO_ROOT / "Paper" / "sections"
FIG_DIR = REPO_ROOT / "Paper" / "figures"

YEARS = list(range(2010, 2020)) + list(range(2021, 2026))
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
    2008: 69.80,
    2009: 71.20,
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


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
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
    dash: int = 18,
    gap: int = 10,
) -> None:
    for start, end in zip(points[:-1], points[1:]):
        draw_dashed_line(draw, start, end, fill=fill, width=width, dash=dash, gap=gap)


def fmt_decimal(value: float, digits: int = 1) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_percent(value: float, digits: int = 2) -> str:
    return f"{fmt_decimal(100 * value, digits)}\\%"


def annual_growth(start: float, end: float) -> float:
    return (end / start) ** (1 / (END_YEAR - START_YEAR)) - 1


def is_higher_education(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"superior o universitaria", "universitaria o superior"}
    return float(value) == 6


def nice_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if raw <= step:
            return step
    return 10 * magnitude


def axis_bounds(values: pd.Series, target_ticks: int = 6) -> tuple[float, float, float]:
    min_value = float(values.min())
    max_value = float(values.max())
    span = max_value - min_value
    padding = max(span * 0.12, abs(max_value) * 0.03)
    step = nice_step(span + 2 * padding, target_ticks)
    lower = math.floor((min_value - padding) / step) * step
    upper = math.ceil((max_value + padding) / step) * step
    if lower == upper:
        upper = lower + step
    return lower, upper, step


def build_series() -> pd.DataFrame:
    columns = ["anio", "edad", "fex", "horas", "ingreso_hora_real", "educ_hom_cod"]
    totals: dict[int, dict[str, float]] = {}
    reader = pd.read_stata(DATA_PATH, columns=columns, convert_categoricals=False, chunksize=250_000)
    for chunk in reader:
        chunk = chunk[chunk["anio"].isin(YEARS)].copy()
        for column in ["anio", "edad", "fex", "horas", "ingreso_hora_real"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=columns)
        chunk = chunk[
            (chunk["edad"] >= 25)
            & (chunk["edad"] <= 29)
            & (chunk["fex"] > 0)
            & (chunk["horas"] > 0)
            & (chunk["horas"] <= 112)
            & (chunk["ingreso_hora_real"] > 0)
        ].copy()
        if chunk.empty:
            continue
        chunk = chunk[chunk["educ_hom_cod"].map(is_higher_education)].copy()
        if chunk.empty:
            continue
        chunk["rem_mensual"] = chunk["ingreso_hora_real"] * chunk["horas"] * MONTHS_PER_WEEK
        chunk["horas_mensuales"] = chunk["horas"] * MONTHS_PER_WEEK
        chunk["rem_total"] = chunk["rem_mensual"] * chunk["fex"]
        chunk["horas_total"] = chunk["horas_mensuales"] * chunk["fex"]
        for year, group in chunk.groupby(chunk["anio"].astype(int)):
            current = totals.setdefault(
                year,
                {
                    "ocupados": 0.0,
                    "rem_total": 0.0,
                    "horas_total": 0.0,
                },
            )
            current["ocupados"] += float(group["fex"].sum())
            current["rem_total"] += float(group["rem_total"].sum())
            current["horas_total"] += float(group["horas_total"].sum())

    rows = []
    for year in sorted(totals):
        values = totals[year]
        rows.append(
            {
                "anio": year,
                "ocupados": values["ocupados"],
                "rem_por_trabajador": values["rem_total"] / values["ocupados"],
                "rem_por_hora": values["rem_total"] / values["horas_total"],
            }
        )
    return pd.DataFrame(rows)


def write_table(series: pd.DataFrame) -> None:
    start = series[series["anio"] == START_YEAR].iloc[0]
    end = series[series["anio"] == END_YEAR].iloc[0]
    worker_growth = annual_growth(start["rem_por_trabajador"], end["rem_por_trabajador"])
    hour_growth = annual_growth(start["rem_por_hora"], end["rem_por_hora"])
    occupied_growth = annual_growth(start["ocupados"], end["ocupados"])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Remuneración laboral de ocupados de 25 a 29 años con educación universitaria o superior, 2010 y 2025}",
        r"\label{tab:remuneracion_recien_graduados}",
        r"\footnotesize",
        r"\begin{tabular}{@{}p{6.6cm}rrr@{}}",
        r"\toprule",
        r"Indicador & 2010 & 2025 & Crec. anual \\",
        r"\midrule",
        (
            "Ocupados (millones de personas) & "
            f"{fmt_decimal(start['ocupados'] / 1e6, 2)} & "
            f"{fmt_decimal(end['ocupados'] / 1e6, 2)} & "
            f"{fmt_percent(occupied_growth)} \\\\"
        ),
        (
            "Remuneración mensual por trabajador (millones de pesos de 2025) & "
            f"{fmt_decimal(start['rem_por_trabajador'] / 1e6, 2)} & "
            f"{fmt_decimal(end['rem_por_trabajador'] / 1e6, 2)} & "
            f"{fmt_percent(worker_growth)} \\\\"
        ),
        (
            "Remuneración por hora trabajada (miles de pesos de 2025) & "
            f"{fmt_decimal(start['rem_por_hora'] / 1e3, 1)} & "
            f"{fmt_decimal(end['rem_por_hora'] / 1e3, 1)} & "
            f"{fmt_percent(hour_growth)} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption*{\footnotesize Nota: la GEIH no identifica si la persona acaba de graduarse ni el año de graduación. "
            r"Este cuadro aproxima la remuneración de recién graduados con personas ocupadas "
            r"de 25 a 29 años que reportan educación universitaria o superior. "
            r"Por lo tanto, debe leerse como una aproximación a jóvenes con educación superior, no como graduados observados directamente. "
            r"El crecimiento es anualizado para 2010--2025. "
            r"La serie excluye 2020. Fuente: cálculos propios con GEIH del DANE.}"
        ),
        r"\end{table}",
        "",
    ]
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    (SECTION_DIR / "remuneracion_educacion_recien_graduados_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def draw_series(series: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = series.sort_values("anio").copy()
    data["rem_trabajador_millones"] = data["rem_por_trabajador"] / 1e6
    data["rem_hora_miles"] = data["rem_por_hora"] / 1e3
    data["smlmv_millones"] = data["anio"].map(
        lambda year: SMLMV_NOMINAL[int(year)] * IPC_DIC[END_YEAR] / IPC_DIC[int(year)] / 1e6
    )

    worker_color = "#0072b2"
    hour_color = "#d55e00"
    smlmv_color = "#555555"
    accent_color = "#009e73"
    years = sorted(data["anio"].unique())
    x_min = min(years)
    x_max = max(years)
    tick_years = [2010, 2012, 2015, 2019, 2021, 2025]

    worker_min, worker_max, worker_step = axis_bounds(
        pd.concat([data["rem_trabajador_millones"], data["smlmv_millones"]])
    )
    hour_min, hour_max, hour_step = axis_bounds(data["rem_hora_miles"])

    width, height = 1600, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 145, 190, 1315, 735

    draw_text(draw, (85, 50), "Remuneración laboral: 25 a 29 años con educación universitaria o superior", "#111111", 38, True)
    draw_text(draw, (85, 98), "Pesos constantes de 2025. La serie excluye 2020", "#444444", 26)
    draw.line((85, 138, 1515, 138), fill=accent_color, width=5)
    draw.rectangle((left, top, right, bottom), outline="#222222", width=2)
    draw_text(draw, (left, top - 44), "Eje izquierdo: rem. mensual y SMLMV (millones)", worker_color, 24, True)
    draw_text(draw, (right, top - 44), "Eje derecho: rem. por hora trabajada (miles)", hour_color, 24, True, "ra")

    tick = worker_min
    while tick <= worker_max + worker_step / 2:
        y = bottom - (tick - worker_min) / (worker_max - worker_min) * (bottom - top)
        draw.line((left, y, right, y), fill="#e8e8e8", width=1)
        draw_text(draw, (left - 14, y), fmt_decimal(tick, 2), worker_color, 20, anchor="rm")
        tick += worker_step

    tick = hour_min
    while tick <= hour_max + hour_step / 2:
        y = bottom - (tick - hour_min) / (hour_max - hour_min) * (bottom - top)
        draw_text(draw, (right + 14, y), fmt_decimal(tick, 1), hour_color, 20, anchor="lm")
        tick += hour_step

    for year in tick_years:
        x = left + (year - x_min) / (x_max - x_min) * (right - left)
        draw.line((x, bottom, x, bottom + 8), fill="#222222", width=2)
        draw_text(draw, (x, bottom + 18), str(year), "#555555", 20, anchor="mt")

    def x_pos(year: int) -> float:
        return left + (year - x_min) / (x_max - x_min) * (right - left)

    def y_worker(value: float) -> float:
        return bottom - (value - worker_min) / (worker_max - worker_min) * (bottom - top)

    def y_hour(value: float) -> float:
        return bottom - (value - hour_min) / (hour_max - hour_min) * (bottom - top)

    worker_points = [(x_pos(int(row.anio)), y_worker(row.rem_trabajador_millones)) for row in data.itertuples(index=False)]
    hour_points = [(x_pos(int(row.anio)), y_hour(row.rem_hora_miles)) for row in data.itertuples(index=False)]
    smlmv_points = [(x_pos(int(row.anio)), y_worker(row.smlmv_millones)) for row in data.itertuples(index=False)]

    draw.line(worker_points, fill=worker_color, width=6)
    draw.line(hour_points, fill=hour_color, width=6)
    draw_dashed_polyline(draw, smlmv_points, fill=smlmv_color, width=4, dash=22, gap=12)
    for x, y in worker_points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=worker_color)
    for x, y in hour_points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=hour_color)

    last = data.iloc[-1]
    draw_text(
        draw,
        (worker_points[-1][0] - 14, worker_points[-1][1] - 34),
        f"Trabajador: {fmt_decimal(last['rem_trabajador_millones'], 2)}",
        worker_color,
        22,
        True,
        "rm",
    )
    draw_text(
        draw,
        (hour_points[-1][0] - 14, hour_points[-1][1] + 34),
        f"Hora: {fmt_decimal(last['rem_hora_miles'], 1)}",
        hour_color,
        22,
        True,
        "rm",
    )

    legend_y = 820
    draw.line((145, legend_y, 205, legend_y), fill=worker_color, width=7)
    draw_text(draw, (220, legend_y), "Remuneración mensual por trabajador", "#222222", 24, anchor="lm")
    draw_dashed_line(draw, (670, legend_y), (730, legend_y), fill=smlmv_color, width=5, dash=18, gap=8)
    draw_text(draw, (745, legend_y), "SMLMV", "#222222", 24, anchor="lm")
    draw.line((945, legend_y, 1005, legend_y), fill=hour_color, width=7)
    draw_text(draw, (1020, legend_y), "Remuneración por hora trabajada", "#222222", 24, anchor="lm")
    draw_text(draw, (145, 885), "Fuente: cálculos propios con GEIH del DANE y decretos del Gobierno nacional para SMLMV.", "#555555", 21)

    img.save(FIG_DIR / "fig_remuneracion_recien_graduados_series.png", quality=95)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    series = build_series()
    series.to_csv(TABLE_DIR / "remuneracion_recien_graduados_series.csv", index=False)
    start = series[series["anio"] == START_YEAR].iloc[0]
    end = series[series["anio"] == END_YEAR].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "indicador": "ocupados",
                "valor_2010": start["ocupados"],
                "valor_2025": end["ocupados"],
                "crecimiento_anual": annual_growth(start["ocupados"], end["ocupados"]),
            },
            {
                "indicador": "rem_por_trabajador",
                "valor_2010": start["rem_por_trabajador"],
                "valor_2025": end["rem_por_trabajador"],
                "crecimiento_anual": annual_growth(start["rem_por_trabajador"], end["rem_por_trabajador"]),
            },
            {
                "indicador": "rem_por_hora",
                "valor_2010": start["rem_por_hora"],
                "valor_2025": end["rem_por_hora"],
                "crecimiento_anual": annual_growth(start["rem_por_hora"], end["rem_por_hora"]),
            },
        ]
    )
    summary.to_csv(TABLE_DIR / "remuneracion_recien_graduados_summary.csv", index=False)
    write_table(series)
    draw_series(series)


if __name__ == "__main__":
    main()
