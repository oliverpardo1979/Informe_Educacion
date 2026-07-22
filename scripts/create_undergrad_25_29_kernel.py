from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_detailed_education_from_microdata import (  # noqa: E402
    DOWNLOADS,
    KEY_COLS,
    MONTHLY_HOURS_FACTOR,
    OCC_COLS,
    PRICE_FACTORS_2025,
    RAW_DIR,
    list_pairs,
    read_selected_csv,
    to_number,
)

FIG_DIR = ROOT / "Paper" / "figures"
TABLE_DIR = ROOT / "Paper" / "tables"
SECTION_DIR = ROOT / "Paper" / "sections"

FIG_OUTPUT = FIG_DIR / "fig_kernel_pregrado_universitario_25_29_2021_2025.png"
STATS_OUTPUT = TABLE_DIR / "remuneracion_pregrado_universitario_25_29_stats.csv"
TABLE_OUTPUT = SECTION_DIR / "remuneracion_pregrado_universitario_25_29_table.tex"

YEARS = (2021, 2025)
SMLMV_NOMINAL = {
    2021: 908_526,
    2025: 1_423_500,
}

YEAR_STYLE = {
    2021: {"color": "#6f50ff", "dash": True},
    2025: {"color": "#0072b2", "dash": False},
}


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


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int,
    dashed: bool,
) -> None:
    if dashed:
        for start, end in zip(points[:-1], points[1:]):
            draw_dashed_line(draw, start, end, fill=fill, width=width, dash=16, gap=10)
    else:
        draw.line(points, fill=fill, width=width)


def fmt_decimal(value: float, digits: int = 1) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%".replace(".", ",")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return np.interp(np.asarray(probs) * cumulative[-1], cumulative, values)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    average = np.average(values, weights=weights)
    return math.sqrt(np.average((values - average) ** 2, weights=weights))


def weighted_kde(
    log_values: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_values = np.asarray(log_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    std = weighted_std(log_values, weights)
    effective_n = 1.0 / np.sum(weights**2)
    bandwidth = 1.06 * std * (effective_n ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 0.15
    density = np.zeros_like(grid, dtype=float)
    normalizer = 1 / (math.sqrt(2 * math.pi) * bandwidth)
    for start in range(0, len(log_values), 6000):
        chunk_values = log_values[start : start + 6000]
        chunk_weights = weights[start : start + 6000]
        z = (grid[:, None] - chunk_values[None, :]) / bandwidth
        density += normalizer * np.exp(-0.5 * z * z).dot(chunk_weights)
    return density, bandwidth


def load_microdata() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    char_cols = KEY_COLS + ["P3042", "P3043", "P6040"]

    for year in YEARS:
        for filename, _url in DOWNLOADS[year]:
            zip_path = RAW_DIR / filename
            if not zip_path.exists():
                raise FileNotFoundError(f"No se encontro {zip_path}")
            for pair in list_pairs(zip_path):
                with zipfile.ZipFile(zip_path) as zf:
                    char = read_selected_csv(zf, pair.characteristics, char_cols)
                    occ = read_selected_csv(zf, pair.occupied, OCC_COLS)

                if char.duplicated(KEY_COLS).any():
                    char = char.drop_duplicates(KEY_COLS, keep="first")

                merged = occ.merge(
                    char[KEY_COLS + ["P3042", "P3043", "P6040"]],
                    on=KEY_COLS,
                    how="left",
                    validate="many_to_one",
                )
                merged["p3042"] = to_number(merged["P3042"])
                merged["p3043"] = to_number(merged["P3043"])
                merged["edad"] = to_number(merged["P6040"])
                merged["oci"] = to_number(merged["OCI"])
                merged["hours_week"] = to_number(merged["P6800"])
                merged["income_month"] = to_number(merged["INGLABO"])
                merged["weight"] = to_number(merged["FEX_C18"]) / 12

                valid = merged[
                    (merged["oci"] == 1)
                    & (merged["p3042"] == 10)
                    & (merged["p3043"] == 7)
                    & (merged["edad"] >= 25)
                    & (merged["edad"] <= 29)
                    & (merged["income_month"] > 0)
                    & (merged["hours_week"] > 0)
                    & (merged["hours_week"] <= 168)
                    & (merged["weight"] > 0)
                ].copy()
                if valid.empty:
                    continue

                valid["anio"] = year
                valid["rem_mensual"] = valid["income_month"] * PRICE_FACTORS_2025[year]
                valid["rem_hora"] = valid["rem_mensual"] / (
                    valid["hours_week"] * MONTHLY_HOURS_FACTOR
                )
                rows.append(
                    valid[["anio", "edad", "weight", "rem_mensual", "rem_hora"]].rename(
                        columns={"weight": "peso"}
                    )
                )

    if not rows:
        raise RuntimeError("No hay observaciones para construir el kernel")
    return pd.concat(rows, ignore_index=True)


def build_stats(microdata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("rem_mensual", "Remuneración mensual", 1.0),
        ("rem_hora", "Remuneración por hora", 1.0),
    ]
    for variable, label, scale in specs:
        for year in YEARS:
            subset = microdata[microdata["anio"] == year]
            values = subset[variable].to_numpy(dtype=float) / scale
            weights = subset["peso"].to_numpy(dtype=float)
            p10, p25, p50, p75, p90 = weighted_quantile(
                values,
                weights,
                [0.10, 0.25, 0.50, 0.75, 0.90],
            )
            row = {
                "indicador": label,
                "anio": year,
                "obs_sin_expandir": len(subset),
                "ocupados_expandido": weights.sum(),
                "promedio": np.average(values, weights=weights),
                "mediana": p50,
                "desv_est": weighted_std(values, weights),
                "p10": p10,
                "p25": p25,
                "p75": p75,
                "p90": p90,
                "p90_p10": p90 / p10,
                "desv_est_log": weighted_std(np.log(values), weights),
            }
            if variable == "rem_mensual":
                smlmv = SMLMV_NOMINAL[year] * PRICE_FACTORS_2025[year]
                close = (values >= 0.9 * smlmv) & (values <= 1.1 * smlmv)
                row["masa_0_9_1_1_smlmv"] = weights[close].sum() / weights.sum()
            else:
                row["masa_0_9_1_1_smlmv"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def write_table(stats: pd.DataFrame) -> None:
    monthly = stats[stats["indicador"] == "Remuneración mensual"].set_index("anio")
    hourly = stats[stats["indicador"] == "Remuneración por hora"].set_index("anio")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Remuneración de ocupados de 25 a 29 años con pregrado universitario, 2021 y 2025}",
        r"\label{tab:remuneracion_pregrado_universitario_25_29}",
        r"\footnotesize",
        r"\begin{tabular}{@{}p{7.4cm}rr@{}}",
        r"\toprule",
        r"Indicador & 2021 & 2025 \\",
        r"\midrule",
        r"\multicolumn{3}{@{}l}{\textbf{Panel A. Remuneración mensual por trabajador}} \\",
        (
            "Ocupados (miles)"
            + f" & {fmt_decimal(monthly.loc[2021, 'ocupados_expandido'] / 1_000, 1)}"
            + f" & {fmt_decimal(monthly.loc[2025, 'ocupados_expandido'] / 1_000, 1)}"
            + r" \\"
        ),
        (
            "Promedio (millones de pesos de 2025)"
            + f" & {fmt_decimal(monthly.loc[2021, 'promedio'] / 1_000_000, 1)}"
            + f" & {fmt_decimal(monthly.loc[2025, 'promedio'] / 1_000_000, 1)}"
            + r" \\"
        ),
        (
            "Mediana (millones de pesos de 2025)"
            + f" & {fmt_decimal(monthly.loc[2021, 'mediana'] / 1_000_000, 1)}"
            + f" & {fmt_decimal(monthly.loc[2025, 'mediana'] / 1_000_000, 1)}"
            + r" \\"
        ),
        (
            "Desviación estándar (millones de pesos de 2025)"
            + f" & {fmt_decimal(monthly.loc[2021, 'desv_est'] / 1_000_000, 1)}"
            + f" & {fmt_decimal(monthly.loc[2025, 'desv_est'] / 1_000_000, 1)}"
            + r" \\"
        ),
        (
            "Razón P90/P10"
            + f" & {fmt_decimal(monthly.loc[2021, 'p90_p10'], 1)}"
            + f" & {fmt_decimal(monthly.loc[2025, 'p90_p10'], 1)}"
            + r" \\"
        ),
        (
            "Entre 0,9 y 1,1 SMLMV"
            + f" & {fmt_percent(monthly.loc[2021, 'masa_0_9_1_1_smlmv'], 1)}"
            + f" & {fmt_percent(monthly.loc[2025, 'masa_0_9_1_1_smlmv'], 1)}"
            + r" \\"
        ),
        r"\addlinespace[0.6em]",
        r"\multicolumn{3}{@{}l}{\textbf{Panel B. Remuneración por hora trabajada}} \\",
        (
            "Promedio (miles de pesos de 2025)"
            + f" & {fmt_decimal(hourly.loc[2021, 'promedio'] / 1_000, 1)}"
            + f" & {fmt_decimal(hourly.loc[2025, 'promedio'] / 1_000, 1)}"
            + r" \\"
        ),
        (
            "Mediana (miles de pesos de 2025)"
            + f" & {fmt_decimal(hourly.loc[2021, 'mediana'] / 1_000, 1)}"
            + f" & {fmt_decimal(hourly.loc[2025, 'mediana'] / 1_000, 1)}"
            + r" \\"
        ),
        (
            "Desviación estándar (miles de pesos de 2025)"
            + f" & {fmt_decimal(hourly.loc[2021, 'desv_est'] / 1_000, 1)}"
            + f" & {fmt_decimal(hourly.loc[2025, 'desv_est'] / 1_000, 1)}"
            + r" \\"
        ),
        (
            "Razón P90/P10"
            + f" & {fmt_decimal(hourly.loc[2021, 'p90_p10'], 1)}"
            + f" & {fmt_decimal(hourly.loc[2025, 'p90_p10'], 1)}"
            + r" \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption*{\footnotesize Nota: la muestra se restringe a ocupados de 25 a 29 años "
            r"con \texttt{P3042} igual a universitaria y \texttt{P3043} igual a título universitario. "
            r"El SMLMV se expresa en pesos constantes de 2025 y no incluye auxilio de transporte. "
            r"Fuente: cálculos propios con microdatos mensuales de la GEIH marco 2018 del DANE.}"
        ),
        r"\end{table}",
        "",
    ]
    TABLE_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def draw_figure(microdata: pd.DataFrame, stats: pd.DataFrame) -> None:
    specs = [
        {
            "key": "rem_mensual",
            "title": "Remuneración mensual por trabajador",
            "unit": "Millones de pesos de 2025",
            "scale": 1e6,
            "ticks": [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12],
            "show_smlmv": True,
        },
        {
            "key": "rem_hora",
            "title": "Remuneración por hora trabajada",
            "unit": "Miles de pesos de 2025 por hora",
            "scale": 1e3,
            "ticks": [4, 6, 8, 10, 15, 20, 30, 50, 80],
            "show_smlmv": False,
        },
    ]
    for spec in specs:
        scaled_all = microdata[str(spec["key"])].to_numpy(dtype=float) / float(spec["scale"])
        weights_all = microdata["peso"].to_numpy(dtype=float)
        support = weighted_quantile(np.log(scaled_all), weights_all, [0.005, 0.995])
        grid = np.linspace(float(support[0]), float(support[1]), 360)
        series = {}
        for year in YEARS:
            subset = microdata[microdata["anio"] == year]
            values = subset[str(spec["key"])].to_numpy(dtype=float) / float(spec["scale"])
            weights = subset["peso"].to_numpy(dtype=float)
            density, bandwidth = weighted_kde(np.log(values), weights, grid)
            series[year] = {"density": density, "bandwidth": bandwidth}
        spec["support"] = support
        spec["grid"] = grid
        spec["series"] = series

    width, height = 1700, 1000
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_text(
        draw,
        (80, 48),
        "Distribución de la remuneración: pregrado universitario 25-29 años",
        "#111111",
        40,
        True,
    )
    draw_text(
        draw,
        (80, 102),
        "Ocupados con credencial y título universitario reportado. 2021 y 2025. Pesos constantes de 2025.",
        "#555555",
        24,
    )
    draw_text(
        draw,
        (80, 136),
        "Distribuciones ponderadas por factor de expansión sobre el log de la remuneración. Eje horizontal en escala logarítmica.",
        "#555555",
        22,
    )

    boxes = [(115, 250, 790, 720), (930, 250, 1605, 720)]

    def x_for(value: float, left: int, right: int, support: np.ndarray) -> float:
        return left + (math.log(value) - support[0]) / (support[1] - support[0]) * (right - left)

    def y_for(value: float, top: int, bottom: int, max_density: float) -> float:
        return bottom - (value / max_density) * (bottom - top)

    for spec, box in zip(specs, boxes):
        left, top, right, bottom = box
        support = np.asarray(spec["support"], dtype=float)
        grid = np.asarray(spec["grid"], dtype=float)
        max_density = max(
            float(spec["series"][year]["density"].max()) for year in YEARS
        ) * 1.12
        draw_text(draw, (left, top - 74), str(spec["title"]), "#111111", 28, True)
        draw_text(draw, (left, top - 40), str(spec["unit"]), "#555555", 21)
        for grid_value in np.linspace(0, max_density, 5):
            y = y_for(float(grid_value), top, bottom, max_density)
            draw.line((left, y, right, y), fill="#e9e9e9", width=1)
        for tick in spec["ticks"]:
            log_tick = math.log(float(tick))
            if support[0] <= log_tick <= support[1]:
                x = x_for(float(tick), left, right, support)
                draw.line((x, top, x, bottom), fill="#f1f1f1", width=1)
                digits = 1 if float(tick) < 2 else 0
                draw_text(
                    draw,
                    (x, bottom + 18),
                    fmt_decimal(float(tick), digits),
                    "#555555",
                    18,
                    anchor="mt",
                )
        if bool(spec["show_smlmv"]):
            for year in YEARS:
                real_smlmv = SMLMV_NOMINAL[year] * PRICE_FACTORS_2025[year] / float(
                    spec["scale"]
                )
                if support[0] <= math.log(real_smlmv) <= support[1]:
                    x = x_for(real_smlmv, left, right, support)
                    draw_dashed_line(
                        draw,
                        (x, top),
                        (x, bottom),
                        YEAR_STYLE[year]["color"],
                        3,
                        dash=14,
                        gap=10,
                    )
        draw.line((left, bottom, right, bottom), fill="#444444", width=2)
        draw.line((left, top, left, bottom), fill="#444444", width=2)
        for year in YEARS:
            density = spec["series"][year]["density"]
            points = []
            for grid_value, density_value in zip(grid, density):
                x = left + (grid_value - support[0]) / (support[1] - support[0]) * (
                    right - left
                )
                y = y_for(float(density_value), top, bottom, max_density)
                points.append((x, y))
            draw_polyline(
                draw,
                points,
                YEAR_STYLE[year]["color"],
                5,
                bool(YEAR_STYLE[year]["dash"]),
            )
        med = stats[
            stats["indicador"].str.contains(
                "mensual" if spec["key"] == "rem_mensual" else "hora"
            )
        ].set_index("anio")["mediana"]
        med_text = (
            f"Mediana: {fmt_decimal(float(med.loc[2021]) / float(spec['scale']), 1)} "
            f"a {fmt_decimal(float(med.loc[2025]) / float(spec['scale']), 1)}"
        )
        draw_text(draw, (left, bottom + 58), med_text, "#333333", 21)

    legend_y = 835
    legend_x = 215
    for index, year in enumerate(YEARS):
        x = legend_x + index * 260
        draw_polyline(
            draw,
            [(x, legend_y), (x + 74, legend_y)],
            YEAR_STYLE[year]["color"],
            6,
            bool(YEAR_STYLE[year]["dash"]),
        )
        draw_text(draw, (x + 90, legend_y), str(year), "#222222", 24, anchor="lm")
    smlmv_x = legend_x + 560
    for index, year in enumerate(YEARS):
        x = smlmv_x + index * 285
        draw_dashed_line(
            draw,
            (x, legend_y - 22),
            (x, legend_y + 22),
            YEAR_STYLE[year]["color"],
            4,
            dash=14,
            gap=9,
        )
        draw_text(draw, (x + 28, legend_y), f"SMLMV {year}", "#222222", 24, anchor="lm")
    draw_text(
        draw,
        (80, 925),
        "Nota: se exige P3042 = universitaria, P3043 = título universitario y edad de 25 a 29 años.",
        "#555555",
        21,
    )
    draw_text(
        draw,
        (80, 957),
        "Fuente: cálculos propios con microdatos mensuales de la GEIH marco 2018 del DANE.",
        "#555555",
        21,
    )
    img.save(FIG_OUTPUT, quality=95)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    microdata = load_microdata()
    stats = build_stats(microdata)
    stats.to_csv(STATS_OUTPUT, index=False)
    write_table(stats)
    draw_figure(microdata, stats)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
