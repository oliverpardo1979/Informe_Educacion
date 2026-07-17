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
OUTPUT = FIG_DIR / "fig_kernel_remuneracion_credenciales_superiores_2021_2025.png"
STATS_OUTPUT = TABLE_DIR / "remuneracion_credenciales_superiores_kernel_stats.csv"

YEARS = (2021, 2025)
SMLMV_NOMINAL = {
    2021: 908_526,
    2025: 1_423_500,
}

TARGETS = {
    10: {
        "title_code": 7,
        "name": "Universitaria",
        "color": "#009e73",
    },
    11: {
        "title_code": 8,
        "name": "Especialización",
        "color": "#7f3c8d",
    },
    12: {
        "title_code": 9,
        "name": "Maestría",
        "color": "#d55e00",
    },
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


def weighted_kde(log_values: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float]:
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
    char_cols = KEY_COLS + ["P3042", "P3043"]

    for year in YEARS:
        for filename, _url in DOWNLOADS[year]:
            zip_path = RAW_DIR / filename
            if not zip_path.exists():
                raise FileNotFoundError(f"No se encontró {zip_path}")
            for pair in list_pairs(zip_path):
                with zipfile.ZipFile(zip_path) as zf:
                    char = read_selected_csv(zf, pair.characteristics, char_cols)
                    occ = read_selected_csv(zf, pair.occupied, OCC_COLS)

                if char.duplicated(KEY_COLS).any():
                    char = char.drop_duplicates(KEY_COLS, keep="first")

                merged = occ.merge(
                    char[KEY_COLS + ["P3042", "P3043"]],
                    on=KEY_COLS,
                    how="left",
                    validate="many_to_one",
                )

                merged["p3042"] = to_number(merged["P3042"])
                merged["p3043"] = to_number(merged["P3043"])
                merged["oci"] = to_number(merged["OCI"])
                merged["hours_week"] = to_number(merged["P6800"])
                merged["income_month"] = to_number(merged["INGLABO"])
                merged["weight"] = to_number(merged["FEX_C18"]) / 12

                valid = merged[
                    (merged["oci"] == 1)
                    & merged["p3042"].isin(TARGETS.keys())
                    & (merged["income_month"] > 0)
                    & (merged["hours_week"] > 0)
                    & (merged["hours_week"] <= 168)
                    & (merged["weight"] > 0)
                ].copy()
                if valid.empty:
                    continue

                valid["p3042_int"] = valid["p3042"].astype(int)
                valid["title_code"] = valid["p3042_int"].map(
                    lambda code: TARGETS[code]["title_code"]
                )
                valid = valid[valid["p3043"] == valid["title_code"]].copy()
                if valid.empty:
                    continue

                valid["credencial"] = valid["p3042_int"].map(lambda code: TARGETS[code]["name"])
                valid["anio"] = year
                valid["rem_mensual"] = valid["income_month"] * PRICE_FACTORS_2025[year]
                valid["rem_hora"] = valid["rem_mensual"] / (
                    valid["hours_week"] * MONTHLY_HOURS_FACTOR
                )

                rows.append(
                    valid[
                        [
                            "anio",
                            "credencial",
                            "weight",
                            "rem_mensual",
                            "rem_hora",
                        ]
                    ].rename(columns={"weight": "peso"})
                )

    if not rows:
        raise RuntimeError("No hay observaciones para construir los kernels.")
    return pd.concat(rows, ignore_index=True)


def build_stats(microdata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indicators = [
        ("rem_mensual", "Remuneración mensual por trabajador", 1.0),
        ("rem_hora", "Remuneración por hora trabajada", 1.0),
    ]
    for variable, label, scale in indicators:
        for credential in TARGETS.values():
            name = str(credential["name"])
            for year in YEARS:
                subset = microdata[
                    (microdata["credencial"] == name)
                    & (microdata["anio"] == year)
                ]
                values = subset[variable].to_numpy(dtype=float) / scale
                weights = subset["peso"].to_numpy(dtype=float)
                p10, p25, p50, p75, p90 = weighted_quantile(
                    values,
                    weights,
                    [0.10, 0.25, 0.50, 0.75, 0.90],
                )
                rows.append(
                    {
                        "indicador": label,
                        "credencial": name,
                        "anio": year,
                        "obs_sin_expandir": len(subset),
                        "ocupados_expandido": weights.sum(),
                        "promedio": np.average(values, weights=weights),
                        "p10": p10,
                        "p25": p25,
                        "mediana": p50,
                        "p75": p75,
                        "p90": p90,
                        "p90_p10": p90 / p10,
                        "desv_est_log": weighted_std(np.log(values), weights),
                    }
                )
    return pd.DataFrame(rows)


def build_kernels(microdata: pd.DataFrame) -> list[dict[str, object]]:
    specs = [
        {
            "key": "rem_mensual",
            "title": "Remuneración mensual por trabajador",
            "unit": "Millones de pesos mensuales de 2025",
            "scale": 1e6,
            "ticks": [0.5, 1, 2, 4, 8, 16, 32],
            "show_smlmv": True,
        },
        {
            "key": "rem_hora",
            "title": "Remuneración por hora trabajada",
            "unit": "Miles de pesos de 2025 por hora",
            "scale": 1e3,
            "ticks": [5, 10, 20, 40, 80, 160],
            "show_smlmv": False,
        },
    ]

    for spec in specs:
        key = str(spec["key"])
        scale = float(spec["scale"])
        scaled = microdata[key].to_numpy(dtype=float) / scale
        weights = microdata["peso"].to_numpy(dtype=float)
        support = weighted_quantile(np.log(scaled), weights, [0.005, 0.995])
        grid = np.linspace(float(support[0]), float(support[1]), 360)
        series = {}
        for credential in TARGETS.values():
            name = str(credential["name"])
            for year in YEARS:
                subset = microdata[
                    (microdata["credencial"] == name)
                    & (microdata["anio"] == year)
                ]
                values = subset[key].to_numpy(dtype=float) / scale
                subset_weights = subset["peso"].to_numpy(dtype=float)
                density, bandwidth = weighted_kde(np.log(values), subset_weights, grid)
                series[(name, year)] = {
                    "density": density,
                    "bandwidth": bandwidth,
                }
        spec["support"] = support
        spec["grid"] = grid
        spec["series"] = series
    return specs


def draw_figure(microdata: pd.DataFrame, stats: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_kernels(microdata)

    width, height = 2200, 1780
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw_text(
        draw,
        (80, 48),
        "Distribución de la remuneración por credencial de educación superior",
        "#111111",
        46,
        True,
    )
    draw_text(
        draw,
        (80, 106),
        "Ocupados con título o diploma reportado. 2021 y 2025. Pesos constantes de 2025.",
        "#555555",
        27,
    )
    draw_text(
        draw,
        (80, 144),
        "Kernels ponderados por factor de expansión sobre el log de la remuneración. Eje horizontal en escala logarítmica.",
        "#555555",
        24,
    )

    left_col = (130, 325, 1010, 625)
    right_col = (1240, 325, 2120, 625)
    row_step = 405
    boxes = []
    for row_idx, credential in enumerate(TARGETS.values()):
        y_shift = row_idx * row_step
        boxes.append(
            (
                str(credential["name"]),
                str(credential["color"]),
                [
                    (
                        left_col[0],
                        left_col[1] + y_shift,
                        left_col[2],
                        left_col[3] + y_shift,
                    ),
                    (
                        right_col[0],
                        right_col[1] + y_shift,
                        right_col[2],
                        right_col[3] + y_shift,
                    ),
                ],
            )
        )

    draw_text(draw, (left_col[0], 236), str(specs[0]["title"]), "#111111", 32, True)
    draw_text(draw, (left_col[0], 274), str(specs[0]["unit"]), "#555555", 22)
    draw_text(draw, (right_col[0], 236), str(specs[1]["title"]), "#111111", 32, True)
    draw_text(draw, (right_col[0], 274), str(specs[1]["unit"]), "#555555", 22)

    def x_for(value: float, left: int, right: int, support: np.ndarray) -> float:
        log_value = math.log(value)
        return left + (log_value - support[0]) / (support[1] - support[0]) * (right - left)

    def y_for(value: float, top: int, bottom: int, max_density: float) -> float:
        return bottom - (value / max_density) * (bottom - top)

    for credential, row_color, row_boxes in boxes:
        draw_text(
            draw,
            (left_col[0], row_boxes[0][1] - 37),
            credential,
            row_color,
            30,
            True,
        )
        for spec, box in zip(specs, row_boxes):
            left, top, right, bottom = box
            support = np.asarray(spec["support"], dtype=float)
            grid = np.asarray(spec["grid"], dtype=float)
            series = spec["series"]
            densities = [
                series[(credential, year)]["density"]
                for year in YEARS
            ]
            max_density = max(float(density.max()) for density in densities) * 1.12

            for level in np.linspace(0, max_density, 5):
                y = y_for(level, top, bottom, max_density)
                draw.line((left, y, right, y), fill="#eeeeee", width=1)

            for tick in spec["ticks"]:
                log_tick = math.log(float(tick))
                if support[0] <= log_tick <= support[1]:
                    x = x_for(float(tick), left, right, support)
                    draw.line((x, top, x, bottom), fill="#f2f2f2", width=1)
                    digits = 1 if float(tick) < 1 else 0
                    draw_text(
                        draw,
                        (x, bottom + 17),
                        fmt_decimal(float(tick), digits),
                        "#555555",
                        18,
                        anchor="mt",
                    )

            if bool(spec["show_smlmv"]):
                for year in YEARS:
                    real_smlmv = (
                        SMLMV_NOMINAL[year]
                        * PRICE_FACTORS_2025[year]
                        / float(spec["scale"])
                    )
                    log_smlmv = math.log(real_smlmv)
                    if support[0] <= log_smlmv <= support[1]:
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
                style = YEAR_STYLE[year]
                density = series[(credential, year)]["density"]
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
                    str(style["color"]),
                    5,
                    bool(style["dash"]),
                )

            stat_label = (
                "Remuneración mensual por trabajador"
                if spec["key"] == "rem_mensual"
                else "Remuneración por hora trabajada"
            )
            medians = stats[
                (stats["credencial"] == credential)
                & (stats["indicador"] == stat_label)
            ].set_index("anio")["mediana"]
            start_median = float(medians.loc[2021]) / float(spec["scale"])
            end_median = float(medians.loc[2025]) / float(spec["scale"])
            draw_text(
                draw,
                (right - 8, top + 12),
                f"Mediana {fmt_decimal(start_median, 1)} a {fmt_decimal(end_median, 1)}",
                "#333333",
                18,
                anchor="ra",
            )

    legend_y = 1582
    legend_x = 220
    for index, year in enumerate(YEARS):
        x = legend_x + index * 260
        style = YEAR_STYLE[year]
        draw_polyline(
            draw,
            [(x, legend_y), (x + 76, legend_y)],
            str(style["color"]),
            6,
            bool(style["dash"]),
        )
        draw_text(draw, (x + 92, legend_y), str(year), "#222222", 24, anchor="lm")

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
        (80, 1660),
        "Nota: se exige que P3042 reporte la credencial y que P3043 reporte el título o diploma del mismo nivel.",
        "#555555",
        21,
    )
    draw_text(
        draw,
        (80, 1692),
        "La remuneración mensual corresponde al ingreso laboral mensual. La remuneración por hora divide el ingreso mensual por horas mensuales.",
        "#555555",
        21,
    )
    draw_text(
        draw,
        (80, 1724),
        "Fuente: cálculos propios con microdatos mensuales de la GEIH marco 2018 del DANE.",
        "#555555",
        21,
    )

    img.save(OUTPUT, quality=95)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    microdata = load_microdata()
    stats = build_stats(microdata)
    stats.to_csv(STATS_OUTPUT, index=False)
    draw_figure(microdata, stats)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
