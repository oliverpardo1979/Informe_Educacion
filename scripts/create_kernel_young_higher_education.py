from __future__ import annotations

from pathlib import Path
import math

import numpy as np
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
FIG_DIR = REPO_ROOT / "Paper" / "figures"
TABLE_DIR = REPO_ROOT / "Paper" / "tables"

START_YEAR = 2010
END_YEAR = 2025
MONTHS_PER_WEEK = 52.0 / 12.0

SMLMV_NOMINAL = {
    2010: 515_000,
    2025: 1_423_500,
}

IPC_DIC = {
    2010: 73.45,
    2025: 152.27,
}

YEAR_STYLE = {
    START_YEAR: {"color": "#6f50ff", "dash": True, "label": "2010"},
    END_YEAR: {"color": "#0072b2", "dash": False, "label": "2025"},
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


def fmt_pesos(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def is_higher_education(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"superior o universitaria", "universitaria o superior"}
    return float(value) == 6


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
    columns = ["anio", "edad", "fex", "horas", "ingreso_hora_real", "educ_hom_cod"]
    parts: list[pd.DataFrame] = []
    reader = pd.read_stata(
        DATA_PATH,
        columns=columns,
        convert_categoricals=False,
        chunksize=250_000,
    )
    for chunk in reader:
        chunk = chunk[chunk["anio"].isin([START_YEAR, END_YEAR])].copy()
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
        parts.append(
            pd.DataFrame(
                {
                    "anio": chunk["anio"].astype(int),
                    "peso": chunk["fex"].astype(float),
                    "rem_hora": chunk["ingreso_hora_real"].astype(float),
                    "rem_mensual": (
                        chunk["ingreso_hora_real"] * chunk["horas"] * MONTHS_PER_WEEK
                    ).astype(float),
                }
            )
        )

    if not parts:
        raise RuntimeError("No observations found for the requested group.")
    return pd.concat(parts, ignore_index=True)


def build_summary(microdata: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("rem_mensual", "Remuneración mensual equivalente", 1.0),
        ("rem_hora", "Remuneración por hora trabajada", 1.0),
    ]
    rows = []
    for variable, label, scale in specs:
        for year in [START_YEAR, END_YEAR]:
            subset = microdata[microdata["anio"] == year]
            values = subset[variable].to_numpy(dtype=float) / scale
            weights = subset["peso"].to_numpy(dtype=float)
            p10, p25, p50, p75, p90 = weighted_quantile(values, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
            rows.append(
                {
                    "indicador": label,
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


def prepare_kernel_series(microdata: pd.DataFrame, variable: str, scale: float) -> dict[str, object]:
    scaled = microdata[variable].to_numpy(dtype=float) / scale
    weights = microdata["peso"].to_numpy(dtype=float)
    support = weighted_quantile(np.log(scaled), weights, [0.005, 0.995])
    grid = np.linspace(float(support[0]), float(support[1]), 360)
    densities: dict[int, np.ndarray] = {}
    bandwidths: dict[int, float] = {}
    for year in [START_YEAR, END_YEAR]:
        subset = microdata[microdata["anio"] == year]
        values = subset[variable].to_numpy(dtype=float) / scale
        year_weights = subset["peso"].to_numpy(dtype=float)
        densities[year], bandwidths[year] = weighted_kde(np.log(values), year_weights, grid)
    return {
        "support": support,
        "grid": grid,
        "densities": densities,
        "bandwidths": bandwidths,
    }


def draw_kernel_figure(microdata: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "key": "rem_mensual",
            "title": "Remuneración mensual equivalente",
            "unit": "Millones de pesos de 2025",
            "scale": 1e6,
            "ticks": [0.3, 0.5, 1, 2, 4, 8, 16],
            "show_smlmv": True,
        },
        {
            "key": "rem_hora",
            "title": "Remuneración por hora trabajada",
            "unit": "Miles de pesos de 2025 por hora",
            "scale": 1e3,
            "ticks": [2, 3, 5, 10, 20, 40, 80],
            "show_smlmv": False,
        },
    ]
    for spec in specs:
        spec["kernel"] = prepare_kernel_series(microdata, spec["key"], spec["scale"])

    width, height = 1700, 1040
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw_text(
        draw,
        (80, 48),
        "Distribución de la remuneración: jóvenes con educación universitaria o superior",
        "#111111",
        41,
        True,
    )
    draw_text(
        draw,
        (80, 102),
        "Ocupados de 25 a 29 años. 2010 y 2025. Pesos constantes de 2025.",
        "#555555",
        25,
    )
    draw_text(
        draw,
        (80, 137),
        "Kernels ponderados por factor de expansión sobre el log de la remuneración. Eje horizontal en escala logarítmica.",
        "#555555",
        23,
    )

    boxes = [(115, 250, 790, 720), (930, 250, 1605, 720)]

    def x_for(value: float, left: int, right: int, support: np.ndarray) -> float:
        return left + (math.log(value) - support[0]) / (support[1] - support[0]) * (right - left)

    def y_for(value: float, top: int, bottom: int, max_density: float) -> float:
        return bottom - (value / max_density) * (bottom - top)

    for spec, box in zip(specs, boxes):
        left, top, right, bottom = box
        kernel = spec["kernel"]
        support = kernel["support"]
        grid = kernel["grid"]
        densities = kernel["densities"]
        max_density = max(float(density.max()) for density in densities.values()) * 1.10

        draw_text(draw, (left, top - 74), spec["title"], "#111111", 28, True)
        draw_text(draw, (left, top - 40), spec["unit"], "#555555", 21)

        for grid_value in np.linspace(0, max_density, 5):
            y = y_for(grid_value, top, bottom, max_density)
            draw.line((left, y, right, y), fill="#e9e9e9", width=1)

        for tick in spec["ticks"]:
            log_tick = math.log(float(tick))
            if support[0] <= log_tick <= support[1]:
                x = x_for(float(tick), left, right, support)
                draw.line((x, top, x, bottom), fill="#f1f1f1", width=1)
                digits = 1 if tick < 1 else 0
                draw_text(draw, (x, bottom + 18), fmt_decimal(float(tick), digits), "#555555", 18, anchor="mt")

        if spec["show_smlmv"]:
            for year in [START_YEAR, END_YEAR]:
                real_smlmv = SMLMV_NOMINAL[year] * IPC_DIC[END_YEAR] / IPC_DIC[year] / spec["scale"]
                log_smlmv = math.log(real_smlmv)
                if support[0] <= log_smlmv <= support[1]:
                    x = x_for(float(real_smlmv), left, right, support)
                    draw_dashed_line(draw, (x, top), (x, bottom), YEAR_STYLE[year]["color"], 3, dash=14, gap=10)

        draw.line((left, bottom, right, bottom), fill="#444444", width=2)
        draw.line((left, top, left, bottom), fill="#444444", width=2)

        for year in [START_YEAR, END_YEAR]:
            style = YEAR_STYLE[year]
            points = []
            density = densities[year]
            for grid_value, density_value in zip(grid, density):
                x = left + (grid_value - support[0]) / (support[1] - support[0]) * (right - left)
                y = y_for(float(density_value), top, bottom, max_density)
                points.append((x, y))
            draw_polyline(draw, points, style["color"], 5, bool(style["dash"]))

        variable_summary = summary[summary["indicador"].str.contains("mensual" if spec["key"] == "rem_mensual" else "hora")]
        start_median = variable_summary[variable_summary["anio"] == START_YEAR]["mediana"].iloc[0] / spec["scale"]
        end_median = variable_summary[variable_summary["anio"] == END_YEAR]["mediana"].iloc[0] / spec["scale"]
        median_text = (
            f"Mediana: {fmt_decimal(start_median, 2 if spec['key'] == 'rem_mensual' else 1)} "
            f"→ {fmt_decimal(end_median, 2 if spec['key'] == 'rem_mensual' else 1)}"
        )
        draw_text(draw, (left, bottom + 58), median_text, "#333333", 21)

    legend_y = 835
    legend_x = 215
    for index, year in enumerate([START_YEAR, END_YEAR]):
        x = legend_x + index * 260
        style = YEAR_STYLE[year]
        draw_polyline(draw, [(x, legend_y), (x + 74, legend_y)], style["color"], 6, bool(style["dash"]))
        draw_text(draw, (x + 90, legend_y), str(year), "#222222", 24, anchor="lm")

    smlmv_x = legend_x + 560
    for index, year in enumerate([START_YEAR, END_YEAR]):
        x = smlmv_x + index * 285
        draw_dashed_line(draw, (x, legend_y - 22), (x, legend_y + 22), YEAR_STYLE[year]["color"], 4, dash=14, gap=9)
        draw_text(draw, (x + 28, legend_y), f"SMLMV {year}", "#222222", 24, anchor="lm")

    draw_text(
        draw,
        (80, 930),
        "Nota: la remuneración mensual equivalente se calcula como ingreso por hora multiplicado por horas semanales y 52/12.",
        "#555555",
        21,
    )
    draw_text(
        draw,
        (80, 962),
        "El grupo aproxima jóvenes con educación superior. La GEIH no identifica año de graduación. SMLMV sin auxilio de transporte.",
        "#555555",
        21,
    )
    draw_text(draw, (80, 994), "Fuente: cálculos propios con GEIH del DANE. SMLMV según decretos del Gobierno nacional.", "#555555", 21)

    img.save(FIG_DIR / "fig_kernel_remuneracion_universitaria_25_29_2010_2025.png", quality=95)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    microdata = load_microdata()
    summary = build_summary(microdata)
    summary.to_csv(TABLE_DIR / "remuneracion_universitaria_25_29_kernel_stats.csv", index=False)
    draw_kernel_figure(microdata, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
