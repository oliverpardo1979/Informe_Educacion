from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT.parent
    / "CJC-Monitor"
    / "Datos"
    / "Processed"
    / "Paper-GEIH_base_modelo_personas_2008_2025.dta"
)
FIG_DIR = ROOT / "Paper" / "figures"
TABLE_DIR = ROOT / "Paper" / "tables"
OUTPUT = FIG_DIR / "fig_kernel_remuneracion_educacion_comparable_2010_2025.png"
STATS_OUTPUT = TABLE_DIR / "remuneracion_educacion_comparable_kernel_stats.csv"

START_YEAR = 2010
END_YEAR = 2025
MONTHS_PER_WEEK = 52.0 / 12.0

SMLMV_NOMINAL = {
    START_YEAR: 515_000,
    END_YEAR: 1_423_500,
}

IPC_DIC = {
    START_YEAR: 73.45,
    END_YEAR: 152.27,
}

COMPARABLE_ORDER = {
    "Primaria o menos": 1,
    "Secundaria": 2,
    "Superior o normalista": 3,
}

GROUP_COLORS = {
    "Total": "#222222",
    "Primaria o menos": "#8a3ffc",
    "Secundaria": "#0072b2",
    "Superior o normalista": "#009e73",
}

YEAR_COLORS = {
    START_YEAR: "#6f50ff",
    END_YEAR: "#0072b2",
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


def fmt_decimal(value: float, digits: int = 1) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def education_comparable(code: object) -> str | None:
    if pd.isna(code):
        return None
    value = float(code)
    if value in (1, 2, 3):
        return "Primaria o menos"
    if value in (4, 5):
        return "Secundaria"
    if value == 6:
        return "Superior o normalista"
    return None


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
    columns = ["anio", "fex", "horas", "ingreso_hora_real", "educ_hom_cod"]
    parts: list[pd.DataFrame] = []
    reader = pd.read_stata(
        DATA_PATH,
        columns=columns,
        convert_categoricals=False,
        chunksize=250_000,
    )
    for chunk in reader:
        chunk = chunk[chunk["anio"].isin([START_YEAR, END_YEAR])].copy()
        for column in ["anio", "fex", "horas", "ingreso_hora_real", "educ_hom_cod"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=columns)
        chunk = chunk[
            (chunk["fex"] > 0)
            & (chunk["horas"] > 0)
            & (chunk["horas"] <= 112)
            & (chunk["ingreso_hora_real"] > 0)
        ].copy()
        if chunk.empty:
            continue

        chunk["grupo_educativo"] = chunk["educ_hom_cod"].map(education_comparable)
        chunk = chunk.dropna(subset=["grupo_educativo"])
        if chunk.empty:
            continue

        parts.append(
            pd.DataFrame(
                {
                    "anio": chunk["anio"].astype(int),
                    "grupo_educativo": chunk["grupo_educativo"],
                    "peso": chunk["fex"].astype(float),
                    "rem_hora": chunk["ingreso_hora_real"].astype(float),
                    "rem_mensual": (
                        chunk["ingreso_hora_real"] * chunk["horas"] * MONTHS_PER_WEEK
                    ).astype(float),
                }
            )
        )

    if not parts:
        raise RuntimeError("No hay observaciones para construir la figura.")
    return pd.concat(parts, ignore_index=True)


def draw_dashed_vertical(
    draw: ImageDraw.ImageDraw,
    x: float,
    top: int,
    bottom: int,
    color: str,
) -> None:
    dash = 16
    gap = 10
    y = top
    while y < bottom:
        draw.line((x, y, x, min(y + dash, bottom)), fill=color, width=3)
        y += dash + gap


def build_kernel_specs(microdata: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    figure_stats: list[dict[str, float | str | int]] = []
    groups = ["Total"] + sorted(COMPARABLE_ORDER, key=COMPARABLE_ORDER.get)
    specs: list[dict[str, object]] = [
        {
            "key": "rem_mensual",
            "title": "Remuneración mensual por trabajador",
            "unit": "Millones de pesos de 2025",
            "scale": 1e6,
            "ticks": [0.2, 0.5, 1, 2, 4, 8, 16],
        },
        {
            "key": "rem_hora",
            "title": "Remuneración por hora trabajada",
            "unit": "Miles de pesos de 2025 por hora",
            "scale": 1e3,
            "ticks": [2, 3, 5, 10, 20, 40, 80],
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

        for group in groups:
            for year in [START_YEAR, END_YEAR]:
                if group == "Total":
                    subset = microdata[microdata["anio"] == year]
                else:
                    subset = microdata[
                        (microdata["grupo_educativo"] == group)
                        & (microdata["anio"] == year)
                    ]
                values = subset[key].to_numpy(dtype=float) / scale
                group_weights = subset["peso"].to_numpy(dtype=float)
                log_values = np.log(values)
                density, bandwidth = weighted_kde(log_values, group_weights, grid)
                q10, q50, q90 = weighted_quantile(values, group_weights, [0.10, 0.50, 0.90])
                figure_stats.append(
                    {
                        "indicador": key,
                        "grupo_educativo": group,
                        "anio": year,
                        "obs_sin_expandir": len(subset),
                        "ocupados_expandido": group_weights.sum(),
                        "p10": q10,
                        "p50": q50,
                        "p90": q90,
                        "p90_p10": q90 / q10,
                        "sd_log": weighted_std(log_values, group_weights),
                        "bandwidth_log": bandwidth,
                    }
                )
                series[(group, year)] = density

        spec["support"] = support
        spec["grid"] = grid
        spec["series"] = series

    return specs, pd.DataFrame(figure_stats)


def draw_figure(microdata: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    specs, stats = build_kernel_specs(microdata)
    stats.to_csv(STATS_OUTPUT, index=False)

    groups = ["Total"] + sorted(COMPARABLE_ORDER, key=COMPARABLE_ORDER.get)
    width = 2200
    height = 2560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw_text(
        draw,
        (80, 48),
        "Distribución de la remuneración por logro educativo",
        "#111111",
        size=52,
        bold=True,
    )
    draw_text(
        draw,
        (80, 112),
        f"{START_YEAR} y {END_YEAR}. Kernels ponderados por factor de expansión. Eje horizontal en escala logarítmica.",
        "#555555",
        size=30,
    )
    draw_text(
        draw,
        (80, 153),
        "En los paneles mensuales, las líneas punteadas verticales muestran el SMLMV de cada año, expresado en pesos de 2025.",
        "#555555",
        size=27,
    )

    left_col = (120, 365, 1035, 725)
    right_col = (1225, 365, 2140, 725)
    row_step = 435
    plot_boxes = []
    for row_idx, group in enumerate(groups):
        y_shift = row_idx * row_step
        plot_boxes.append(
            (
                group,
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

    draw_text(draw, (left_col[0], 265), str(specs[0]["title"]), "#111111", size=35, bold=True)
    draw_text(draw, (left_col[0], 306), str(specs[0]["unit"]), "#555555", size=25)
    draw_text(draw, (right_col[0], 265), str(specs[1]["title"]), "#111111", size=35, bold=True)
    draw_text(draw, (right_col[0], 306), str(specs[1]["unit"]), "#555555", size=25)

    def x_for(value: float, left: int, right: int, support: np.ndarray) -> float:
        log_value = math.log(value)
        return left + (log_value - support[0]) / (support[1] - support[0]) * (right - left)

    def y_for(value: float, top: int, bottom: int, max_density: float) -> float:
        return bottom - (value / max_density) * (bottom - top)

    for group, boxes in plot_boxes:
        draw_text(
            draw,
            (left_col[0], boxes[0][1] - 35),
            group,
            GROUP_COLORS[group],
            size=31,
            bold=True,
        )
        for box, spec in zip(boxes, specs):
            left, top, right, bottom = box
            support = np.asarray(spec["support"], dtype=float)
            grid = np.asarray(spec["grid"], dtype=float)
            series = spec["series"]
            densities = [series[(group, year)] for year in [START_YEAR, END_YEAR]]
            max_density = max(float(density.max()) for density in densities) * 1.08

            for grid_value in np.linspace(0, max_density, 5):
                y = y_for(float(grid_value), top, bottom, max_density)
                draw.line((left, y, right, y), fill="#e9e9e9", width=1)

            for tick in spec["ticks"]:
                tick_value = float(tick)
                log_tick = math.log(tick_value)
                if support[0] <= log_tick <= support[1]:
                    x = x_for(tick_value, left, right, support)
                    draw.line((x, top, x, bottom), fill="#f0f0f0", width=1)
                    digits = 1 if tick_value < 1 else 0
                    draw_text(
                        draw,
                        (x, bottom + 18),
                        fmt_decimal(tick_value, digits),
                        "#555555",
                        size=21,
                        anchor="mt",
                    )

            if spec["key"] == "rem_mensual":
                for year in [START_YEAR, END_YEAR]:
                    smlmv_real = (
                        SMLMV_NOMINAL[year]
                        * IPC_DIC[END_YEAR]
                        / IPC_DIC[year]
                        / float(spec["scale"])
                    )
                    log_smlmv = math.log(smlmv_real)
                    if support[0] <= log_smlmv <= support[1]:
                        x = x_for(float(smlmv_real), left, right, support)
                        draw_dashed_vertical(draw, x, top, bottom, YEAR_COLORS[year])

            draw.line((left, bottom, right, bottom), fill="#444444", width=2)
            draw.line((left, top, left, bottom), fill="#444444", width=2)

            for year in [START_YEAR, END_YEAR]:
                density = series[(group, year)]
                points = []
                for grid_value, density_value in zip(grid, density):
                    x = left + (grid_value - support[0]) / (support[1] - support[0]) * (right - left)
                    y = y_for(float(density_value), top, bottom, max_density)
                    points.append((x, y))
                draw.line(points, fill=YEAR_COLORS[year], width=5)

    legend_y = 2255
    legend_x = 185
    for idx, year in enumerate([START_YEAR, END_YEAR]):
        x = legend_x + idx * 230
        draw.line((x, legend_y, x + 70, legend_y), fill=YEAR_COLORS[year], width=7)
        draw_text(draw, (x + 88, legend_y), str(year), "#222222", size=28, anchor="lm")
    for idx, year in enumerate([START_YEAR, END_YEAR]):
        x = legend_x + 540 + idx * 330
        draw_dashed_vertical(draw, x + 35, legend_y - 21, legend_y + 21, YEAR_COLORS[year])
        draw_text(draw, (x + 68, legend_y), f"SMLMV {year}", "#222222", size=28, anchor="lm")

    draw_text(
        draw,
        (80, 2375),
        "Nota: remuneración mensual equivalente por trabajador e ingreso por hora en pesos constantes de 2025. SMLMV sin auxilio de transporte.",
        "#555555",
        size=24,
    )
    draw_text(
        draw,
        (80, 2412),
        "Fuente: cálculos propios con GEIH del DANE y decretos del Gobierno nacional para SMLMV.",
        "#555555",
        size=24,
    )

    img.save(OUTPUT, quality=95)


def main() -> None:
    microdata = load_microdata()
    draw_figure(microdata)
    print(f"Figura guardada en {OUTPUT}")
    print(f"Estadísticos guardados en {STATS_OUTPUT}")


if __name__ == "__main__":
    main()
