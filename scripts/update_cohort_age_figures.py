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
FIG_DIR = REPO_ROOT / "Paper" / "figures"

YEARS = [2010, 2015, 2021, 2025]
MONTHS_PER_WEEK = 52.0 / 12.0
COHORT_WIDTH = 10
COHORT_STARTS = list(range(1960, 2010, COHORT_WIDTH))
AGE_BIN_WIDTH = 10
AGE_MAX_LABEL = 64

EDUCATION_ORDER = {
    "Primaria o menos": 1,
    "Secundaria": 2,
    "Universitaria o superior": 3,
}

EDUCATION_COLORS = {
    "Primaria o menos": "#8a3ffc",
    "Secundaria": "#0072b2",
    "Universitaria o superior": "#009e73",
}

EDUCATION_SLUGS = {
    "Primaria o menos": "primaria",
    "Secundaria": "secundaria",
    "Universitaria o superior": "superior",
}

COHORT_COLORS = {
    "1960--1970": "#6c757d",
    "1970--1980": "#d45087",
    "1980--1990": "#009e73",
    "1990--2000": "#e69f00",
    "2000--2010": "#56b4e9",
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


def fmt_decimal(value: float, digits: int = 1) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def education_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"ninguno", "preescolar", "básica primaria", "basica primaria"}:
            return "Primaria o menos"
        if normalized in {"básica secundaria", "basica secundaria", "media"}:
            return "Secundaria"
        if normalized in {"superior o universitaria", "universitaria o superior"}:
            return "Universitaria o superior"
        return None
    code = float(value)
    if code in (1, 2, 3):
        return "Primaria o menos"
    if code in (4, 5):
        return "Secundaria"
    if code == 6:
        return "Universitaria o superior"
    return None


def cohort_label(birth_year: float) -> str | None:
    if pd.isna(birth_year):
        return None
    for start in COHORT_STARTS:
        if start <= birth_year < start + COHORT_WIDTH:
            return f"{start}--{start + COHORT_WIDTH}"
    return None


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float((values * weights).sum() / weights.sum())


def age_interval_start(age: float, start: int = 15) -> int:
    return start + int(math.floor((age - start) / AGE_BIN_WIDTH) * AGE_BIN_WIDTH)


def age_interval_end(start: int) -> int:
    return min(start + AGE_BIN_WIDTH - 1, AGE_MAX_LABEL)


def age_interval_label(start: int) -> str:
    return f"{start}--{age_interval_end(start)}"


def age_interval_center(start: int) -> float:
    return (start + age_interval_end(start)) / 2


def add_age_axis_columns(data: pd.DataFrame, start: int) -> pd.DataFrame:
    data = data.copy()
    data["edad_eje_inicio"] = data["edad_media"].map(lambda value: age_interval_start(value, start))
    data["edad_eje"] = data["edad_eje_inicio"].map(age_interval_label)
    data["edad_eje_centro"] = data["edad_eje_inicio"].map(age_interval_center)
    return data


def keep_cohorts_with_min_realizations(data: pd.DataFrame, minimum: int = 2) -> pd.DataFrame:
    counts = data.groupby("cohorte")["anio"].nunique()
    keep = set(counts[counts >= minimum].index)
    return data[data["cohorte"].isin(keep)].copy()


def build_microdata() -> pd.DataFrame:
    columns = ["anio", "edad", "educ_hom_cod", "fex", "horas", "ingreso_hora_real"]
    df = pd.read_stata(DATA_PATH, columns=columns)
    df = df[df["anio"].isin(YEARS)].copy()
    for column in ["anio", "edad", "fex", "horas", "ingreso_hora_real"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["anio", "edad", "fex", "horas", "ingreso_hora_real", "educ_hom_cod"])
    df = df[
        (df["edad"] >= 15)
        & (df["fex"] > 0)
        & (df["horas"] > 0)
        & (df["horas"] <= 112)
        & (df["ingreso_hora_real"] > 0)
    ].copy()
    df["grupo_educativo"] = df["educ_hom_cod"].map(education_group)
    df = df[df["grupo_educativo"].notna()].copy()
    df["anio"] = df["anio"].astype(int)
    df["birth_year"] = df["anio"] - df["edad"]
    df["cohorte"] = df["birth_year"].map(cohort_label)
    df = df[df["cohorte"].notna()].copy()
    df["orden_cohorte"] = df["cohorte"].map(
        {f"{start}--{start + COHORT_WIDTH}": idx for idx, start in enumerate(COHORT_STARTS)}
    )
    df["orden_educ"] = df["grupo_educativo"].map(EDUCATION_ORDER)
    df["rem_mensual"] = df["ingreso_hora_real"] * df["horas"] * MONTHS_PER_WEEK
    df["horas_mensuales"] = df["horas"] * MONTHS_PER_WEEK
    return df


def aggregate(data: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by, keys))
        peso = group["fex"]
        total_rem = (group["rem_mensual"] * peso).sum()
        total_hours = (group["horas_mensuales"] * peso).sum()
        ocupados = peso.sum()
        edad_media = weighted_average(group["edad"], peso)
        edad_inicio = age_interval_start(edad_media)
        row.update(
            {
                "edad_media": edad_media,
                "edad_intervalo": age_interval_label(edad_inicio),
                "edad_intervalo_inicio": edad_inicio,
                "edad_intervalo_centro": age_interval_center(edad_inicio),
                "ocupados": ocupados,
                "rem_trabajador": total_rem / ocupados,
                "rem_hora": total_rem / total_hours,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = build_microdata()
    total = aggregate(data, ["anio", "cohorte", "orden_cohorte"])
    total = keep_observed_cohorts(total)
    educ = aggregate(
        data,
        ["anio", "cohorte", "grupo_educativo", "orden_cohorte", "orden_educ"],
    )
    educ = educ[educ["cohorte"].isin(set(total["cohorte"]))].copy()
    educ = educ.sort_values(["orden_cohorte", "orden_educ", "anio"])
    total = total.sort_values(["orden_cohorte", "anio"])
    return total, educ


def keep_observed_cohorts(data: pd.DataFrame) -> pd.DataFrame:
    return keep_cohorts_with_min_realizations(data)


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
    padding = max(span * 0.14, abs(max_value) * 0.04)
    step = nice_step(span + 2 * padding, target_ticks)
    lower = math.floor((min_value - padding) / step) * step
    lower = max(0, lower)
    upper = math.ceil((max_value + padding) / step) * step
    if lower == upper:
        upper = lower + step
    return lower, upper, step


def line_points(
    data: pd.DataFrame,
    left: int,
    right: int,
    top: int,
    bottom: int,
    metric: str,
    scale: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    points = []
    for row in data.sort_values("anio").itertuples(index=False):
        x = left + (row.edad_eje_centro - x_min) / (x_max - x_min) * (right - left)
        y_value = getattr(row, metric) / scale
        y = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
        points.append((x, y))
    return points


def draw_axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    y_step: float,
    y_digits: int,
    x_label_max: int = AGE_MAX_LABEL,
) -> None:
    left, top, right, bottom = bounds
    tick = y_min
    while tick <= y_max + y_step / 2:
        y = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, y, right, y), fill="#e7e7e7", width=1)
        draw_text(draw, (left - 14, y), fmt_decimal(tick, y_digits), "#555555", 22, anchor="rm")
        tick += y_step

    for age in range(int(x_min), x_label_max + 1, AGE_BIN_WIDTH):
        center = age_interval_center(age)
        x = left + (center - x_min) / (x_max - x_min) * (right - left)
        draw.line((x, bottom, x, bottom + 8), fill="#333333", width=2)
        draw_text(draw, (x, bottom + 20), age_interval_label(age), "#555555", 20, anchor="mt")

    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    cohorts: list[str],
    x: int,
    y: int,
    column_width: int,
    row_height: int,
    columns: int,
) -> None:
    for idx, cohort in enumerate(cohorts):
        col = idx % columns
        row = idx // columns
        item_x = x + col * column_width
        item_y = y + row * row_height
        color = COHORT_COLORS[cohort]
        draw.line((item_x, item_y, item_x + 52, item_y), fill=color, width=6)
        draw.ellipse((item_x + 19, item_y - 7, item_x + 33, item_y + 7), fill=color)
        draw_text(draw, (item_x + 68, item_y), cohort, "#222222", 23, anchor="lm")


def draw_total_figure(
    data: pd.DataFrame,
    metric: str,
    scale: float,
    y_digits: int,
    title: str,
    subtitle: str,
    output_file: str,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = add_age_axis_columns(data, 15)
    plot_data = keep_cohorts_with_min_realizations(plot_data)
    plot_data["valor"] = plot_data[metric] / scale
    y_min, y_max, y_step = axis_bounds(plot_data["valor"])
    x_min = 15
    x_max = 65
    cohorts = list(plot_data.sort_values("orden_cohorte")["cohorte"].drop_duplicates())

    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    bounds = (150, 190, 1660, 750)

    draw_text(draw, (80, 48), title, "#111111", 44, True)
    draw_text(draw, (80, 102), subtitle, "#444444", 28)
    draw_text(draw, (80, 145), "Eje horizontal: tramos decenales de edad", "#555555", 24)

    draw_axes(draw, bounds, x_min, x_max, y_min, y_max, y_step, y_digits)

    for cohort in cohorts:
        subset = plot_data[plot_data["cohorte"] == cohort]
        left, top, right, bottom = bounds
        points = line_points(subset, left, right, top, bottom, metric, scale, x_min, x_max, y_min, y_max)
        color = COHORT_COLORS[cohort]
        if len(points) >= 2:
            draw.line(points, fill=color, width=5)
        for row, (x, y) in zip(subset.sort_values("anio").itertuples(index=False), points):
            if row.anio == 2021:
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="white", outline=color, width=3)
            else:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)

    draw_legend(draw, cohorts, 150, 835, 440, 42, 3)
    draw_text(draw, (80, 1010), "Nota: 2021 se usa porque la base procesada no contiene 2020.", "#555555", 23)
    draw_text(draw, (80, 1040), "Fuente: cálculos propios con GEIH del DANE.", "#555555", 23)
    image.save(FIG_DIR / output_file, quality=95)


def education_axis_limits(group: str) -> tuple[int, int]:
    return 15, 65


def draw_single_education_figure(
    data: pd.DataFrame,
    metric: str,
    scale: float,
    y_digits: int,
    group: str,
    title_prefix: str,
    subtitle: str,
    output_file: str,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x_min, x_max = education_axis_limits(group)
    plot_data = data[
        (data["grupo_educativo"] == group)
        & (data["edad_media"] >= x_min)
        & (data["edad_media"] < x_max)
    ].copy()
    plot_data = add_age_axis_columns(plot_data, x_min)
    plot_data["valor"] = plot_data[metric] / scale
    plot_data = keep_cohorts_with_min_realizations(plot_data)
    cohorts = list(plot_data.sort_values("orden_cohorte")["cohorte"].drop_duplicates())
    y_min, y_max, y_step = axis_bounds(plot_data["valor"])

    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    bounds = (150, 210, 1660, 750)

    draw_text(draw, (80, 45), f"{title_prefix}: {group}", "#111111", 43, True)
    draw_text(draw, (80, 97), subtitle, "#444444", 27)
    range_text = "Eje horizontal: tramos decenales de edad, de 15--24 a 55--64 años"
    draw_text(draw, (80, 137), range_text, "#555555", 23)
    draw.line((80, 170, 1660, 170), fill=EDUCATION_COLORS[group], width=5)

    draw_axes(draw, bounds, x_min, x_max, y_min, y_max, y_step, y_digits)

    left, top, right, bottom = bounds
    for cohort in cohorts:
        subset = plot_data[plot_data["cohorte"] == cohort].copy()
        points = line_points(subset, left, right, top, bottom, metric, scale, x_min, x_max, y_min, y_max)
        color = COHORT_COLORS[cohort]
        draw.line(points, fill=color, width=5)
        for row, (x, y) in zip(subset.sort_values("anio").itertuples(index=False), points):
            if row.anio == 2021:
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="white", outline=color, width=3)
            else:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)

    draw_legend(draw, cohorts, 150, 835, 440, 42, 3)
    draw_text(draw, (80, 1010), "Nota: 2021 se usa porque la base procesada no contiene 2020.", "#555555", 23)
    draw_text(draw, (80, 1040), "Fuente: cálculos propios con GEIH del DANE.", "#555555", 23)
    image.save(FIG_DIR / output_file, quality=95)


def draw_education_figures(
    data: pd.DataFrame,
    metric: str,
    scale: float,
    y_digits: int,
    title_prefix: str,
    subtitle: str,
    output_prefix: str,
) -> None:
    for group in sorted(EDUCATION_ORDER, key=EDUCATION_ORDER.get):
        slug = EDUCATION_SLUGS[group]
        draw_single_education_figure(
            data,
            metric,
            scale,
            y_digits,
            group,
            title_prefix,
            subtitle,
            f"{output_prefix}_{slug}.png",
        )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    total, educ = build_series()

    total.to_csv(TABLE_DIR / "remuneracion_cohortes_edad_series.csv", index=False)
    educ.to_csv(TABLE_DIR / "remuneracion_cohortes_edad_educacion_series.csv", index=False)

    draw_total_figure(
        total,
        "rem_trabajador",
        1e6,
        1,
        "Remuneración mensual por edad y cohorte de nacimiento",
        "Cohortes decenales con al menos dos años observados. Millones de pesos mensuales de 2025",
        "fig_remuneracion_cohortes_edad_trabajador.png",
    )
    draw_total_figure(
        total,
        "rem_hora",
        1e3,
        1,
        "Remuneración por hora por edad y cohorte de nacimiento",
        "Cohortes decenales con al menos dos años observados. Miles de pesos de 2025 por hora",
        "fig_remuneracion_cohortes_edad_hora.png",
    )
    draw_education_figures(
        educ,
        "rem_trabajador",
        1e6,
        1,
        "Remuneración mensual por edad y cohorte",
        "Cohortes decenales con al menos dos años observados. Millones de pesos mensuales de 2025",
        "fig_remuneracion_cohortes_edad_educacion_trabajador",
    )
    draw_education_figures(
        educ,
        "rem_hora",
        1e3,
        1,
        "Remuneración por hora por edad y cohorte",
        "Cohortes decenales con al menos dos años observados. Miles de pesos de 2025 por hora",
        "fig_remuneracion_cohortes_edad_educacion_hora",
    )


if __name__ == "__main__":
    main()
