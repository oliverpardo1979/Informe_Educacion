from __future__ import annotations

import argparse
import math
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "Paper" / "tables"
SECTION_DIR = ROOT / "Paper" / "sections"
FIGURE_DIR = ROOT / "Paper" / "figures"
RAW_DIR = ROOT / "outputs" / "microdata_raw"

MONTHLY_HOURS_FACTOR = 52 / 12
YEARS = (2021, 2025)
PRICE_FACTORS_2025 = {
    2021: 1.366753,
    2025: 1.0,
}

DOWNLOADS = {
    2021: [
        (
            "GEIH_2021_marco2018_I.zip",
            "https://microdatos.dane.gov.co/index.php/catalog/701/download/22829",
        ),
        (
            "GEIH_2021_marco2018_II.zip",
            "https://microdatos.dane.gov.co/index.php/catalog/701/download/22661",
        ),
    ],
    2025: [
        ("GEIH_2025_01.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24263"),
        ("GEIH_2025_02.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24264"),
        ("GEIH_2025_03.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24267"),
        ("GEIH_2025_04.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24269"),
        ("GEIH_2025_05.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24268"),
        ("GEIH_2025_06.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24266"),
        ("GEIH_2025_07.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24265"),
        ("GEIH_2025_08.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24307"),
        ("GEIH_2025_09.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24324"),
        ("GEIH_2025_10.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24382"),
        ("GEIH_2025_11.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24406"),
        ("GEIH_2025_12.zip", "https://microdatos.dane.gov.co/index.php/catalog/853/download/24463"),
    ],
}

P3042_DETAILED = {
    1: ("Preescolar o ninguno", 1),
    2: ("Preescolar o ninguno", 1),
    3: ("Basica primaria", 3),
    4: ("Basica secundaria", 4),
    5: ("Media academica", 5.1),
    6: ("Media tecnica", 5.2),
    7: ("Normalista", 5.3),
    8: ("Tecnica profesional", 6.1),
    9: ("Tecnologica", 6.2),
    10: ("Universitaria", 6.3),
    11: ("Especializacion", 6.4),
    12: ("Maestria", 6.5),
    13: ("Doctorado", 6.6),
}

DISPLAY_LABELS = {
    "Preescolar o ninguno": "Preescolar o ninguno",
    "Basica primaria": "Básica primaria",
    "Basica secundaria": "Básica secundaria",
    "Media academica": "Media académica",
    "Media tecnica": "Media técnica",
    "Normalista": "Normalista",
    "Tecnica profesional": "Técnica profesional",
    "Tecnologica": "Tecnológica",
    "Universitaria": "Universitaria",
    "Especializacion": "Especialización",
    "Maestria": "Maestría",
    "Doctorado": "Doctorado",
}

ORDERED_CATEGORIES = [
    "Preescolar o ninguno",
    "Basica primaria",
    "Basica secundaria",
    "Media academica",
    "Media tecnica",
    "Normalista",
    "Tecnica profesional",
    "Tecnologica",
    "Universitaria",
    "Especializacion",
    "Maestria",
    "Doctorado",
]

COMPARABLE_GROUPS = {
    "Preescolar o ninguno": "Primaria o menos",
    "Basica primaria": "Primaria o menos",
    "Basica secundaria": "Secundaria o media",
    "Media academica": "Secundaria o media",
    "Media tecnica": "Secundaria o media",
    "Normalista": "Superior o universitaria",
    "Tecnica profesional": "Superior o universitaria",
    "Tecnologica": "Superior o universitaria",
    "Universitaria": "Superior o universitaria",
    "Especializacion": "Superior o universitaria",
    "Maestria": "Superior o universitaria",
    "Doctorado": "Superior o universitaria",
}

KEY_COLS = ["DIRECTORIO", "SECUENCIA_P", "ORDEN", "HOGAR"]
CHAR_COLS = KEY_COLS + ["P3042"]
OCC_COLS = KEY_COLS + ["OCI", "P6800", "INGLABO", "FEX_C18"]


@dataclass
class MicroPair:
    zip_path: Path
    group: str
    characteristics: str
    occupied: str


def clean_name(name: str) -> str:
    return name.strip().upper()


def is_csv(name: str) -> bool:
    return name.lower().endswith(".csv")


def is_characteristics(name: str) -> bool:
    low = name.lower()
    return "caracter" in low and "salud" in low and "educ" in low


def is_occupied(name: str) -> bool:
    return Path(name).name.lower() in {"ocupados.csv", "ocupados.csv"}


def group_key(name: str) -> str:
    return str(Path(name).parent)


def list_pairs(zip_path: Path) -> list[MicroPair]:
    pairs: list[MicroPair] = []
    with zipfile.ZipFile(zip_path) as zf:
        groups: dict[str, dict[str, str]] = {}
        for entry in zf.namelist():
            if not is_csv(entry):
                continue
            group = group_key(entry)
            if is_characteristics(entry):
                groups.setdefault(group, {})["characteristics"] = entry
            elif is_occupied(entry):
                groups.setdefault(group, {})["occupied"] = entry

        for group, files in groups.items():
            if "characteristics" in files and "occupied" in files:
                pairs.append(
                    MicroPair(
                        zip_path=zip_path,
                        group=group,
                        characteristics=files["characteristics"],
                        occupied=files["occupied"],
                    )
                )
    return pairs


def detect_separator(header: str) -> str:
    semicolon = chr(59)
    if header.count(semicolon) >= header.count(","):
        return semicolon
    return ","


def read_selected_csv(zf: zipfile.ZipFile, entry: str, required: Iterable[str]) -> pd.DataFrame:
    required_set = {clean_name(col) for col in required}
    with zf.open(entry) as stream:
        sample = stream.read(4096).decode("latin1", errors="replace")
    sep = detect_separator(sample.splitlines()[0])

    with zf.open(entry) as stream:
        df = pd.read_csv(
            stream,
            sep=sep,
            encoding="latin1",
            dtype=str,
            usecols=lambda col: clean_name(col) in required_set,
            skipinitialspace=True,
            low_memory=False,
        )

    df = df.rename(columns={col: clean_name(col) for col in df.columns})
    missing = sorted(required_set.difference(df.columns))
    if missing:
        raise ValueError(f"{entry} no contiene columnas requeridas: {missing}")

    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.str.replace(",", ".", regex=False), errors="coerce")


def process_pair(pair: MicroPair, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    with zipfile.ZipFile(pair.zip_path) as zf:
        char = read_selected_csv(zf, pair.characteristics, CHAR_COLS)
        occ = read_selected_csv(zf, pair.occupied, OCC_COLS)

    char_dupes = int(char.duplicated(KEY_COLS).sum())
    occ_dupes = int(occ.duplicated(KEY_COLS).sum())
    if char_dupes:
        char = char.drop_duplicates(KEY_COLS, keep="first")

    occ_rows = len(occ)
    merged = occ.merge(char[KEY_COLS + ["P3042"]], on=KEY_COLS, how="left", validate="many_to_one")
    unmatched = int(merged["P3042"].isna().sum())

    merged["p3042"] = to_number(merged["P3042"])
    merged["oci"] = to_number(merged["OCI"])
    merged["hours_week"] = to_number(merged["P6800"])
    merged["income_month"] = to_number(merged["INGLABO"])
    merged["weight"] = to_number(merged["FEX_C18"]) / 12

    valid = merged[
        (merged["oci"] == 1)
        & merged["p3042"].isin(P3042_DETAILED.keys())
        & (merged["income_month"] > 0)
        & (merged["hours_week"] > 0)
        & (merged["hours_week"] <= 168)
        & (merged["weight"] > 0)
    ].copy()

    price_factor = PRICE_FACTORS_2025[year]
    valid["category"] = valid["p3042"].astype(int).map(lambda code: P3042_DETAILED[code][0])
    valid["order"] = valid["p3042"].astype(int).map(lambda code: P3042_DETAILED[code][1])
    valid["income_month_real"] = valid["income_month"] * price_factor
    valid["monthly_hours"] = valid["hours_week"] * MONTHLY_HOURS_FACTOR
    valid["weighted_income"] = valid["income_month_real"] * valid["weight"]
    valid["weighted_hours"] = valid["monthly_hours"] * valid["weight"]

    grouped = (
        valid.groupby(["category", "order"], as_index=False)
        .agg(
            workers=("weight", "sum"),
            income_total=("weighted_income", "sum"),
            hours_total=("weighted_hours", "sum"),
            observations=("weight", "size"),
        )
        .assign(year=year)
    )
    grouped["rem_worker"] = grouped["income_total"] / grouped["workers"]
    grouped["rem_hour"] = grouped["income_total"] / grouped["hours_total"]

    audit = {
        "year": year,
        "zip": pair.zip_path.name,
        "group": pair.group,
        "characteristics": pair.characteristics,
        "occupied": pair.occupied,
        "occupied_rows": occ_rows,
        "characteristics_duplicate_keys": char_dupes,
        "occupied_duplicate_keys": occ_dupes,
        "unmatched_occupied_rows": unmatched,
        "valid_rows": int(len(valid)),
        "expanded_workers": float(valid["weight"].sum()),
    }
    return grouped, audit


def add_total(df: pd.DataFrame, year: int) -> pd.DataFrame:
    total = pd.DataFrame(
        {
            "category": ["Total"],
            "order": [99.0],
            "workers": [df["workers"].sum()],
            "income_total": [df["income_total"].sum()],
            "hours_total": [df["hours_total"].sum()],
            "observations": [df["observations"].sum()],
            "year": [year],
        }
    )
    total["rem_worker"] = total["income_total"] / total["workers"]
    total["rem_hour"] = total["income_total"] / total["hours_total"]
    return pd.concat([df, total], ignore_index=True)


def process_year(year: int, zip_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for zip_path in zip_paths:
        pairs = list_pairs(zip_path)
        if not pairs:
            raise ValueError(f"No encontre pares de caracteristicas y ocupados en {zip_path}")
        for pair in pairs:
            frame, audit = process_pair(pair, year)
            frames.append(frame)
            audits.append(audit)

    annual = (
        pd.concat(frames, ignore_index=True)
        .groupby(["category", "order", "year"], as_index=False)
        .agg(
            workers=("workers", "sum"),
            income_total=("income_total", "sum"),
            hours_total=("hours_total", "sum"),
            observations=("observations", "sum"),
        )
    )
    annual["rem_worker"] = annual["income_total"] / annual["workers"]
    annual["rem_hour"] = annual["income_total"] / annual["hours_total"]
    annual = add_total(annual, year)
    annual["share"] = annual["workers"] / annual.loc[annual["category"] == "Total", "workers"].iloc[0]

    audit_df = pd.DataFrame(audits)
    return annual, audit_df


def category_sort_value(category: str) -> float:
    if category == "Total":
        return 99
    return P3042_DETAILED[
        next(code for code, value in P3042_DETAILED.items() if value[0] == category)
    ][1]


def annual_growth(v0: float, v1: float, years: int = 4) -> float:
    if v0 <= 0 or v1 <= 0:
        return float("nan")
    return (v1 / v0) ** (1 / years) - 1


def build_summary(series: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in ORDERED_CATEGORIES + ["Total"]:
        sub = series[series["category"] == category].set_index("year")
        if not all(year in sub.index for year in YEARS):
            continue
        row: dict[str, object] = {
            "categoria_educativa": DISPLAY_LABELS.get(category, category),
            "categoria_codigo": category,
            "educacion_orden": category_sort_value(category),
        }
        for year in YEARS:
            row[f"trabajadores_{year}"] = sub.loc[year, "workers"]
            row[f"participacion_{year}"] = sub.loc[year, "share"]
            row[f"rem_trabajador_{year}"] = sub.loc[year, "rem_worker"]
            row[f"rem_hora_{year}"] = sub.loc[year, "rem_hour"]
            row[f"observaciones_{year}"] = sub.loc[year, "observations"]
        row["dif_participacion"] = row["participacion_2025"] - row["participacion_2021"]
        row["crec_rem_trabajador"] = annual_growth(row["rem_trabajador_2021"], row["rem_trabajador_2025"])
        row["crec_rem_hora"] = annual_growth(row["rem_hora_2021"], row["rem_hora_2025"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("educacion_orden").reset_index(drop=True)


def fmt_decimal(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}".replace(".", ",")


def fmt_signed_pp(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "--"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}".replace(".", ",")


def fmt_percent(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "--"
    return f"{100 * value:.{digits}f}\\%".replace(".", ",")


def fmt_growth(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{100 * value:.1f}\\%".replace(".", ",")


def latex_escape(text: str) -> str:
    return text.replace("&", "\\&")


def table_rows(summary: pd.DataFrame, columns: list[tuple[str, str, int]]) -> list[str]:
    rows: list[str] = []
    for _, row in summary.iterrows():
        label = latex_escape(row["categoria_educativa"])
        if row["categoria_educativa"] == "Total":
            label = r"\textbf{Total}"
        values = []
        for col, kind, digits in columns:
            value = row[col]
            if kind == "millions":
                values.append(fmt_decimal(value / 1_000_000, digits))
            elif kind == "thousands":
                values.append(fmt_decimal(value / 1_000, digits))
            elif kind == "share":
                values.append(fmt_percent(value, digits))
            elif kind == "pp":
                values.append(fmt_signed_pp(100 * value, digits))
            elif kind == "growth":
                values.append(fmt_growth(value))
            else:
                values.append(fmt_decimal(value, digits))
        rows.append(label + " & " + " & ".join(values) + r" \\")
        if row["categoria_educativa"] == "Doctorado":
            rows.append(r"\midrule")
    return rows


def write_detail_table(summary: pd.DataFrame) -> None:
    occupation_rows = table_rows(
        summary,
        [
            ("trabajadores_2021", "millions", 1),
            ("trabajadores_2025", "millions", 1),
            ("participacion_2021", "share", 1),
            ("participacion_2025", "share", 1),
            ("dif_participacion", "pp", 1),
        ],
    )
    monthly_rows = table_rows(
        summary,
        [
            ("rem_trabajador_2021", "millions", 1),
            ("rem_trabajador_2025", "millions", 1),
            ("crec_rem_trabajador", "growth", 1),
        ],
    )
    hourly_rows = table_rows(
        summary,
        [
            ("rem_hora_2021", "thousands", 1),
            ("rem_hora_2025", "thousands", 1),
            ("crec_rem_hora", "growth", 1),
        ],
    )
    remuneration_rows: list[str] = []
    for monthly_row, hourly_row in zip(monthly_rows, hourly_rows):
        if monthly_row == r"\midrule":
            remuneration_rows.append(monthly_row)
            continue
        monthly_parts = monthly_row.replace(r" \\", "").split(" & ")
        hourly_parts = hourly_row.replace(r" \\", "").split(" & ")
        remuneration_rows.append(" & ".join(monthly_parts + hourly_parts[1:]) + r" \\")

    text = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Ocupación por logro educativo detallado, 2021 y 2025}",
            r"\label{tab:ocupacion_educacion_detallada}",
            r"\footnotesize",
            r"\begin{tabular}{@{}p{4.3cm}rrrrr@{}}",
            r"\toprule",
            r"Logro educativo & Ocupados 2021 & Ocupados 2025 & Part. 2021 & Part. 2025 & Dif. (p.p.) \\",
            r"\midrule",
            *occupation_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption*{\footnotesize Nota: ocupados en millones de personas. Participaciones en porcentaje. Diferencia en puntos porcentuales. Por redondeo a una cifra decimal, categorías con menos de 50 mil ocupados pueden aparecer como 0,0 millones. Cálculos con microdatos mensuales de la GEIH marco 2018. Fuente: cálculos propios con GEIH del DANE.}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Remuneración laboral por logro educativo detallado, 2021 y 2025}",
            r"\label{tab:remuneracion_educacion_detallada_trabajador}",
            r"\label{tab:remuneracion_educacion_detallada}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}p{3.6cm}rrrrrr@{}}",
            r"\toprule",
            r"& \multicolumn{3}{c}{Remuneración mensual} & \multicolumn{3}{c}{Remuneración por hora} \\",
            r"\cmidrule(lr){2-4} \cmidrule(l){5-7}",
            r"Logro educativo & 2021 & 2025 & Crec. anual & 2021 & 2025 & Crec. anual \\",
            r"\midrule",
            *remuneration_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption*{\footnotesize Nota: la remuneración mensual se reporta en millones de pesos mensuales de 2025 por trabajador. La remuneración por hora se reporta en miles de pesos de 2025 por hora trabajada. El crecimiento es anualizado para 2021--2025. Cálculos con microdatos mensuales de la GEIH marco 2018. Fuente: cálculos propios con GEIH del DANE.}",
            r"\end{table}",
            "",
        ]
    )
    (SECTION_DIR / "remuneracion_educacion_detallada_table.tex").write_text(text, encoding="utf-8")


def write_reconciliation(series: pd.DataFrame) -> None:
    comparable_path = TABLE_DIR / "remuneracion_educacion_comparable_series.csv"
    if not comparable_path.exists():
        return

    detailed = series[series["category"] != "Total"].copy()
    detailed["grupo_educativo"] = detailed["category"].map(COMPARABLE_GROUPS)
    grouped = (
        detailed.groupby(["year", "grupo_educativo"], as_index=False)
        .agg(
            ocupados_detalle=("workers", "sum"),
            ingreso_detalle=("income_total", "sum"),
            horas_detalle=("hours_total", "sum"),
        )
    )
    grouped["rem_trabajador_detalle"] = grouped["ingreso_detalle"] / grouped["ocupados_detalle"]
    grouped["rem_hora_detalle"] = grouped["ingreso_detalle"] / grouped["horas_detalle"]

    comparable = pd.read_csv(comparable_path)
    comparable = comparable[comparable["anio"].isin(YEARS)].rename(columns={"anio": "year"})
    comparable = comparable[
        ["year", "grupo_educativo", "ocupados", "rem_por_trabajador", "rem_por_hora"]
    ].rename(
        columns={
            "ocupados": "ocupados_comparable",
            "rem_por_trabajador": "rem_trabajador_comparable",
            "rem_por_hora": "rem_hora_comparable",
        }
    )

    reconciliation = grouped.merge(comparable, on=["year", "grupo_educativo"], how="inner")
    reconciliation["dif_ocupados"] = (
        reconciliation["ocupados_detalle"] - reconciliation["ocupados_comparable"]
    )
    reconciliation["dif_ocupados_pct"] = (
        reconciliation["dif_ocupados"] / reconciliation["ocupados_comparable"]
    )
    reconciliation["dif_rem_trabajador"] = (
        reconciliation["rem_trabajador_detalle"] - reconciliation["rem_trabajador_comparable"]
    )
    reconciliation["dif_rem_trabajador_pct"] = (
        reconciliation["dif_rem_trabajador"] / reconciliation["rem_trabajador_comparable"]
    )
    reconciliation["dif_rem_hora"] = (
        reconciliation["rem_hora_detalle"] - reconciliation["rem_hora_comparable"]
    )
    reconciliation["dif_rem_hora_pct"] = (
        reconciliation["dif_rem_hora"] / reconciliation["rem_hora_comparable"]
    )
    reconciliation.to_csv(TABLE_DIR / "remuneracion_educacion_detallada_reconciliacion.csv", index=False)


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


def nice_step(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exponent)


def format_axis_value(value: float) -> str:
    if value >= 10:
        return fmt_decimal(value, 0)
    return fmt_decimal(value, 1)


def plot_dumbbell(summary: pd.DataFrame, value_2021: str, value_2025: str, scale: float, xlabel: str, title: str, output: Path) -> None:
    plot_data = summary[summary["categoria_educativa"] != "Total"].copy()
    plot_data = plot_data.sort_values("educacion_orden", ascending=False)
    x0 = (plot_data[value_2021] / scale).to_numpy()
    x1 = (plot_data[value_2025] / scale).to_numpy()
    labels = plot_data["categoria_educativa"].to_list()

    width, height = 1800, 1280
    left, right, top, bottom = 470, 160, 170, 145
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(labels)

    max_value = float(max(np.nanmax(x0), np.nanmax(x1)))
    step = nice_step(max_value / 5)
    xmax = step * math.ceil(max_value / step)
    ticks = np.arange(0, xmax + step * 0.5, step)

    def x_pos(value: float) -> float:
        return left + (value / xmax) * plot_w

    def y_pos(index: int) -> float:
        if n == 1:
            return top + plot_h / 2
        return top + index * (plot_h / (n - 1))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw_text(draw, (60, 55), title, "#1f2933", 40, bold=True)
    draw_text(draw, (60, 105), "Cálculos con microdatos mensuales. Valores en pesos constantes de 2025.", "#52616b", 24)

    for tick in ticks:
        x = x_pos(float(tick))
        draw.line((x, top - 18, x, top + plot_h + 10), fill="#e5e7eb", width=2)
        draw_text(draw, (x, top + plot_h + 34), format_axis_value(float(tick)), "#52616b", 22, anchor="mm")

    draw.line((left, top + plot_h + 10, left + plot_w, top + plot_h + 10), fill="#8b949e", width=2)
    draw_text(draw, (left + plot_w / 2, height - 72), xlabel, "#1f2933", 25, anchor="mm")

    for idx, (label, a, b) in enumerate(zip(labels, x0, x1)):
        y = y_pos(idx)
        draw_text(draw, (left - 22, y), label, "#1f2933", 26, anchor="rm")
        draw.line((x_pos(float(a)), y, x_pos(float(b)), y), fill="#a8b0b8", width=5)
        r = 10
        xa = x_pos(float(a))
        xb = x_pos(float(b))
        draw.ellipse((xa - r, y - r, xa + r, y + r), fill="#5161ce", outline="white", width=3)
        draw.ellipse((xb - r, y - r, xb + r, y + r), fill="#0b8f78", outline="white", width=3)

    legend_y = 112
    legend_x = width - 350
    draw.ellipse((legend_x, legend_y - 10, legend_x + 20, legend_y + 10), fill="#5161ce")
    draw_text(draw, (legend_x + 32, legend_y), "2021", "#1f2933", 24, anchor="lm")
    draw.ellipse((legend_x + 120, legend_y - 10, legend_x + 140, legend_y + 10), fill="#0b8f78")
    draw_text(draw, (legend_x + 152, legend_y), "2025", "#1f2933", 24, anchor="lm")

    draw_text(draw, (60, height - 35), "Fuente: cálculos propios con microdatos GEIH del DANE.", "#52616b", 21)
    image.save(output)


def download_files(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year, files in DOWNLOADS.items():
        for filename, url in files:
            target = raw_dir / filename
            if target.exists() and target.stat().st_size > 0:
                continue
            print(f"Descargando {year}: {filename}")
            tmp = target.with_suffix(target.suffix + ".tmp")
            with urllib.request.urlopen(url) as response:
                with tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
            tmp.replace(target)


def required_paths(raw_dir: Path) -> dict[int, list[Path]]:
    paths: dict[int, list[Path]] = {}
    missing: list[str] = []
    for year, files in DOWNLOADS.items():
        year_paths: list[Path] = []
        for filename, url in files:
            path = raw_dir / filename
            if not path.exists():
                missing.append(f"{filename}  {url}")
            year_paths.append(path)
        paths[year] = year_paths
    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(
            "Faltan archivos de microdata en outputs/microdata_raw. "
            "Ejecute con --download o descargue estos archivos:\n" + formatted
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_files(args.raw_dir)

    paths = required_paths(args.raw_dir)
    all_series: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []

    for year in YEARS:
        print(f"Procesando {year}")
        series, audit = process_year(year, paths[year])
        all_series.append(series)
        all_audits.append(audit)

    series = pd.concat(all_series, ignore_index=True)
    summary = build_summary(series)
    audit = pd.concat(all_audits, ignore_index=True)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(TABLE_DIR / "remuneracion_educacion_detallada_summary.csv", index=False)
    series.to_csv(TABLE_DIR / "remuneracion_educacion_detallada_series.csv", index=False)
    audit.to_csv(TABLE_DIR / "remuneracion_educacion_detallada_microdata_audit.csv", index=False)
    write_reconciliation(series)

    write_detail_table(summary)
    plot_dumbbell(
        summary,
        "rem_trabajador_2021",
        "rem_trabajador_2025",
        1_000_000,
        "Millones de pesos mensuales de 2025",
        "Remuneración mensual por trabajador por logro educativo detallado",
        FIGURE_DIR / "fig_remuneracion_educacion_detallada_trabajador_2021_2025.png",
    )
    plot_dumbbell(
        summary,
        "rem_hora_2021",
        "rem_hora_2025",
        1_000,
        "Miles de pesos de 2025 por hora trabajada",
        "Remuneración por hora por logro educativo detallado",
        FIGURE_DIR / "fig_remuneracion_educacion_detallada_2021_2025.png",
    )

    print("Resumen detallado escrito en Paper/tables/remuneracion_educacion_detallada_summary.csv")
    print("Auditoria escrita en Paper/tables/remuneracion_educacion_detallada_microdata_audit.csv")


if __name__ == "__main__":
    main()
