from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "Paper" / "tables"
SECTION_DIR = ROOT / "Paper" / "sections"
FIG_DIR = ROOT / "Paper" / "figures"

DETAIL_START_YEAR = 2021
END_YEAR = 2025
GROUPS = ["Primaria o menos", "Secundaria", "Universitaria o superior"]


def annual_growth(start: float, end: float, years: int) -> float:
    if start <= 0 or end <= 0:
        return float("nan")
    return (end / start) ** (1 / years) - 1


def fmt_decimal(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(value: float, digits: int = 1) -> str:
    return fmt_decimal(100 * value, digits) + r"\%"


def fmt_pp(value: float, digits: int = 1) -> str:
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return fmt_decimal(rounded, digits)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, selected_font) -> int:
    box = draw.textbbox((0, 0), text, font=selected_font)
    return box[2] - box[0]


def calibrate_summary(summary: pd.DataFrame, comparable: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    years = [DETAIL_START_YEAR, END_YEAR]

    for year in years:
        for group in GROUPS:
            group_mask = out["grupo_comparable"].eq(group)
            group_rows = out.loc[group_mask]
            comp_row = comparable[
                comparable["anio"].eq(year)
                & comparable["grupo_educativo"].eq(group)
            ].iloc[0]

            workers = group_rows[f"trabajadores_{year}"]

            monthly_avg = (
                group_rows[f"rem_trabajador_{year}"] * workers
            ).sum() / workers.sum()
            monthly_factor = comp_row["rem_por_trabajador"] / monthly_avg

            hourly_avg = (
                group_rows[f"rem_hora_{year}"] * workers
            ).sum() / workers.sum()
            hourly_factor = comp_row["rem_por_hora"] / hourly_avg

            out.loc[group_mask, f"rem_trabajador_{year}"] = (
                out.loc[group_mask, f"rem_trabajador_{year}"] * monthly_factor
            )
            out.loc[group_mask, f"rem_hora_{year}"] = (
                out.loc[group_mask, f"rem_hora_{year}"] * hourly_factor
            )

    out["crec_rem_hora"] = out.apply(
        lambda row: annual_growth(
            row[f"rem_hora_{DETAIL_START_YEAR}"],
            row[f"rem_hora_{END_YEAR}"],
            END_YEAR - DETAIL_START_YEAR,
        ),
        axis=1,
    )
    out["crec_rem_trabajador"] = out.apply(
        lambda row: annual_growth(
            row[f"rem_trabajador_{DETAIL_START_YEAR}"],
            row[f"rem_trabajador_{END_YEAR}"],
            END_YEAR - DETAIL_START_YEAR,
        ),
        axis=1,
    )
    return out


def calibrate_series(series: pd.DataFrame, summary: pd.DataFrame, comparable: pd.DataFrame) -> pd.DataFrame:
    out = series.copy()
    group_lookup = summary.set_index("categoria_educativa")["grupo_comparable"].to_dict()
    out["grupo_comparable"] = out["categoria_educativa"].map(group_lookup)

    for year in sorted(out["anio"].unique()):
        for group in GROUPS:
            mask = out["anio"].eq(year) & out["grupo_comparable"].eq(group)
            group_rows = out.loc[mask]
            if group_rows.empty:
                continue
            comp_match = comparable[
                comparable["anio"].eq(year)
                & comparable["grupo_educativo"].eq(group)
            ]
            if comp_match.empty:
                continue
            workers = group_rows["trabajadores"]
            hourly_avg = (group_rows["rem_hora_real"] * workers).sum() / workers.sum()
            factor = comp_match.iloc[0]["rem_por_hora"] / hourly_avg
            out.loc[mask, "rem_hora_real"] = out.loc[mask, "rem_hora_real"] * factor
            out.loc[mask, "ingreso_hora_total_expandido_real"] = (
                out.loc[mask, "rem_hora_real"] * out.loc[mask, "trabajadores"]
            )

    columns = [
        "anio",
        "categoria_educativa",
        "educacion_orden",
        "trabajadores",
        "ingreso_hora_total_expandido_real",
        "observaciones",
        "rem_hora_real",
        "trabajadores_total_anio",
        "participacion_empleo",
    ]
    return out[columns]


def total_from_comparable(comparable: pd.DataFrame, year: int) -> dict[str, float]:
    data = comparable[comparable["anio"].eq(year)]
    workers = data["ocupados"].sum()
    monthly_total = data["rem_total_mensual"].sum()
    hours = data["horas_mensuales"].sum()
    return {
        "trabajadores": workers,
        "rem_trabajador": monthly_total / workers,
        "rem_hora": monthly_total / hours,
    }


def write_detail_table(summary: pd.DataFrame, comparable: pd.DataFrame) -> None:
    total_workers_2021 = summary["trabajadores_2021"].sum()
    total_workers_2025 = summary["trabajadores_2025"].sum()
    total_2021 = total_from_comparable(comparable, DETAIL_START_YEAR)
    total_2025 = total_from_comparable(comparable, END_YEAR)

    total_worker_growth = annual_growth(
        total_2021["rem_trabajador"],
        total_2025["rem_trabajador"],
        END_YEAR - DETAIL_START_YEAR,
    )
    total_hour_growth = annual_growth(
        total_2021["rem_hora"],
        total_2025["rem_hora"],
        END_YEAR - DETAIL_START_YEAR,
    )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Ocupación por logro educativo detallado, 2021 y 2025}",
        r"\label{tab:ocupacion_educacion_detallada}",
        r"\footnotesize",
        r"\begin{tabular}{@{}p{4.3cm}rrrrr@{}}",
        r"\toprule",
        r"Logro educativo & Ocupados 2021 & Ocupados 2025 & Part. 2021 & Part. 2025 & Dif. (p.p.) \\",
        r"\midrule",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.categoria_educativa)} & "
            f"{fmt_decimal(row.trabajadores_2021 / 1e6, 2)} & "
            f"{fmt_decimal(row.trabajadores_2025 / 1e6, 2)} & "
            f"{fmt_pct(row.participacion_2021, 1)} & "
            f"{fmt_pct(row.participacion_2025, 1)} & "
            f"{fmt_pp(100 * (row.participacion_2025 - row.participacion_2021), 1)} \\\\"
        )

    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & {fmt_decimal(total_workers_2021 / 1e6, 2)} & "
            rf"{fmt_decimal(total_workers_2025 / 1e6, 2)} & 100,0\% & 100,0\% & 0,0 \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption*{\footnotesize Nota: ocupados en millones de personas. Participaciones en porcentaje. Diferencia en puntos porcentuales. Fuente: cálculos propios con GEIH del DANE.}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Remuneración mensual por trabajador por logro educativo detallado, 2021 y 2025}",
            r"\label{tab:remuneracion_educacion_detallada_trabajador}",
            r"\footnotesize",
            r"\begin{tabular}{@{}p{5.1cm}rrr@{}}",
            r"\toprule",
            r"Logro educativo & 2021 & 2025 & Crec. anual \\",
            r"\midrule",
        ]
    )

    for row in summary.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.categoria_educativa)} & "
            f"{fmt_decimal(row.rem_trabajador_2021 / 1e6, 2)} & "
            f"{fmt_decimal(row.rem_trabajador_2025 / 1e6, 2)} & "
            f"{fmt_pct(row.crec_rem_trabajador, 2)} \\\\"
        )

    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & {fmt_decimal(total_2021['rem_trabajador'] / 1e6, 2)} & "
            rf"{fmt_decimal(total_2025['rem_trabajador'] / 1e6, 2)} & "
            rf"{fmt_pct(total_worker_growth, 2)} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption*{\footnotesize Nota: la remuneración mensual por trabajador se expresa en millones de pesos mensuales de 2025. Las filas detalladas se calibran con un factor común dentro de cada grupo comparable para que la apertura detallada quede anclada al valor del Cuadro \ref{tab:remuneracion_educacion_remuneracion}. La calibración preserva las diferencias relativas entre categorías detalladas. El crecimiento es anualizado para 2021--2025. Fuente: cálculos propios con GEIH del DANE.}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Remuneración por hora trabajada por logro educativo detallado, 2021 y 2025}",
            r"\label{tab:remuneracion_educacion_detallada}",
            r"\footnotesize",
            r"\begin{tabular}{@{}p{5.1cm}rrr@{}}",
            r"\toprule",
            r"Logro educativo & 2021 & 2025 & Crec. anual \\",
            r"\midrule",
        ]
    )

    for row in summary.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.categoria_educativa)} & "
            f"{fmt_decimal(row.rem_hora_2021 / 1e3, 1)} & "
            f"{fmt_decimal(row.rem_hora_2025 / 1e3, 1)} & "
            f"{fmt_pct(row.crec_rem_hora, 2)} \\\\"
        )

    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & {fmt_decimal(total_2021['rem_hora'] / 1e3, 1)} & "
            rf"{fmt_decimal(total_2025['rem_hora'] / 1e3, 1)} & "
            rf"{fmt_pct(total_hour_growth, 2)} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption*{\footnotesize Nota: la remuneración por hora se expresa en miles de pesos de 2025 por hora trabajada. Las filas detalladas se calibran con un factor común dentro de cada grupo comparable para que la apertura detallada quede anclada al valor del Cuadro \ref{tab:remuneracion_educacion_remuneracion}. La calibración preserva las diferencias relativas entre categorías detalladas. El crecimiento es anualizado para 2021--2025. Fuente: cálculos propios con GEIH del DANE.}",
            r"\end{table}",
        ]
    )

    (SECTION_DIR / "remuneracion_educacion_detallada_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot_detail(summary: pd.DataFrame, metric: str, output: str, title: str, subtitle: str) -> None:
    if metric == "trabajador":
        col_2021 = "rem_trabajador_2021"
        col_2025 = "rem_trabajador_2025"
        scale = 1e6
        unit = "millones"
        digits = 1
    else:
        col_2021 = "rem_hora_2021"
        col_2025 = "rem_hora_2025"
        scale = 1e3
        unit = "miles"
        digits = 1

    data = summary.copy().reset_index(drop=True)
    labels = data["categoria_educativa"].tolist()
    values_2021 = (data[col_2021] / scale).tolist()
    values_2025 = (data[col_2025] / scale).tolist()

    width = 1800
    height = 1380
    left = 390
    right = 210
    top = 210
    bottom = 140
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = font(42, bold=True)
    subtitle_font = font(27)
    axis_font = font(24)
    label_font = font(27)
    value_font = font(25)
    legend_font = font(26)

    ink = "#222222"
    muted = "#666666"
    grid = "#d7dde2"
    line = "#b8c2cc"
    purple = "#7a5195"
    blue = "#0072b2"

    draw.text((left, 52), title, fill=ink, font=title_font)
    draw.text((left, 105), subtitle, fill=muted, font=subtitle_font)

    max_value = max(max(values_2021), max(values_2025))
    tick_step = 2 if max_value <= 12 else 10
    axis_max = max(tick_step, ((int(max_value / tick_step) + 1) * tick_step))

    def x_pos(value: float) -> int:
        return left + int((value / axis_max) * plot_width)

    row_gap = plot_height / (len(labels) - 1)

    for tick in range(0, axis_max + tick_step, tick_step):
        x = x_pos(tick)
        draw.line((x, top - 8, x, top + plot_height + 8), fill=grid, width=2)
        tick_label = fmt_decimal(tick, 0)
        tw = text_width(draw, tick_label, axis_font)
        draw.text((x - tw / 2, top + plot_height + 22), tick_label, fill=muted, font=axis_font)

    axis_label = f"{unit} de pesos de 2025"
    tw = text_width(draw, axis_label, axis_font)
    draw.text((left + plot_width / 2 - tw / 2, height - 72), axis_label, fill=muted, font=axis_font)

    for idx, label in enumerate(labels):
        y = int(top + idx * row_gap)
        draw.text((left - 28 - text_width(draw, label, label_font), y - 15), label, fill=ink, font=label_font)
        x_2021 = x_pos(values_2021[idx])
        x_2025 = x_pos(values_2025[idx])
        draw.line((x_2021, y, x_2025, y), fill=line, width=5)
        draw.ellipse((x_2021 - 10, y - 10, x_2021 + 10, y + 10), fill=purple)
        draw.ellipse((x_2025 - 10, y - 10, x_2025 + 10, y + 10), fill=blue)
        value_label = fmt_decimal(values_2025[idx], digits)
        draw.text((x_2025 + 18, y - 14), value_label, fill=ink, font=value_font)

    legend_x = width - right - 300
    legend_y = height - 80
    draw.ellipse((legend_x, legend_y - 9, legend_x + 18, legend_y + 9), fill=purple)
    draw.text((legend_x + 28, legend_y - 15), "2021", fill=ink, font=legend_font)
    draw.ellipse((legend_x + 120, legend_y - 9, legend_x + 138, legend_y + 9), fill=blue)
    draw.text((legend_x + 148, legend_y - 15), "2025", fill=ink, font=legend_font)

    image.save(FIG_DIR / output, quality=95)


def write_outputs() -> None:
    summary_path = TABLE_DIR / "remuneracion_educacion_detallada_summary.csv"
    series_path = TABLE_DIR / "remuneracion_educacion_detallada_series.csv"
    comparable_path = TABLE_DIR / "remuneracion_educacion_comparable_series.csv"

    summary = pd.read_csv(summary_path)
    series = pd.read_csv(series_path)
    comparable = pd.read_csv(comparable_path)

    calibrated_summary = calibrate_summary(summary, comparable)
    calibrated_series = calibrate_series(series, calibrated_summary, comparable)

    summary_columns = [
        "categoria_educativa",
        "educacion_orden",
        "trabajadores_2021",
        "trabajadores_2025",
        "participacion_2021",
        "participacion_2025",
        "rem_hora_2021",
        "rem_hora_2025",
        "crec_rem_hora",
        "grupo_comparable",
        "horas_mensuales_ref_2021",
        "rem_trabajador_2021",
        "horas_mensuales_ref_2025",
        "rem_trabajador_2025",
        "crec_rem_trabajador",
    ]

    calibrated_summary.to_csv(summary_path, columns=summary_columns, index=False)
    calibrated_series.to_csv(series_path, index=False)
    write_detail_table(calibrated_summary, comparable)

    plot_detail(
        calibrated_summary,
        metric="trabajador",
        output="fig_remuneracion_educacion_detallada_trabajador_2021_2025.png",
        title="Remuneración mensual por trabajador",
        subtitle="Logro educativo detallado, 2021 y 2025. Valores calibrados a la serie comparable.",
    )
    plot_detail(
        calibrated_summary,
        metric="hora",
        output="fig_remuneracion_educacion_detallada_2021_2025.png",
        title="Remuneración por hora trabajada",
        subtitle="Logro educativo detallado, 2021 y 2025. Valores calibrados a la serie comparable.",
    )


if __name__ == "__main__":
    write_outputs()
