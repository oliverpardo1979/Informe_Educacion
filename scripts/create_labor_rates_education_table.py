from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_2010 = ROOT / "outputs" / "geih_raw"
RAW_2025 = ROOT / "outputs" / "microdata_raw"
TABLE_DIR = ROOT / "Paper" / "tables"
SECTION_DIR = ROOT / "Paper" / "sections"
CSV_SEPARATOR = chr(59)

KEY_2010 = ["DIRECTORIO", "SECUENCIA_P", "ORDEN", "MES"]
KEY_2025 = ["DIRECTORIO", "SECUENCIA_P", "ORDEN", "HOGAR", "MES"]

CATEGORIES = [
    "Ninguno o preescolar",
    "Básica primaria",
    "Básica secundaria",
    "Educación media",
    "Superior o universitaria",
]


def map_educ_2010(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return None
    mapping = {
        1: "Ninguno o preescolar",
        2: "Ninguno o preescolar",
        3: "Básica primaria",
        4: "Básica secundaria",
        5: "Educación media",
        6: "Superior o universitaria",
    }
    return mapping.get(code)


def map_educ_2025(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return None
    if code in (1, 2):
        return "Ninguno o preescolar"
    if code == 3:
        return "Básica primaria"
    if code == 4:
        return "Básica secundaria"
    if code in (5, 6):
        return "Educación media"
    if 7 <= code <= 13:
        return "Superior o universitaria"
    return None


def matches_2010_module(name: str, kind: str) -> bool:
    lower_name = name.lower()
    if not lower_name.endswith(".csv"):
        return False
    if kind == "characteristics":
        return "generales" in lower_name and "personas" in lower_name
    if kind == "labor_force":
        return "fuerza de trabajo" in lower_name
    if kind == "occupied":
        return "ocupados" in lower_name and "desocupados" not in lower_name
    if kind == "unemployed":
        return "desocupados" in lower_name
    return False


def read_2010_parts(zipped_file: zipfile.ZipFile, kind: str, columns: list[str]) -> pd.DataFrame:
    frames = []
    for member in zipped_file.namelist():
        if not matches_2010_module(member, kind):
            continue
        domain = member.split("/")[1].split(" - ")[0]
        if domain not in {"Cabecera", "Resto"}:
            continue
        frame = pd.read_csv(
            zipped_file.open(member),
            sep=CSV_SEPARATOR,
            decimal=",",
            encoding="utf-8-sig",
            usecols=lambda column: column.replace("\ufeff", "") in columns,
            low_memory=False,
        )
        frame.columns = [column.replace("\ufeff", "") for column in frame.columns]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No se encontró el módulo {kind} en el archivo de 2010")
    return pd.concat(frames, ignore_index=True)


def add_record(
    records: dict[str, dict[str, float]],
    category: str,
    pet: float,
    pea: float,
    occupied: float,
    unemployed: float,
    formal_numerator: float,
    formal_denominator: float,
) -> None:
    record = records.setdefault(
        category,
        {
            "pet": 0.0,
            "pea": 0.0,
            "occupied": 0.0,
            "unemployed": 0.0,
            "formal_numerator": 0.0,
            "formal_denominator": 0.0,
        },
    )
    record["pet"] += float(pet)
    record["pea"] += float(pea)
    record["occupied"] += float(occupied)
    record["unemployed"] += float(unemployed)
    record["formal_numerator"] += float(formal_numerator)
    record["formal_denominator"] += float(formal_denominator)


def calculate_2010() -> dict[str, dict[str, float]]:
    factors = pd.read_csv(
        RAW_2010 / "total_2010_c.csv",
        sep=CSV_SEPARATOR,
        decimal=",",
        encoding="utf-8-sig",
    )
    factors.columns = [column.replace("\ufeff", "") for column in factors.columns]
    factors = factors[KEY_2010 + ["FEX_DPTO_C"]]

    records: dict[str, dict[str, float]] = {}
    monthly_files = sorted(RAW_2010.glob("2010_*_csv.zip"))
    if len(monthly_files) != 12:
        raise FileNotFoundError("Se requieren los 12 archivos mensuales CSV de 2010")

    for zip_path in monthly_files:
        with zipfile.ZipFile(zip_path) as zipped_file:
            characteristics = read_2010_parts(zipped_file, "characteristics", KEY_2010 + ["P6210"])
            labor_force = read_2010_parts(zipped_file, "labor_force", KEY_2010)
            occupied = read_2010_parts(zipped_file, "occupied", KEY_2010 + ["OCI", "P6920"])
            unemployed = read_2010_parts(zipped_file, "unemployed", KEY_2010 + ["DSI"])

        characteristics = characteristics.drop_duplicates(KEY_2010)
        labor_force = labor_force.drop_duplicates(KEY_2010)
        occupied = occupied.drop_duplicates(KEY_2010)
        unemployed = unemployed.drop_duplicates(KEY_2010)

        frame = labor_force.merge(characteristics, on=KEY_2010, how="left")
        frame = frame.merge(factors, on=KEY_2010, how="left")
        frame = frame.merge(
            occupied[KEY_2010 + ["OCI", "P6920"]].assign(occupied_status=1),
            on=KEY_2010,
            how="left",
        )
        frame = frame.merge(
            unemployed[KEY_2010 + ["DSI"]].assign(unemployed_status=1),
            on=KEY_2010,
            how="left",
        )
        frame["occupied_status"] = frame["occupied_status"].fillna(0)
        frame["unemployed_status"] = frame["unemployed_status"].fillna(0)
        frame["pea_status"] = (
            frame["occupied_status"].eq(1) | frame["unemployed_status"].eq(1)
        ).astype(int)
        frame["category"] = frame["P6210"].map(map_educ_2010)
        frame = frame[frame["category"].notna() & frame["FEX_DPTO_C"].notna()].copy()

        weight = frame["FEX_DPTO_C"]
        frame["pet_w"] = weight
        frame["pea_w"] = weight * frame["pea_status"]
        frame["occupied_w"] = weight * frame["occupied_status"]
        frame["unemployed_w"] = weight * frame["unemployed_status"]
        frame["formal_numerator_w"] = weight * (
            frame["occupied_status"].eq(1) & frame["P6920"].eq(1)
        )
        frame["formal_denominator_w"] = weight * (
            frame["occupied_status"].eq(1) & frame["P6920"].isin([1, 2])
        )

        for category, group in frame.groupby("category"):
            add_record(
                records,
                category,
                group["pet_w"].sum(),
                group["pea_w"].sum(),
                group["occupied_w"].sum(),
                group["unemployed_w"].sum(),
                group["formal_numerator_w"].sum(),
                group["formal_denominator_w"].sum(),
            )
        add_record(
            records,
            "Total",
            frame["pet_w"].sum(),
            frame["pea_w"].sum(),
            frame["occupied_w"].sum(),
            frame["unemployed_w"].sum(),
            frame["formal_numerator_w"].sum(),
            frame["formal_denominator_w"].sum(),
        )

    return records


def find_2025_member(zipped_file: zipfile.ZipFile, kind: str) -> str:
    members = zipped_file.namelist()
    if kind == "characteristics":
        return [member for member in members if member.endswith(".DTA") and "Caracter" in member][0]
    if kind == "labor_force":
        return [member for member in members if member.endswith(".DTA") and "Fuerza" in member][0]
    if kind == "occupied":
        return [member for member in members if member.endswith("Ocupados.DTA")][0]
    if kind == "not_occupied":
        return [member for member in members if member.endswith(".DTA") and "No ocupados" in member][0]
    raise ValueError(kind)


def calculate_2025() -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    monthly_files = sorted(RAW_2025.glob("GEIH_2025_*.zip"))
    if len(monthly_files) != 12:
        raise FileNotFoundError("Se requieren los 12 archivos mensuales de 2025")

    for zip_path in monthly_files:
        with zipfile.ZipFile(zip_path) as zipped_file:
            characteristics_name = find_2025_member(zipped_file, "characteristics")
            labor_force_name = find_2025_member(zipped_file, "labor_force")
            occupied_name = find_2025_member(zipped_file, "occupied")
            not_occupied_name = find_2025_member(zipped_file, "not_occupied")

            with zipped_file.open(characteristics_name) as file:
                characteristics = pd.read_stata(
                    file,
                    columns=KEY_2025 + ["P3042"],
                    convert_categoricals=False,
                )
            with zipped_file.open(labor_force_name) as file:
                labor_force = pd.read_stata(
                    file,
                    columns=KEY_2025 + ["PET", "FT", "FEX_C18"],
                    convert_categoricals=False,
                )
            with zipped_file.open(occupied_name) as file:
                occupied = pd.read_stata(
                    file,
                    columns=KEY_2025 + ["OCI", "P6920"],
                    convert_categoricals=False,
                )
            with zipped_file.open(not_occupied_name) as file:
                not_occupied = pd.read_stata(
                    file,
                    columns=KEY_2025 + ["DSI"],
                    convert_categoricals=False,
                )

        frame = labor_force.merge(characteristics, on=KEY_2025, how="left")
        frame = frame.merge(
            occupied[KEY_2025 + ["OCI", "P6920"]].assign(occupied_status=1),
            on=KEY_2025,
            how="left",
        )
        frame = frame.merge(
            not_occupied[KEY_2025 + ["DSI"]]
            .assign(unemployed_status=lambda data: data["DSI"].eq(1).astype(int))[
                KEY_2025 + ["unemployed_status"]
            ],
            on=KEY_2025,
            how="left",
        )
        frame["occupied_status"] = frame["occupied_status"].fillna(0)
        frame["unemployed_status"] = frame["unemployed_status"].fillna(0)
        frame["category"] = frame["P3042"].map(map_educ_2025)
        frame = frame[frame["category"].notna() & frame["FEX_C18"].notna()].copy()

        weight = frame["FEX_C18"] / 12
        frame["pet_w"] = weight * frame["PET"].eq(1)
        frame["pea_w"] = weight * frame["FT"].eq(1)
        frame["occupied_w"] = weight * frame["occupied_status"].eq(1)
        frame["unemployed_w"] = weight * frame["unemployed_status"].eq(1)
        frame["formal_numerator_w"] = weight * (
            frame["occupied_status"].eq(1) & frame["P6920"].eq(1)
        )
        frame["formal_denominator_w"] = weight * (
            frame["occupied_status"].eq(1) & frame["P6920"].isin([1, 2])
        )

        for category, group in frame.groupby("category"):
            add_record(
                records,
                category,
                group["pet_w"].sum(),
                group["pea_w"].sum(),
                group["occupied_w"].sum(),
                group["unemployed_w"].sum(),
                group["formal_numerator_w"].sum(),
                group["formal_denominator_w"].sum(),
            )
        add_record(
            records,
            "Total",
            frame["pet_w"].sum(),
            frame["pea_w"].sum(),
            frame["occupied_w"].sum(),
            frame["unemployed_w"].sum(),
            frame["formal_numerator_w"].sum(),
            frame["formal_denominator_w"].sum(),
        )

    return records


def records_to_table(records_by_year: dict[int, dict[str, dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES + ["Total"]:
        row = {"logro_educativo": category}
        for year, records in records_by_year.items():
            record = records[category]
            row[f"participacion_{year}"] = 100 * record["pea"] / record["pet"]
            row[f"ocupacion_{year}"] = 100 * record["occupied"] / record["pet"]
            row[f"desempleo_{year}"] = 100 * record["unemployed"] / record["pea"]
            row[f"formalidad_{year}"] = (
                100 * record["formal_numerator"] / record["formal_denominator"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_decimal(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def write_outputs(table: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "tasas_laborales_educacion_cinco.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Tasas laborales por logro educativo, 2010 y 2025}",
        r"\label{tab:tasas-laborales-educacion}",
        r"\begin{threeparttable}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Participación}",
        r"& \multicolumn{2}{c}{Ocupación}",
        r"& \multicolumn{2}{c}{Desempleo}",
        r"& \multicolumn{2}{c}{Formalidad} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(l){8-9}",
        r"Logro educativo & 2010 & 2025 & 2010 & 2025 & 2010 & 2025 & 2010 & 2025 \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        category = row["logro_educativo"]
        label = rf"\textbf{{{category}}}" if category == "Total" else category
        values = [
            format_decimal(row[f"{metric}_{year}"])
            for metric in ["participacion", "ocupacion", "desempleo", "formalidad"]
            for year in [2010, 2025]
        ]
        if category == "Total":
            lines.append(r"\midrule")
        lines.append(f"{label} & " + " & ".join(values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}",
            r"\scriptsize",
            (
                r"\item \textit{Nota:} La tasa de participación se calcula como PEA/PET, "
                r"la tasa de ocupación como ocupados/PET, la tasa de desempleo como desocupados/PEA "
                r"y la tasa de formalidad como ocupados que cotizan a pensión sobre ocupados "
                r"con clasificación válida de cotización. Las tasas se reportan en porcentaje. "
                r"La agregación de cinco grupos permite calcular las tasas sobre toda la población "
                r"en edad de trabajar en 2010 y 2025. Fuente: cálculos propios con microdatos GEIH del DANE."
            ),
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    (SECTION_DIR / "tasas_laborales_educacion_cinco_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    records_by_year = {
        2010: calculate_2010(),
        2025: calculate_2025(),
    }
    table = records_to_table(records_by_year)
    write_outputs(table)

    selected = table[table["logro_educativo"].isin(["Ninguno o preescolar", "Superior o universitaria"])]
    print(table.to_string(index=False))
    low_formality = float(selected.loc[selected["logro_educativo"].eq("Ninguno o preescolar"), "formalidad_2025"].iloc[0])
    high_formality = float(selected.loc[selected["logro_educativo"].eq("Superior o universitaria"), "formalidad_2025"].iloc[0])
    print(f"Razón formalidad 2025: {high_formality / low_formality:.1f}")


if __name__ == "__main__":
    main()
