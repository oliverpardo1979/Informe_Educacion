param(
    [string]$DataPath
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workspaceRoot = (Resolve-Path (Join-Path $repoRoot "..")).Path
if (-not $DataPath) {
    $candidate = Join-Path $workspaceRoot "tmp\cine11_download\BaseconCINE11.dta"
    if (Test-Path $candidate) {
        $DataPath = $candidate
    } else {
        $candidate = Join-Path $workspaceRoot "CJC-Monitor\Datos\Processed\BaseconCINE11.dta"
        $DataPath = $candidate
    }
}

$DataPath = (Resolve-Path $DataPath).Path
$tableDir = Join-Path $repoRoot "Paper\tables"
$sectionDir = Join-Path $repoRoot "Paper\sections"
$figureDir = Join-Path $repoRoot "Paper\figures"

New-Item -ItemType Directory -Force -Path $tableDir, $sectionDir, $figureDir | Out-Null

$code = @'
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

public class Cine11LongRunBuilder {
    public class Header {
        public ushort K;
        public ulong N;
        public ulong[] Map;
        public string[] Names;
        public ushort[] Types;
        public int[] Offsets;
        public int RowLength;
        public long DataStart;
    }

    public class Stats {
        public long Rows;
        public double Ocupados;
        public double RemRows;
        public double RemOcupados;
        public double RemHoras;
        public double RemTotal;
    }

    public class SeriesRow {
        public int Year;
        public int Code;
        public string Label;
        public double Ocupados;
        public double RemOcupados;
        public double RemHoras;
        public double RemTotal;
        public double RemTrabajador;
        public double RemHora;
    }

    static readonly int[] Years = new int[] {2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2021,2022,2023,2024,2025};
    const int StartYear = 2010;
    const int EndYear = 2025;
    const double MonthlyHoursFactor = 52.0 / 12.0;

    static readonly Dictionary<int,string> Labels = new Dictionary<int,string> {
        {1, "Ninguno"},
        {2, "B\u00e1sica primaria"},
        {3, "B\u00e1sica secundaria"},
        {4, "Educaci\u00f3n media"},
        {5, "T\u00e9cnico o tecnol\u00f3gico"},
        {6, "Pregrado universitario"},
        {7, "Posgrado"},
        {98, "No determinado"}
    };

    static readonly int[] OccupationCodes = new int[] {1,2,3,4,5,6,7,98};
    static readonly int[] RemunerationCodes = new int[] {1,2,3,4,5,6,7};

    static int Find(byte[] haystack, string needle, int start) {
        byte[] pat = Encoding.ASCII.GetBytes(needle);
        for (int i = start; i <= haystack.Length - pat.Length; i++) {
            bool ok = true;
            for (int j = 0; j < pat.Length; j++) {
                if (haystack[i + j] != pat[j]) { ok = false; break; }
            }
            if (ok) return i;
        }
        return -1;
    }

    static int TypeSize(ushort t) {
        if (t >= 1 && t <= 2045) return t;
        if (t == 32768) return 8;
        if (t == 65530) return 1;
        if (t == 65529) return 2;
        if (t == 65528) return 4;
        if (t == 65527) return 4;
        if (t == 65526) return 8;
        throw new Exception("Unsupported Stata type " + t.ToString(CultureInfo.InvariantCulture));
    }

    static Header ReadHeader(string path) {
        byte[] buffer = new byte[1048576];
        using (FileStream fs = File.OpenRead(path)) {
            fs.Read(buffer, 0, buffer.Length);
        }

        int kTag = Find(buffer, "<K>", 0);
        int nTag = Find(buffer, "<N>", 0);
        int mapTag = Find(buffer, "<map>", 0);
        int vtTag = Find(buffer, "<variable_types>", 0);
        int vnTag = Find(buffer, "<varnames>", 0);

        ushort K = BitConverter.ToUInt16(buffer, kTag + 3);
        ulong N = BitConverter.ToUInt64(buffer, nTag + 3);

        ulong[] map = new ulong[14];
        int mapStart = mapTag + "<map>".Length;
        for (int i = 0; i < 14; i++) map[i] = BitConverter.ToUInt64(buffer, mapStart + 8 * i);

        ushort[] types = new ushort[K];
        int typesStart = vtTag + "<variable_types>".Length;
        for (int i = 0; i < K; i++) types[i] = BitConverter.ToUInt16(buffer, typesStart + 2 * i);

        string[] names = new string[K];
        int namesStart = vnTag + "<varnames>".Length;
        for (int i = 0; i < K; i++) {
            int off = namesStart + 129 * i;
            int len = 0;
            while (len < 129 && buffer[off + len] != 0) len++;
            names[i] = Encoding.ASCII.GetString(buffer, off, len);
        }

        int[] offsets = new int[K];
        int rowLen = 0;
        for (int i = 0; i < K; i++) {
            offsets[i] = rowLen;
            rowLen += TypeSize(types[i]);
        }

        return new Header {
            K = K,
            N = N,
            Map = map,
            Names = names,
            Types = types,
            Offsets = offsets,
            RowLength = rowLen,
            DataStart = (long)map[9] + "<data>".Length
        };
    }

    static int IndexOf(Header h, string name) {
        for (int i = 0; i < h.Names.Length; i++) if (h.Names[i] == name) return i;
        return -1;
    }

    static int ReadInt(byte[] row, int off, ushort type) {
        if (type == 65530) return row[off];
        if (type == 65529) return BitConverter.ToInt16(row, off);
        if (type == 65528) return BitConverter.ToInt32(row, off);
        throw new Exception("Not an integer type");
    }

    static double ReadNum(byte[] row, int off, ushort type) {
        if (type == 65526) return BitConverter.ToDouble(row, off);
        if (type == 65527) return BitConverter.ToSingle(row, off);
        if (type == 65528) return BitConverter.ToInt32(row, off);
        if (type == 65529) return BitConverter.ToInt16(row, off);
        if (type == 65530) return row[off];
        throw new Exception("Not a numeric type");
    }

    static bool ValidNum(double x) {
        return !(Double.IsNaN(x) || Double.IsInfinity(x) || Math.Abs(x) > 1e100);
    }

    static string CsvEscape(string value) {
        if (value == null) return "";
        if (value.Contains(",") || value.Contains("\"") || value.Contains("\n")) {
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    static string F(double value) {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    static string Dec(double value, int digits) {
        string text = value.ToString("N" + digits.ToString(CultureInfo.InvariantCulture), new CultureInfo("es-CO"));
        return text;
    }

    static string Pct(double value, int digits) {
        return Dec(100.0 * value, digits) + "\\%";
    }

    static string SignedDec(double value, int digits) {
        string text = Dec(Math.Abs(value), digits);
        if (value > 0) return "+" + text;
        if (value < 0) return "-" + text;
        return text;
    }

    static string Growth(double start, double end) {
        if (start <= 0 || end <= 0) return "";
        double g = Math.Pow(end / start, 1.0 / 15.0) - 1.0;
        return Pct(g, 1);
    }

    static string Tex(string text) {
        return text
            .Replace("\\", "\\textbackslash{}")
            .Replace("&", "\\&")
            .Replace("%", "\\%")
            .Replace("_", "\\_");
    }

    static Stats GetStats(Dictionary<string, Stats> stats, int year, int code) {
        string key = year.ToString(CultureInfo.InvariantCulture) + "|" + code.ToString(CultureInfo.InvariantCulture);
        Stats s;
        if (!stats.TryGetValue(key, out s)) {
            s = new Stats();
            stats[key] = s;
        }
        return s;
    }

    static Stats SumStats(Dictionary<string, Stats> stats, int year, IEnumerable<int> codes) {
        Stats outStats = new Stats();
        foreach (int code in codes) {
            string key = year.ToString(CultureInfo.InvariantCulture) + "|" + code.ToString(CultureInfo.InvariantCulture);
            Stats s;
            if (!stats.TryGetValue(key, out s)) continue;
            outStats.Rows += s.Rows;
            outStats.Ocupados += s.Ocupados;
            outStats.RemRows += s.RemRows;
            outStats.RemOcupados += s.RemOcupados;
            outStats.RemHoras += s.RemHoras;
            outStats.RemTotal += s.RemTotal;
        }
        return outStats;
    }

    public static void Build(string dataPath, string tableDir, string sectionDir, string figureDir) {
        Header h = ReadHeader(dataPath);
        int anioI = IndexOf(h, "anio");
        int fexI = IndexOf(h, "fex");
        int horasI = IndexOf(h, "horas");
        int ingresoI = IndexOf(h, "ingreso_hora_real");
        int cineI = IndexOf(h, "cine11_hom_cod");

        if (anioI < 0 || fexI < 0 || horasI < 0 || ingresoI < 0 || cineI < 0) {
            throw new Exception("The DTA file does not contain the required variables.");
        }

        Dictionary<string, Stats> stats = new Dictionary<string, Stats>();
        HashSet<int> yearSet = new HashSet<int>(Years);
        byte[] row = new byte[h.RowLength];

        using (FileStream fs = File.OpenRead(dataPath)) {
            fs.Seek(h.DataStart, SeekOrigin.Begin);
            for (ulong r = 0; r < h.N; r++) {
                int got = 0;
                while (got < row.Length) {
                    int n = fs.Read(row, got, row.Length - got);
                    if (n <= 0) throw new EndOfStreamException();
                    got += n;
                }

                int year = ReadInt(row, h.Offsets[anioI], h.Types[anioI]);
                if (!yearSet.Contains(year)) continue;

                int code = ReadInt(row, h.Offsets[cineI], h.Types[cineI]);
                if (!Labels.ContainsKey(code)) continue;

                double fex = ReadNum(row, h.Offsets[fexI], h.Types[fexI]);
                double horas = ReadNum(row, h.Offsets[horasI], h.Types[horasI]);
                double ingreso = ReadNum(row, h.Offsets[ingresoI], h.Types[ingresoI]);
                if (!ValidNum(fex) || fex < 0) fex = 0.0;

                Stats s = GetStats(stats, year, code);
                s.Rows++;
                s.Ocupados += fex;

                bool productive = fex > 0 && ValidNum(horas) && ValidNum(ingreso) && horas > 0 && horas <= 112 && ingreso > 0;
                if (productive) {
                    double monthlyHours = fex * horas * MonthlyHoursFactor;
                    s.RemRows++;
                    s.RemOcupados += fex;
                    s.RemHoras += monthlyHours;
                    s.RemTotal += monthlyHours * ingreso;
                }
            }
        }

        List<SeriesRow> series = new List<SeriesRow>();
        foreach (int year in Years) {
            foreach (int code in OccupationCodes) {
                Stats s = SumStats(stats, year, new int[] { code });
                double remWorker = s.RemOcupados > 0 ? s.RemTotal / s.RemOcupados : Double.NaN;
                double remHour = s.RemHoras > 0 ? s.RemTotal / s.RemHoras : Double.NaN;
                series.Add(new SeriesRow {
                    Year = year,
                    Code = code,
                    Label = Labels[code],
                    Ocupados = s.Ocupados,
                    RemOcupados = s.RemOcupados,
                    RemHoras = s.RemHoras,
                    RemTotal = s.RemTotal,
                    RemTrabajador = remWorker,
                    RemHora = remHour
                });
            }
            Stats total = SumStats(stats, year, RemunerationCodes);
            series.Add(new SeriesRow {
                Year = year,
                Code = 1000,
                Label = "Total",
                Ocupados = total.Ocupados,
                RemOcupados = total.RemOcupados,
                RemHoras = total.RemHoras,
                RemTotal = total.RemTotal,
                RemTrabajador = total.RemOcupados > 0 ? total.RemTotal / total.RemOcupados : Double.NaN,
                RemHora = total.RemHoras > 0 ? total.RemTotal / total.RemHoras : Double.NaN
            });
        }

        WriteSeriesCsv(series, Path.Combine(tableDir, "remuneracion_educacion_comparable_series.csv"));
        WriteSummaryCsv(series, Path.Combine(tableDir, "remuneracion_educacion_comparable_summary.csv"));
        WriteOccupationTable(series, Path.Combine(sectionDir, "ocupacion_educacion_comparable_table.tex"));
        WriteRemunerationTable(series, Path.Combine(sectionDir, "remuneracion_educacion_comparable_table.tex"));
        WriteDecomposition(series, Path.Combine(tableDir, "remuneracion_educacion_descomposicion.csv"), Path.Combine(tableDir, "remuneracion_educacion_descomposicion_detalle.csv"), Path.Combine(sectionDir, "descomposicion_remuneracion_educacion_table.tex"), Path.Combine(sectionDir, "descomposicion_remuneracion_educacion_detalle_table.tex"), Path.Combine(figureDir, "fig_descomposicion_remuneracion_educacion_cascada.png"));
        WriteSeriesFigure(series, Path.Combine(figureDir, "fig_remuneracion_educacion_series_comparables.png"));
    }

    static SeriesRow FindSeries(List<SeriesRow> series, int year, int code) {
        return series.First(x => x.Year == year && x.Code == code);
    }

    static void WriteSeriesCsv(List<SeriesRow> series, string path) {
        List<string> lines = new List<string>();
        lines.Add("anio,cine11_hom_cod,grupo_educativo,ocupados,rem_ocupados,horas_mensuales,rem_total_mensual,rem_por_trabajador,rem_por_hora,participacion_empleo,participacion_horas");
        foreach (SeriesRow r in series.OrderBy(x => x.Code == 1000 ? 9999 : x.Code).ThenBy(x => x.Year)) {
            double totalOcc = series.Where(x => x.Year == r.Year && x.Code != 1000 && x.Code != 98).Sum(x => x.Ocupados);
            double totalHours = series.Where(x => x.Year == r.Year && x.Code != 1000 && x.Code != 98).Sum(x => x.RemHoras);
            double empShare = r.Code == 1000 ? 1.0 : (totalOcc > 0 ? r.Ocupados / totalOcc : Double.NaN);
            double hourShare = r.Code == 1000 ? 1.0 : (totalHours > 0 ? r.RemHoras / totalHours : Double.NaN);
            lines.Add(String.Join(",", new string[] {
                r.Year.ToString(CultureInfo.InvariantCulture),
                r.Code.ToString(CultureInfo.InvariantCulture),
                CsvEscape(r.Label),
                F(r.Ocupados),
                F(r.RemOcupados),
                F(r.RemHoras),
                F(r.RemTotal),
                F(r.RemTrabajador),
                F(r.RemHora),
                F(empShare),
                F(hourShare)
            }));
        }
        File.WriteAllLines(path, lines, new UTF8Encoding(false));
    }

    static void WriteSummaryCsv(List<SeriesRow> series, string path) {
        List<string> lines = new List<string>();
        lines.Add("grupo_educativo,cine11_hom_cod,ocupados_2010,ocupados_2025,rem_trabajador_2010,rem_trabajador_2025,rem_hora_2010,rem_hora_2025,crec_ocupados,crec_rem_trabajador,crec_rem_hora");
        foreach (int code in RemunerationCodes.Concat(new int[] {1000})) {
            SeriesRow a = FindSeries(series, StartYear, code);
            SeriesRow b = FindSeries(series, EndYear, code);
            lines.Add(String.Join(",", new string[] {
                CsvEscape(a.Label),
                code.ToString(CultureInfo.InvariantCulture),
                F(a.Ocupados),
                F(b.Ocupados),
                F(a.RemTrabajador),
                F(b.RemTrabajador),
                F(a.RemHora),
                F(b.RemHora),
                F(Math.Pow(b.Ocupados / a.Ocupados, 1.0 / 15.0) - 1.0),
                F(Math.Pow(b.RemTrabajador / a.RemTrabajador, 1.0 / 15.0) - 1.0),
                F(Math.Pow(b.RemHora / a.RemHora, 1.0 / 15.0) - 1.0)
            }));
        }
        File.WriteAllLines(path, lines, new UTF8Encoding(false));
    }

    static void WriteOccupationTable(List<SeriesRow> series, string path) {
        List<string> lines = new List<string>();
        double totalA = OccupationCodes.Sum(c => FindSeries(series, StartYear, c).Ocupados);
        double totalB = OccupationCodes.Sum(c => FindSeries(series, EndYear, c).Ocupados);
        lines.Add("\\begin{table}[H]");
        lines.Add("\\centering");
        lines.Add("\\caption{Logro educativo de la poblaci\u00f3n ocupada, 2010 y 2025}");
        lines.Add("\\label{tab:remuneracion_educacion_comparable}");
        lines.Add("\\scriptsize");
        lines.Add("\\begin{tabular}{@{}p{4.0cm}rrrrrr@{}}");
        lines.Add("\\toprule");
        lines.Add("Logro educativo & \\multicolumn{3}{c}{Ocupados} & \\multicolumn{3}{c}{Participaci\u00f3n en el empleo} \\\\");
        lines.Add(" & 2010 & 2025 & Crec. anual & 2010 & 2025 & Dif. (p.p.) \\\\");
        lines.Add("\\midrule");
        foreach (int code in OccupationCodes) {
            SeriesRow a = FindSeries(series, StartYear, code);
            SeriesRow b = FindSeries(series, EndYear, code);
            double shareA = a.Ocupados / totalA;
            double shareB = b.Ocupados / totalB;
            lines.Add(String.Format(CultureInfo.InvariantCulture,
                "{0} & {1} & {2} & {3} & {4} & {5} & {6} \\\\",
                Tex(a.Label),
                Dec(a.Ocupados / 1000000.0, 1),
                Dec(b.Ocupados / 1000000.0, 1),
                Growth(a.Ocupados, b.Ocupados),
                Pct(shareA, 1),
                Pct(shareB, 1),
                SignedDec(100.0 * (shareB - shareA), 1)
            ));
        }
        lines.Add("\\midrule");
        lines.Add(String.Format(CultureInfo.InvariantCulture,
            "\\textbf{{Total}} & {0} & {1} & {2} & 100,0\\% & 100,0\\% & 0,0 \\\\",
            Dec(totalA / 1000000.0, 1),
            Dec(totalB / 1000000.0, 1),
            Growth(totalA, totalB)
        ));
        lines.Add("\\bottomrule");
        lines.Add("\\end{tabular}");
        lines.Add("\\caption*{\\footnotesize Nota: ocupados en millones de personas. La fila Total incluye el grupo no determinado y coincide con el total oficial de poblaci\u00f3n ocupada del DANE para los a\u00f1os reportados. Los cuadros de remuneraci\u00f3n y la descomposici\u00f3n excluyen el grupo no determinado. Fuente: c\u00e1lculos propios con GEIH del DANE.}");
        lines.Add("\\end{table}");
        File.WriteAllLines(path, lines, new UTF8Encoding(false));
    }

    static void WriteRemunerationTable(List<SeriesRow> series, string path) {
        List<string> lines = new List<string>();
        lines.Add("\\begin{table}[H]");
        lines.Add("\\centering");
        lines.Add("\\caption{Remuneraci\u00f3n laboral por logro educativo, 2010 y 2025}");
        lines.Add("\\label{tab:remuneracion_educacion_remuneracion}");
        lines.Add("\\label{tab:remuneracion_educacion_trabajador}");
        lines.Add("\\label{tab:remuneracion_educacion_hora}");
        lines.Add("\\scriptsize");
        lines.Add("\\setlength{\\tabcolsep}{3.3pt}");
        lines.Add("\\begin{tabular}{@{}p{4.0cm}rrrrr@{}}");
        lines.Add("\\toprule");
        lines.Add("\\multicolumn{6}{@{}l}{\\textbf{Panel A. Remuneraci\u00f3n mensual por trabajador (millones de pesos de 2025)}} \\\\");
        lines.Add("Logro educativo & 2010 & 2025 & Crec. anual & \\shortstack{Raz\u00f3n frente\\\\al menor 2010} & \\shortstack{Raz\u00f3n frente\\\\al menor 2025} \\\\");
        lines.Add("\\midrule");
        double minWorkerA = RemunerationCodes.Select(c => FindSeries(series, StartYear, c).RemTrabajador).Min();
        double minWorkerB = RemunerationCodes.Select(c => FindSeries(series, EndYear, c).RemTrabajador).Min();
        foreach (int code in RemunerationCodes) {
            SeriesRow a = FindSeries(series, StartYear, code);
            SeriesRow b = FindSeries(series, EndYear, code);
            lines.Add(String.Format(CultureInfo.InvariantCulture,
                "{0} & {1} & {2} & {3} & {4}x & {5}x \\\\",
                Tex(a.Label),
                Dec(a.RemTrabajador / 1000000.0, 1),
                Dec(b.RemTrabajador / 1000000.0, 1),
                Growth(a.RemTrabajador, b.RemTrabajador),
                Dec(a.RemTrabajador / minWorkerA, 1),
                Dec(b.RemTrabajador / minWorkerB, 1)
            ));
        }
        SeriesRow ta = FindSeries(series, StartYear, 1000);
        SeriesRow tb = FindSeries(series, EndYear, 1000);
        lines.Add("\\midrule");
        lines.Add(String.Format(CultureInfo.InvariantCulture,
            "\\textbf{{Total}} & {0} & {1} & {2} & {3}x & {4}x \\\\",
            Dec(ta.RemTrabajador / 1000000.0, 1),
            Dec(tb.RemTrabajador / 1000000.0, 1),
            Growth(ta.RemTrabajador, tb.RemTrabajador),
            Dec(ta.RemTrabajador / minWorkerA, 1),
            Dec(tb.RemTrabajador / minWorkerB, 1)
        ));
        lines.Add("\\addlinespace[0.7em]");
        lines.Add("\\multicolumn{6}{@{}l}{\\textbf{Panel B. Remuneraci\u00f3n por hora trabajada (miles de pesos de 2025 por hora)}} \\\\");
        lines.Add("Logro educativo & 2010 & 2025 & Crec. anual & \\shortstack{Raz\u00f3n frente\\\\al menor 2010} & \\shortstack{Raz\u00f3n frente\\\\al menor 2025} \\\\");
        lines.Add("\\midrule");
        double minHourA = RemunerationCodes.Select(c => FindSeries(series, StartYear, c).RemHora).Min();
        double minHourB = RemunerationCodes.Select(c => FindSeries(series, EndYear, c).RemHora).Min();
        foreach (int code in RemunerationCodes) {
            SeriesRow a = FindSeries(series, StartYear, code);
            SeriesRow b = FindSeries(series, EndYear, code);
            lines.Add(String.Format(CultureInfo.InvariantCulture,
                "{0} & {1} & {2} & {3} & {4}x & {5}x \\\\",
                Tex(a.Label),
                Dec(a.RemHora / 1000.0, 1),
                Dec(b.RemHora / 1000.0, 1),
                Growth(a.RemHora, b.RemHora),
                Dec(a.RemHora / minHourA, 1),
                Dec(b.RemHora / minHourB, 1)
            ));
        }
        lines.Add("\\midrule");
        lines.Add(String.Format(CultureInfo.InvariantCulture,
            "\\textbf{{Total}} & {0} & {1} & {2} & {3}x & {4}x \\\\",
            Dec(ta.RemHora / 1000.0, 1),
            Dec(tb.RemHora / 1000.0, 1),
            Growth(ta.RemHora, tb.RemHora),
            Dec(ta.RemHora / minHourA, 1),
            Dec(tb.RemHora / minHourB, 1)
        ));
        lines.Add("\\bottomrule");
        lines.Add("\\end{tabular}");
        lines.Add("\\caption*{\\footnotesize Nota: ``Raz\u00f3n frente al menor'' muestra cu\u00e1ntas veces representa la remuneraci\u00f3n de cada fila frente a la remuneraci\u00f3n del grupo de menor remuneraci\u00f3n en el mismo a\u00f1o. En 2010 y 2025, el grupo de menor remuneraci\u00f3n fue ninguno. El crecimiento es anualizado para 2010--2025. La serie excluye 2020. Fuente: c\u00e1lculos propios con GEIH del DANE.}");
        lines.Add("\\end{table}");
        File.WriteAllLines(path, lines, new UTF8Encoding(false));
    }

    static double ValueFor(SeriesRow row, string metric) {
        if (metric == "trabajador") return row.RemTrabajador;
        return row.RemHora;
    }

    static double WeightFor(SeriesRow row, string metric) {
        if (metric == "trabajador") return row.RemOcupados;
        return row.RemHoras;
    }

    static void WriteDecomposition(List<SeriesRow> series, string summaryPath, string detailPath, string tablePath, string detailTablePath, string figurePath) {
        List<string> summary = new List<string>();
        List<string> detail = new List<string>();
        summary.Add("indicador,cambio_total,cambio_productividad,cambio_logro,participacion_productividad,participacion_logro");
        detail.Add("indicador,componente,cine11_hom_cod,grupo_educativo,aporte,participacion");

        Dictionary<string, Tuple<double,double,double>> panels = new Dictionary<string, Tuple<double,double,double>>();

        foreach (string metric in new string[] {"trabajador", "hora"}) {
            SeriesRow totalA = FindSeries(series, StartYear, 1000);
            SeriesRow totalB = FindSeries(series, EndYear, 1000);
            double rTotalA = ValueFor(totalA, metric);
            double rTotalB = ValueFor(totalB, metric);
            double totalChange = rTotalB - rTotalA;

            double wA = RemunerationCodes.Sum(c => WeightFor(FindSeries(series, StartYear, c), metric));
            double wB = RemunerationCodes.Sum(c => WeightFor(FindSeries(series, EndYear, c), metric));
            double productivity = 0.0;
            double education = 0.0;

            foreach (int code in RemunerationCodes) {
                SeriesRow a = FindSeries(series, StartYear, code);
                SeriesRow b = FindSeries(series, EndYear, code);
                double rA = ValueFor(a, metric);
                double rB = ValueFor(b, metric);
                double sA = WeightFor(a, metric) / wA;
                double sB = WeightFor(b, metric) / wB;
                double prod = 0.5 * (sA + sB) * (rB - rA);
                double edu = 0.5 * (rA + rB) * (sB - sA);
                productivity += prod;
                education += edu;
                detail.Add(String.Join(",", new string[] {
                    metric,
                    "Mayor productividad de cada logro educativo",
                    code.ToString(CultureInfo.InvariantCulture),
                    CsvEscape(Labels[code]),
                    F(prod),
                    F(prod / totalChange)
                }));
                detail.Add(String.Join(",", new string[] {
                    metric,
                    "Mayor logro educativo de la poblaci\u00f3n ocupada",
                    code.ToString(CultureInfo.InvariantCulture),
                    CsvEscape(Labels[code]),
                    F(edu),
                    F(edu / totalChange)
                }));
            }

            summary.Add(String.Join(",", new string[] {
                metric,
                F(totalChange),
                F(productivity),
                F(education),
                F(productivity / totalChange),
                F(education / totalChange)
            }));
            panels[metric] = Tuple.Create(totalChange, productivity, education);
        }

        File.WriteAllLines(summaryPath, summary, new UTF8Encoding(false));
        File.WriteAllLines(detailPath, detail, new UTF8Encoding(false));
        WriteDecompositionTable(tablePath, panels);
        WriteDetailDecompositionTable(detailTablePath, detail);
        DrawWaterfall(figurePath, panels);
    }

    static void WriteDecompositionTable(string tablePath, Dictionary<string, Tuple<double,double,double>> panels) {
        List<string> lines = new List<string>();
        lines.Add("\\begin{table}[H]");
        lines.Add("\\centering");
        lines.Add("\\caption{Descomposici\u00f3n de la variaci\u00f3n en la remuneraci\u00f3n laboral, 2010--2025}");
        lines.Add("\\label{tab:descomposicion_remuneracion_educacion}");
        lines.Add("\\includegraphics[width=0.98\\textwidth]{Paper/figures/fig_descomposicion_remuneracion_educacion_cascada.png}");
        lines.Add("\\caption*{\\footnotesize Nota: el Panel A mide la variaci\u00f3n de la remuneraci\u00f3n mensual por trabajador en miles de pesos mensuales de 2025. El Panel B mide la variaci\u00f3n de la remuneraci\u00f3n por hora trabajada en pesos de 2025 por hora. ``Mayor productividad de cada logro educativo'' mide el aporte de los cambios de remuneraci\u00f3n dentro de cada logro educativo. ``Mayor logro educativo de la poblaci\u00f3n ocupada'' mide el aporte de los cambios en el peso relativo de esos logros. La descomposici\u00f3n usa los ponderadores promedio de 2010 y 2025. Fuente: c\u00e1lculos propios con GEIH del DANE.}");
        lines.Add("\\end{table}");
        File.WriteAllLines(tablePath, lines, new UTF8Encoding(false));
    }

    static void WriteDetailDecompositionTable(string path, List<string> detailCsv) {
        List<string> lines = new List<string>();
        lines.Add("\\begin{table}[H]");
        lines.Add("\\centering");
        lines.Add("\\caption{Aportes por logro educativo a la descomposici\u00f3n de la remuneraci\u00f3n laboral, 2010--2025}");
        lines.Add("\\label{tab:descomposicion_remuneracion_educacion_detalle}");
        lines.Add("\\scriptsize");
        lines.Add("\\begin{tabular}{@{}p{6.4cm}rr@{}}");
        lines.Add("\\toprule");
        WriteDetailPanel(lines, detailCsv, "trabajador", "Panel A. Remuneraci\u00f3n mensual por trabajador", true);
        lines.Add("\\addlinespace[0.6em]");
        WriteDetailPanel(lines, detailCsv, "hora", "Panel B. Remuneraci\u00f3n por hora trabajada", false);
        lines.Add("\\bottomrule");
        lines.Add("\\end{tabular}");
        lines.Add("\\caption*{\\footnotesize Nota: los aportes al cambio preservan el signo de cada contribuci\u00f3n. Los porcentajes dividen cada factor por el cambio total del indicador correspondiente. Fuente: c\u00e1lculos propios con GEIH del DANE.}");
        lines.Add("\\end{table}");
        File.WriteAllLines(path, lines, new UTF8Encoding(false));
    }

    static void WriteDetailPanel(List<string> lines, List<string> detailCsv, string metric, string title, bool monthly) {
        lines.Add("\\multicolumn{3}{@{}l}{\\textbf{" + title + "}} \\\\");
        if (monthly) {
            lines.Add("Factor & \\shortstack{Aporte al cambio\\\\(miles de pesos\\\\mensuales de 2025)} & \\shortstack{\\% del\\\\cambio total} \\\\");
        } else {
            lines.Add("Factor & \\shortstack{Aporte al cambio\\\\(pesos de 2025\\\\por hora)} & \\shortstack{\\% del\\\\cambio total} \\\\");
        }
        lines.Add("\\midrule");
        foreach (string component in new string[] {"Mayor productividad de cada logro educativo", "Mayor logro educativo de la poblaci\u00f3n ocupada"}) {
            double componentTotal = 0.0;
            double componentShare = 0.0;
            foreach (string csv in detailCsv.Skip(1)) {
                string[] parts = SplitCsv(csv);
                if (parts[0] != metric || parts[1] != component) continue;
                double value = Double.Parse(parts[4], CultureInfo.InvariantCulture);
                double share = Double.Parse(parts[5], CultureInfo.InvariantCulture);
                componentTotal += value;
                componentShare += share;
                double display = monthly ? value / 1000.0 : value;
                string factor = (component.StartsWith("Mayor productividad") ? "Productividad: " : "Logro educativo: ") + parts[3];
                lines.Add(String.Format(CultureInfo.InvariantCulture, "{0} & {1} & {2} \\\\", Tex(factor), Dec(display, monthly ? 1 : 0), Pct(share, 1)));
            }
            lines.Add(String.Format(CultureInfo.InvariantCulture,
                "\\textbf{{{0}}} & \\textbf{{{1}}} & \\textbf{{{2}}} \\\\",
                component.StartsWith("Mayor productividad") ? "Total productividad" : "Total logro educativo",
                Dec(monthly ? componentTotal / 1000.0 : componentTotal, monthly ? 1 : 0),
                Pct(componentShare, 1)
            ));
            lines.Add("\\addlinespace[0.35em]");
        }
    }

    static string[] SplitCsv(string line) {
        List<string> parts = new List<string>();
        StringBuilder current = new StringBuilder();
        bool quote = false;
        for (int i = 0; i < line.Length; i++) {
            char c = line[i];
            if (c == '"') { quote = !quote; continue; }
            if (c == ',' && !quote) { parts.Add(current.ToString()); current.Length = 0; }
            else current.Append(c);
        }
        parts.Add(current.ToString());
        return parts.ToArray();
    }

    static void DrawWaterfall(string path, Dictionary<string, Tuple<double,double,double>> panels) {
        int width = 1050;
        int height = 1040;
        using (System.Drawing.Bitmap bmp = new System.Drawing.Bitmap(width, height)) {
            using (System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(bmp)) {
                g.Clear(System.Drawing.Color.White);
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                using (System.Drawing.Font subtitle = new System.Drawing.Font("Arial", 25, System.Drawing.FontStyle.Bold))
                using (System.Drawing.Font label = new System.Drawing.Font("Arial", 22, System.Drawing.FontStyle.Regular))
                using (System.Drawing.Font small = new System.Drawing.Font("Arial", 20, System.Drawing.FontStyle.Regular))
                using (System.Drawing.SolidBrush text = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(25,25,25))) {
                    DrawPanel(g, panels["trabajador"], "Panel A. Variaci\u00f3n 2010-2025 en la remuneraci\u00f3n mensual por trabajador", "Miles de pesos mensuales de 2025", 70, 45, 910, 400, true, subtitle, label, small);
                    DrawPanel(g, panels["hora"], "Panel B. Variaci\u00f3n 2010-2025 en la remuneraci\u00f3n por hora trabajada", "Pesos de 2025", 70, 540, 910, 400, false, subtitle, label, small);
                }
            }
            bmp.Save(path, System.Drawing.Imaging.ImageFormat.Png);
        }
    }

    static void DrawPanel(System.Drawing.Graphics g, Tuple<double,double,double> values, string title, string unit, int x, int y, int w, int h, bool monthly, System.Drawing.Font titleFont, System.Drawing.Font labelFont, System.Drawing.Font smallFont) {
        double total = values.Item1;
        double prod = values.Item2;
        double edu = values.Item3;
        double scaleFactor = monthly ? 1000.0 : 1.0;
        double totalD = total / scaleFactor;
        double prodD = prod / scaleFactor;
        double eduD = edu / scaleFactor;
        double max = Math.Max(totalD, Math.Max(Math.Abs(prodD), Math.Abs(eduD))) * 1.25;
        if (max <= 0) max = 1.0;
        int plotX = x + 70;
        int plotY = y + 85;
        int plotW = w - 140;
        int plotH = h - 150;
        int baseline = plotY + plotH;
        Func<double,int> Y = delegate(double v) { return baseline - (int)Math.Round((v / max) * plotH); };
        int barW = 110;
        int[] xs = new int[] { plotX + 90, plotX + 290, plotX + 490 };
        System.Drawing.Color blue = System.Drawing.Color.FromArgb(41, 121, 255);
        System.Drawing.Color orange = System.Drawing.Color.FromArgb(230, 126, 34);
        System.Drawing.Color green = System.Drawing.Color.FromArgb(0, 150, 110);
        using (System.Drawing.Pen axis = new System.Drawing.Pen(System.Drawing.Color.FromArgb(150,150,150), 2))
        using (System.Drawing.Pen connector = new System.Drawing.Pen(System.Drawing.Color.FromArgb(120,120,120), 3))
        using (System.Drawing.SolidBrush text = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(25,25,25)))
        using (System.Drawing.SolidBrush blueBrush = new System.Drawing.SolidBrush(blue))
        using (System.Drawing.SolidBrush orangeBrush = new System.Drawing.SolidBrush(orange))
        using (System.Drawing.SolidBrush greenBrush = new System.Drawing.SolidBrush(green)) {
            g.DrawString(title, titleFont, text, x, y);
            g.DrawString(unit, smallFont, text, x, y + 38);
            g.DrawLine(axis, plotX, baseline, plotX + plotW, baseline);
            DrawBar(g, blueBrush, xs[0], baseline, Y(prodD), barW);
            DrawBar(g, orangeBrush, xs[1], Y(prodD), Y(prodD + eduD), barW);
            DrawBar(g, greenBrush, xs[2], baseline, Y(totalD), barW);
            g.DrawLine(connector, xs[0] + barW / 2, Y(prodD), xs[1] - barW / 2, Y(prodD));
            g.DrawLine(connector, xs[1] + barW / 2, Y(totalD), xs[2] - barW / 2, Y(totalD));
            g.DrawString("Productividad", smallFont, text, xs[0] - 70, baseline + 14);
            g.DrawString("Logro educativo", smallFont, text, xs[1] - 85, baseline + 14);
            g.DrawString("Total", smallFont, text, xs[2] - 25, baseline + 14);
            DrawValue(g, prodD, prod / total, xs[0], Y(prodD) - 35, labelFont);
            DrawValue(g, eduD, edu / total, xs[1], Y(prodD + eduD) - 35, labelFont);
            DrawValue(g, totalD, 1.0, xs[2], Y(totalD) - 35, labelFont);
        }
    }

    static void DrawBar(System.Drawing.Graphics g, System.Drawing.Brush b, int x, int y0, int y1, int w) {
        int top = Math.Min(y0, y1);
        int h = Math.Abs(y1 - y0);
        g.FillRectangle(b, x - w / 2, top, w, Math.Max(2, h));
    }

    static void DrawValue(System.Drawing.Graphics g, double value, double share, int x, int y, System.Drawing.Font font) {
        string txt = Dec(value, Math.Abs(value) >= 100 ? 0 : 1) + " (" + Dec(100.0 * share, 1) + "%)";
        using (System.Drawing.SolidBrush text = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(25,25,25))) {
            System.Drawing.SizeF size = g.MeasureString(txt, font);
            g.DrawString(txt, font, text, x - size.Width / 2, y);
        }
    }

    static void WriteSeriesFigure(List<SeriesRow> series, string path) {
        int width = 1800;
        int height = 1000;
        using (System.Drawing.Bitmap bmp = new System.Drawing.Bitmap(width, height)) {
            using (System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(bmp)) {
                g.Clear(System.Drawing.Color.White);
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                using (System.Drawing.Font title = new System.Drawing.Font("Arial", 32, System.Drawing.FontStyle.Bold))
                using (System.Drawing.Font panel = new System.Drawing.Font("Arial", 24, System.Drawing.FontStyle.Bold))
                using (System.Drawing.Font small = new System.Drawing.Font("Arial", 18, System.Drawing.FontStyle.Regular))
                using (System.Drawing.SolidBrush text = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(25,25,25))) {
                    g.DrawString("Evoluci\u00f3n de la remuneraci\u00f3n laboral por logro educativo", title, text, 70, 35);
                    DrawLinePanel(g, series, "Remuneraci\u00f3n mensual por trabajador", true, 70, 110, 760, 690, panel, small);
                    DrawLinePanel(g, series, "Remuneraci\u00f3n por hora trabajada", false, 950, 110, 760, 690, panel, small);
                    DrawLegend(g, 110, 850, small);
                }
            }
            bmp.Save(path, System.Drawing.Imaging.ImageFormat.Png);
        }
    }

    static readonly Dictionary<int,System.Drawing.Color> Colors = new Dictionary<int,System.Drawing.Color> {
        {1, System.Drawing.Color.FromArgb(89, 89, 89)},
        {2, System.Drawing.Color.FromArgb(130, 74, 30)},
        {3, System.Drawing.Color.FromArgb(0, 114, 178)},
        {4, System.Drawing.Color.FromArgb(0, 150, 110)},
        {5, System.Drawing.Color.FromArgb(230, 126, 34)},
        {6, System.Drawing.Color.FromArgb(130, 70, 180)},
        {7, System.Drawing.Color.FromArgb(213, 94, 0)},
        {1000, System.Drawing.Color.FromArgb(0, 0, 0)}
    };

    static void DrawLinePanel(System.Drawing.Graphics g, List<SeriesRow> series, string title, bool monthly, int x, int y, int w, int h, System.Drawing.Font titleFont, System.Drawing.Font smallFont) {
        int plotX = x + 75;
        int plotY = y + 60;
        int plotW = w - 110;
        int plotH = h - 120;
        int[] codes = new int[] {1000,1,2,3,4,5,6,7};
        double min = monthly ? 0.7 : 3.5;
        double max = monthly ? 8.0 : 45.0;
        double logMin = Math.Log(min);
        double logMax = Math.Log(max);
        double[] ticks = monthly
            ? new double[] {0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0}
            : new double[] {4.0, 5.0, 7.5, 10.0, 15.0, 25.0, 40.0};
        Func<int,int> X = delegate(int year) { return plotX + (int)Math.Round(((year - 2010.0) / 15.0) * plotW); };
        Func<double,int> Y = delegate(double val) {
            double safeVal = Math.Max(min, Math.Min(max, val));
            return plotY + plotH - (int)Math.Round(((Math.Log(safeVal) - logMin) / (logMax - logMin)) * plotH);
        };
        using (System.Drawing.SolidBrush text = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(25,25,25)))
        using (System.Drawing.Pen axis = new System.Drawing.Pen(System.Drawing.Color.FromArgb(140,140,140), 2))
        using (System.Drawing.Pen grid = new System.Drawing.Pen(System.Drawing.Color.FromArgb(225,225,225), 1)) {
            g.DrawString(title, titleFont, text, x, y);
            foreach (double val in ticks) {
                if (val < min || val > max) continue;
                int yy = Y(val);
                g.DrawLine(grid, plotX, yy, plotX + plotW, yy);
                g.DrawString(Dec(val, 1), smallFont, text, plotX - 65, yy - 10);
            }
            g.DrawLine(axis, plotX, plotY, plotX, plotY + plotH);
            g.DrawLine(axis, plotX, plotY + plotH, plotX + plotW, plotY + plotH);
            foreach (int year in new int[] {2010,2015,2021,2025}) {
                int xx = X(year);
                g.DrawLine(axis, xx, plotY + plotH, xx, plotY + plotH + 6);
                g.DrawString(year.ToString(CultureInfo.InvariantCulture), smallFont, text, xx - 22, plotY + plotH + 12);
            }
            foreach (int code in codes) {
                List<SeriesRow> rows = series.Where(r => r.Code == code).OrderBy(r => r.Year).ToList();
                using (System.Drawing.Pen pen = new System.Drawing.Pen(Colors[code], code == 1000 ? 4 : 3)) {
                    for (int i = 1; i < rows.Count; i++) {
                        double v0 = monthly ? rows[i-1].RemTrabajador / 1000000.0 : rows[i-1].RemHora / 1000.0;
                        double v1 = monthly ? rows[i].RemTrabajador / 1000000.0 : rows[i].RemHora / 1000.0;
                        g.DrawLine(pen, X(rows[i-1].Year), Y(v0), X(rows[i].Year), Y(v1));
                    }
                }
            }
        }
    }

    static void DrawLegend(System.Drawing.Graphics g, int x, int y, System.Drawing.Font font) {
        int[] codes = new int[] {1000,1,2,3,4,5,6,7};
        using (System.Drawing.SolidBrush text = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(25,25,25))) {
            for (int i = 0; i < codes.Length; i++) {
                int code = codes[i];
                int col = i % 4;
                int row = i / 4;
                int xx = x + col * 410;
                int yy = y + row * 48;
                using (System.Drawing.Pen pen = new System.Drawing.Pen(Colors[code], code == 1000 ? 4 : 3)) {
                    g.DrawLine(pen, xx, yy + 12, xx + 45, yy + 12);
                }
                g.DrawString(Labels.ContainsKey(code) ? Labels[code] : "Total", font, text, xx + 55, yy);
            }
        }
    }
}
'@

Add-Type -TypeDefinition $code -ReferencedAssemblies System.Drawing
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Cine11LongRunBuilder]::Build($DataPath, $tableDir, $sectionDir, $figureDir)

Write-Output "Updated long-run CINE 11 outputs from $DataPath"
