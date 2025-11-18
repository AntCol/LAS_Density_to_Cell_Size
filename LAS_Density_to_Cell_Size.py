"""LiDAR density analyzer with GUI-based LAS/LAZ selection."""
from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from typing import Iterable, List, Sequence

import pdal


LOG_NAME = "density_analysis.log"
CSV_NAME = "density_summary.csv"
MD_NAME = "density_summary.md"
GROUND_CLASSIFICATION = 2


@dataclass
class DensityResult:
    path: Path
    label: str
    point_count: int
    area_m2: float
    points_per_m2: float
    nominal_spacing: float
    cell_1pt: float
    cell_4pt: float
    cell_9pt: float


def configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def select_files(initial_dir: Path | None = None) -> List[Path]:
    root = Tk()
    root.withdraw()
    root.update()
    file_paths = filedialog.askopenfilenames(
        title="Select LAS/LAZ files",
        initialdir=str(initial_dir) if initial_dir else None,
        filetypes=[("LiDAR", "*.las *.laz"), ("All files", "*.*")],
    )
    root.update()
    root.destroy()
    return [Path(p) for p in file_paths]


def select_output_directory(initial_dir: Path | None = None) -> Path | None:
    root = Tk()
    root.withdraw()
    root.update()
    directory = filedialog.askdirectory(
        title="Select output directory",
        initialdir=str(initial_dir) if initial_dir else None,
        mustexist=True,
    )
    root.update()
    root.destroy()
    return Path(directory) if directory else None


def run_pdal_info(lidar_path: Path) -> dict:
    logging.info("Running PDAL info on %s", lidar_path)
    try:
        result = subprocess.run(
            ["pdal", "info", "--summary", str(lidar_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.error("PDAL info failed for %s: %s", lidar_path, exc.stderr.strip())
        raise RuntimeError(f"PDAL info failed for {lidar_path}") from exc

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logging.error("Unable to parse PDAL output for %s", lidar_path)
        raise RuntimeError("Invalid PDAL JSON output") from exc


def _extract_bounds(bounds: dict) -> tuple[float, float, float, float]:
    if all(k in bounds for k in ("minx", "maxx", "miny", "maxy")):
        return (
            float(bounds["minx"]),
            float(bounds["maxx"]),
            float(bounds["miny"]),
            float(bounds["maxy"]),
        )
    if "X" in bounds and "Y" in bounds:
        x = bounds["X"]
        y = bounds["Y"]
        return float(x["min"]), float(x["max"]), float(y["min"]), float(y["max"])
    if "minimum" in bounds and "maximum" in bounds:
        mn = bounds["minimum"]
        mx = bounds["maximum"]
        return float(mn["x"]), float(mx["x"]), float(mn["y"]), float(mx["y"])
    raise KeyError("Bounds dictionary missing expected keys")


def build_density_result(
    lidar_path: Path, label: str, point_count: int, area_m2: float
) -> DensityResult:
    if area_m2 == 0:
        raise RuntimeError(f"Computed area is zero for {lidar_path}")
    points_per_m2 = point_count / area_m2
    if points_per_m2 <= 0:
        raise RuntimeError(f"Non-positive density for {lidar_path}")
    nominal_spacing = math.sqrt(1 / points_per_m2)

    def cell_size(points_per_pixel: int) -> float:
        return math.sqrt(points_per_pixel / points_per_m2)

    return DensityResult(
        path=lidar_path,
        label=label,
        point_count=point_count,
        area_m2=area_m2,
        points_per_m2=points_per_m2,
        nominal_spacing=nominal_spacing,
        cell_1pt=cell_size(1),
        cell_4pt=cell_size(4),
        cell_9pt=cell_size(9),
    )


def parse_summary(summary: dict, lidar_path: Path) -> tuple[int, float]:
    try:
        bounds = summary["summary"]["bounds"]
        count = int(summary["summary"]["num_points"])
        minx, maxx, miny, maxy = _extract_bounds(bounds)
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(f"PDAL summary missing data for {lidar_path}") from exc

    area = max((maxx - minx) * (maxy - miny), 0.0)
    return count, area


def summarize(results: Sequence[DensityResult]) -> DensityResult:
    def avg(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else float("nan")

    if not results:
        raise ValueError("Cannot summarize empty results")

    return DensityResult(
        path=Path("AVERAGE"),
        label=results[0].label,
        point_count=int(avg(res.point_count for res in results)),
        area_m2=avg(res.area_m2 for res in results),
        points_per_m2=avg(res.points_per_m2 for res in results),
        nominal_spacing=avg(res.nominal_spacing for res in results),
        cell_1pt=avg(res.cell_1pt for res in results),
        cell_4pt=avg(res.cell_4pt for res in results),
        cell_9pt=avg(res.cell_9pt for res in results),
    )


def write_csv(
    all_results: Sequence[DensityResult],
    ground_results: Sequence[DensityResult],
    csv_path: Path,
) -> None:
    logging.info("Writing CSV to %s", csv_path)

    def write_block(writer, title: str, results: Sequence[DensityResult]):
        writer.writerow([title])
        writer.writerow(
            [
                "File",
                "Point Count",
                "Area (m^2)",
                "Points per m^2",
                "Nominal Spacing (m)",
                "Cell Size 1pt (m)",
                "Cell Size 4pt (m)",
                "Cell Size 9pt (m)",
            ]
        )
        for res in results:
            writer.writerow(
                [
                    res.path.name,
                    res.point_count,
                    f"{res.area_m2:.3f}",
                    f"{res.points_per_m2:.3f}",
                    f"{res.nominal_spacing:.3f}",
                    f"{res.cell_1pt:.3f}",
                    f"{res.cell_4pt:.3f}",
                    f"{res.cell_9pt:.3f}",
                ]
            )
        avg = summarize(results)
        writer.writerow(
            [
                "AVERAGE",
                avg.point_count,
                f"{avg.area_m2:.3f}",
                f"{avg.points_per_m2:.3f}",
                f"{avg.nominal_spacing:.3f}",
                f"{avg.cell_1pt:.3f}",
                f"{avg.cell_4pt:.3f}",
                f"{avg.cell_9pt:.3f}",
            ]
        )

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        write_block(
            writer,
            "All Points (DSM reference)",
            all_results,
        )
        writer.writerow([])
        if ground_results:
            write_block(
                writer,
                f"Ground Only (Class {GROUND_CLASSIFICATION}) - DTM reference",
                ground_results,
            )
        else:
            writer.writerow(
                [
                    f"Ground Only (Class {GROUND_CLASSIFICATION}) - DTM reference",
                    "No ground-classified points detected in selection.",
                ]
            )


def write_markdown(
    all_results: Sequence[DensityResult],
    ground_results: Sequence[DensityResult],
    missing_ground_files: Sequence[Path],
    md_path: Path,
) -> None:
    logging.info("Writing Markdown report to %s", md_path)
    avg_all = summarize(all_results)
    lines: List[str] = [
        "# LiDAR Density Analysis",
        "",
        "## All Points Summary (DSM reference)",
        "",
        "| File | Points | Area (m^2) | Points/m^2 | Nominal Spacing (m) | Cell 1pt (m) | Cell 4pt (m) | Cell 9pt (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for res in all_results:
        lines.append(
            f"| {res.path.name} | {res.point_count:,} | {res.area_m2:.2f} | "
            f"{res.points_per_m2:.2f} | {res.nominal_spacing:.3f} | "
            f"{res.cell_1pt:.3f} | {res.cell_4pt:.3f} | {res.cell_9pt:.3f} |"
        )
    lines.append(
        "| **AVERAGE** | "
        f"{avg_all.point_count:,} | {avg_all.area_m2:.2f} | {avg_all.points_per_m2:.2f} | "
        f"{avg_all.nominal_spacing:.3f} | {avg_all.cell_1pt:.3f} | {avg_all.cell_4pt:.3f} | "
        f"{avg_all.cell_9pt:.3f} |"
    )

    avg_ground = summarize(ground_results) if ground_results else None
    if avg_ground:
        lines.extend(
            [
                "",
                f"## Ground Classification Only (Class {GROUND_CLASSIFICATION}) - DTM reference",
                "",
                "| File | Points | Area (m^2) | Points/m^2 | Nominal Spacing (m) | Cell 1pt (m) | Cell 4pt (m) | Cell 9pt (m) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for res in ground_results:
            lines.append(
                f"| {res.path.name} | {res.point_count:,} | {res.area_m2:.2f} | "
                f"{res.points_per_m2:.2f} | {res.nominal_spacing:.3f} | "
                f"{res.cell_1pt:.3f} | {res.cell_4pt:.3f} | {res.cell_9pt:.3f} |"
            )
        lines.append(
            "| **AVERAGE** | "
            f"{avg_ground.point_count:,} | {avg_ground.area_m2:.2f} | {avg_ground.points_per_m2:.2f} | "
            f"{avg_ground.nominal_spacing:.3f} | {avg_ground.cell_1pt:.3f} | {avg_ground.cell_4pt:.3f} | "
            f"{avg_ground.cell_9pt:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Suggested Grid Cell Sizes",
            "",
            f"- **DSM (all points, 4-pt cells):** {avg_all.cell_4pt:.3f} m",
        ]
    )
    if avg_ground:
        lines.append(
            f"- **DTM (ground class {GROUND_CLASSIFICATION}, 4-pt cells):** {avg_ground.cell_4pt:.3f} m"
        )
    else:
        lines.append(
            f"- **DTM (ground class {GROUND_CLASSIFICATION}):** No ground-classified points detected."
        )

    if missing_ground_files:
        lines.extend(
            [
                "",
                "### Files without ground points",
                "",
            ]
        )
        lines.extend(f"- {p.name}" for p in missing_ground_files)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Nominal spacing approximates the average ground distance between points. "
                "For rasterization, a 1-point cell roughly matches the native spacing; "
                "4- and 9-point cells introduce smoothing suitable for DTM/DSM generation."
            ),
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
def log_console_summary(
    all_results: Sequence[DensityResult], ground_results: Sequence[DensityResult]
) -> None:
    header = (
        f"{'File':40} {'Points/m^2':>12} {'Spacing(m)':>12} "
        f"{'Cell1':>8} {'Cell4':>8} {'Cell9':>8}"
    )
    logging.info("All points (DSM reference)")
    logging.info(header)
    for res in all_results:
        logging.info(
            f"{res.path.name:40} "
            f"{res.points_per_m2:12.2f} {res.nominal_spacing:12.3f} "
            f"{res.cell_1pt:8.3f} {res.cell_4pt:8.3f} {res.cell_9pt:8.3f}"
        )
    avg = summarize(all_results)
    logging.info(
        f"{'AVERAGE':40} "
        f"{avg.points_per_m2:12.2f} {avg.nominal_spacing:12.3f} "
        f"{avg.cell_1pt:8.3f} {avg.cell_4pt:8.3f} {avg.cell_9pt:8.3f}"
    )
    if ground_results:
        logging.info("Ground only (DTM reference)")
        logging.info(header)
        for res in ground_results:
            logging.info(
                f"{res.path.name:40} "
                f"{res.points_per_m2:12.2f} {res.nominal_spacing:12.3f} "
                f"{res.cell_1pt:8.3f} {res.cell_4pt:8.3f} {res.cell_9pt:8.3f}"
            )
        avg_g = summarize(ground_results)
        logging.info(
            f"{'AVERAGE':40} "
            f"{avg_g.points_per_m2:12.2f} {avg_g.nominal_spacing:12.3f} "
            f"{avg_g.cell_1pt:8.3f} {avg_g.cell_4pt:8.3f} {avg_g.cell_9pt:8.3f}"
        )
    else:
        logging.info(
            "No ground-classified points detected. Skipping DTM summary."
        )


def get_classification_counts(lidar_path: Path) -> dict[int, int]:
    """Return classification counts using PDAL's Python API."""
    pipeline_spec = [
        {"type": "readers.las", "filename": str(lidar_path)},
        {
            "type": "filters.stats",
            "dimensions": "Classification",
            "count": "Classification",
        },
    ]
    logging.info("Computing classification counts for %s", lidar_path)
    pipeline = pdal.Pipeline(json.dumps(pipeline_spec))
    try:
        pipeline.execute()
    except RuntimeError as exc:
        raise RuntimeError(f"PDAL stats pipeline failed for {lidar_path}") from exc

    metadata = pipeline.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    try:
        stats = metadata["metadata"]["filters.stats"]["statistic"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Unable to locate stats metadata") from exc

    classification_counts: list[str] | None = None
    for entry in stats:
        if isinstance(entry, dict) and entry.get("name", "").lower() == "classification":
            counts_field = entry.get("counts")
            if isinstance(counts_field, list):
                classification_counts = counts_field
                break
    if not classification_counts:
        return {}

    counts: dict[int, int] = {}
    for raw in classification_counts:
        try:
            class_id_str, count_str = raw.split("/")
            class_id = int(float(class_id_str))
            counts[class_id] = counts.get(class_id, 0) + int(float(count_str))
        except (ValueError, AttributeError):
            continue
    return counts


def ground_point_count(lidar_path: Path) -> int:
    counts = get_classification_counts(lidar_path)
    return counts.get(GROUND_CLASSIFICATION, 0)


def main() -> None:
    files = select_files()
    if not files:
        messagebox.showwarning("LAS Density", "No LAS/LAZ files selected. Exiting.")
        return

    output_dir = select_output_directory(initial_dir=files[0].parent)
    if not output_dir:
        messagebox.showwarning("LAS Density", "No output directory selected. Exiting.")
        return

    configure_logging(output_dir / LOG_NAME)

    results_all: List[DensityResult] = []
    results_ground: List[DensityResult] = []
    missing_ground: List[Path] = []
    for lidar_path in files:
        try:
            summary = run_pdal_info(lidar_path)
            total_points, area = parse_summary(summary, lidar_path)
            results_all.append(
                build_density_result(
                    lidar_path, "All Points", total_points, area
                )
            )
            try:
                ground_count = ground_point_count(lidar_path)
            except Exception as ground_exc:  # pylint: disable=broad-except
                logging.exception(
                    "Failed to extract ground classification info for %s: %s",
                    lidar_path,
                    ground_exc,
                )
                missing_ground.append(lidar_path)
                continue

            if ground_count > 0:
                results_ground.append(
                    build_density_result(
                        lidar_path,
                        f"Class {GROUND_CLASSIFICATION}",
                        ground_count,
                        area,
                    )
                )
            else:
                missing_ground.append(lidar_path)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to process %s: %s", lidar_path, exc)

    if not results_all:
        logging.error("No valid LAS/LAZ files processed. Nothing to report.")
        messagebox.showerror("LAS Density", "All selected files failed to process.")
        return

    write_csv(results_all, results_ground, output_dir / CSV_NAME)
    write_markdown(
        results_all,
        results_ground,
        missing_ground,
        output_dir / MD_NAME,
    )
    log_console_summary(results_all, results_ground)
    messagebox.showinfo(
        "LAS Density",
        f"Analysis complete.\nOutputs saved to:\n{output_dir}",
    )


if __name__ == "__main__":
    main()
