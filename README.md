# LAS_Density_to_Cell_Size

Point-density analyzer for LAS/LAZ collections that recommends DSM/DTM raster cell sizes. A Tk-based workflow lets you interactively pick input point clouds and an output folder; the script wraps PDAL to gather file metadata, compute point densities, and estimate reasonable raster resolutions for both surface (all points) and ground-only products.

## Features

- GUI selection for input LAS/LAZ files plus output directory (no CLI arguments required).
- PDAL-driven stats: bounding boxes, point counts, and class counts.
- Calculates points-per-square-metre, nominal spacing, and cell sizes for 1-, 4-, and 9-point-per-pixel targets.
- Produces CSV, Markdown, and log outputs summarizing each file and aggregated averages.
- Reports DSM (all returns) and DTM (ground class) recommendations, including warnings when no ground points are present.
- Helper GUI (`list_las_classes.py`) that lists all classification IDs/counts within any LAS/LAZ file.

## Requirements

- Conda environment containing:
  - Python 3.9+
  - PDAL (with `pdal` Python bindings installed)
  - GDAL (if you plan to use downstream tools in this toolkit)
  - Tkinter (ships with the default CPython build on Windows)

Example environment creation:

```bash
conda create -n lidar-tools python=3.11 pdal gdal
conda activate lidar-tools
```

## Usage

1. Activate the environment that contains PDAL and Tk (e.g. `conda activate lidar-tools`).
2. Run the analyzer:
   ```bash
   python LAS_Density_to_Cell_Size.py
   ```
3. In the first dialog, select one or more LAS/LAZ files (Ctrl/Shift-click to select multiple).
4. In the second dialog, choose the output directory for reports/logs.
5. Allow PDAL to crunch stats. A message box appears when processing completes.

### Outputs

All outputs land in the directory you selected:

| File | Description |
| --- | --- |
| `density_summary.csv` | Tabular per-file stats plus averages for DSM (all points) and DTM (class 2) scenarios. |
| `density_summary.md` | Human-readable report including suggested cell sizes (4-pt) for DSM/DTM and notes about missing ground classes. |
| `density_analysis.log` | Detailed PDAL/logging output per file for troubleshooting. |

`list_las_classes.py` shares the same GUI style and simply enumerates classification IDs and counts for quick sanity checks before running the full analysis:

```bash
python list_las_classes.py
```

## Notes

- No paths are hard-coded; every run prompts for input/output locations.
- PDAL pipelines remain in native CRS/units—no reprojection happens here.
- Ground-only recommendations rely on ASPRS class `2` by default. Adjust `GROUND_CLASSIFICATION` in `LAS_Density_to_Cell_Size.py` if your data uses a different class ID.

## License

MIT (update as needed for your organization).
