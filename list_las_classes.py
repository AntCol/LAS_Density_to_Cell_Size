"""Quick GUI helper to list LAS/LAZ classification counts."""
from __future__ import annotations

import sys
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LAS_Density_to_Cell_Size.LAS_Density_to_Cell_Size import (  # noqa: E402
    get_classification_counts,
)


def select_file() -> Path | None:
    root = Tk()
    root.withdraw()
    root.update()
    filepath = filedialog.askopenfilename(
        title="Select LAS/LAZ file to inspect",
        filetypes=[("LiDAR", "*.las *.laz"), ("All files", "*.*")],
    )
    root.update()
    root.destroy()
    return Path(filepath) if filepath else None


def main() -> None:
    lidar_path = select_file()
    if not lidar_path:
        messagebox.showwarning("LAS Class List", "No file selected.")
        return
    try:
        counts = get_classification_counts(lidar_path)
    except Exception as exc:  # pylint: disable=broad-except
        messagebox.showerror("LAS Class List", f"Failed to read classes:\n{exc}")
        return

    if not counts:
        messagebox.showinfo(
            "LAS Class List",
            f"No classification counts reported for\n{lidar_path.name}",
        )
        return

    lines = [f"Class counts for {lidar_path}:"]
    lines.extend(f"  Class {cls}: {count:,}" for cls, count in sorted(counts.items()))
    summary = "\n".join(lines)
    print(summary, file=sys.stdout)
    messagebox.showinfo("LAS Class List", summary)


if __name__ == "__main__":
    main()

