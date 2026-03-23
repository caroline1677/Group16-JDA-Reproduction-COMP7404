"""
Plot Figure-3-style grouped curves using reproduced accuracies only.

Default input: first sheet of the first .xlsx in current directory.
Expected reproduced block columns in sheet 0:
    [11..18] = [idx, task, NN, PCA, GFK, TCA, TSL, JDA]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METHODS = ["NN", "PCA", "GFK", "TCA", "TSL", "JDA"]
COLORS = {
    "NN": "#1f77b4",
    "PCA": "#ff7f0e",
    "GFK": "#2ca02c",
    "TCA": "#d62728",
    "TSL": "#9467bd",
    "JDA": "#000000",
}


def _default_excel() -> Path:
    files = sorted(Path(".").glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No .xlsx file found in current directory.")
    return files[0]


def load_reproduced_data(excel_path: Path, sheet_index: int = 0) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_index, header=None)
    repro = raw.iloc[:, [11, 12, 13, 14, 15, 16, 17, 18]].copy()
    repro.columns = ["idx", "task", *METHODS]
    repro["idx"] = pd.to_numeric(repro["idx"], errors="coerce")
    for m in METHODS:
        repro[m] = pd.to_numeric(repro[m], errors="coerce")
    repro = repro.dropna(subset=["idx"]).copy()
    repro["idx"] = repro["idx"].astype(int)
    repro = repro.sort_values("idx").reset_index(drop=True)
    return repro


def plot_reproduced_figure3(df: pd.DataFrame, output: Path, dpi: int = 220) -> None:
    panels = [
        ("USPS/MNIST/COIL", 1, 4, (40, 100)),
        ("PIE", 5, 24, (15, 90)),
        ("Office+Caltech", 25, 36, (15, 100)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (ax, (title, lo, hi, y_lim)) in enumerate(zip(axes, panels)):
        part = df[(df["idx"] >= lo) & (df["idx"] <= hi)].copy()
        # Use local dataset index in each group to match paper's Figure 3 style.
        x = (part["idx"] - lo + 1).to_numpy()

        # Panel 1 and 3: grouped bars. Panel 2: line chart.
        if i in (0, 2):
            width = 0.12
            offsets = [(j - (len(METHODS) - 1) / 2) * width for j in range(len(METHODS))]
            for j, m in enumerate(METHODS):
                y = part[m].to_numpy()
                ax.bar(x + offsets[j], y, width=width, color=COLORS[m], alpha=0.9, label=m)
            pad = (len(METHODS) / 2) * width + 0.05
            ax.set_xlim(1 - pad, (hi - lo + 1) + pad)
        else:
            for m in METHODS:
                y = part[m].to_numpy()
                ax.plot(x, y, color=COLORS[m], lw=2, marker="o", ms=3, label=m)
            ax.set_xlim(1, hi - lo + 1)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Dataset Index")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(*y_lim)
        ax.set_xticks(range(1, hi - lo + 2))
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.suptitle("Reproduced Accuracy Only", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure-3-style curves from reproduced accuracies.")
    parser.add_argument("--excel", type=str, default=None, help="Excel path.")
    parser.add_argument("--sheet-index", type=int, default=0, help="Sheet index.")
    parser.add_argument("--output", type=str, default="figure3_reproduced.png", help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=220, help="Image DPI.")
    args = parser.parse_args()

    excel = Path(args.excel) if args.excel else _default_excel()
    df = load_reproduced_data(excel, args.sheet_index)
    plot_reproduced_figure3(df, Path(args.output), dpi=args.dpi)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
