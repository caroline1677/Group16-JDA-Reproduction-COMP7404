"""
Figure 3 style (paper vs reproduced) comparison plot.

Data source:
- Workbook sheet 0 ("原文数据")
- Paper block: columns [0..7] -> [idx, task, NN, PCA, GFK, TCA, TSL, JDA]
- Reproduced block: columns [11..18] -> [idx, task, NN, PCA, GFK, TCA, TSL, JDA]

Usage:
    python plot_figure3_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

METHODS = ["NN", "PCA", "GFK", "TCA", "TSL", "JDA"]
COLORS = {
    "NN": "#1f77b4",
    "PCA": "#ff7f0e",
    "GFK": "#2ca02c",
    "TCA": "#d62728",
    "TSL": "#9467bd",
    "JDA": "#000000",
}
PANELS = [
    ("USPS/MNIST/COIL (1-4)", 1, 4, (15, 95)),
    ("PIE (5-24)", 5, 24, (10, 95)),
    ("Office+Caltech (25-36)", 25, 36, (15, 95)),
]


def _find_default_excel() -> Path:
    files = sorted(Path(".").glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No .xlsx file found in current directory.")
    return files[0]


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_accuracy_data(excel_path: Path, sheet_index: int = 0, max_index: int = 36) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_index, header=None)

    paper = raw.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].copy()
    paper.columns = ["idx", "task", *METHODS]
    paper["idx"] = _to_num(paper["idx"])
    for m in METHODS:
        paper[m] = _to_num(paper[m])
    paper = paper.dropna(subset=["idx"]).copy()
    paper["idx"] = paper["idx"].astype(int)

    repro = raw.iloc[:, [11, 12, 13, 14, 15, 16, 17, 18]].copy()
    repro.columns = ["idx", "task_repro", *[f"{m}_repro" for m in METHODS]]
    repro["idx"] = _to_num(repro["idx"])
    for m in METHODS:
        repro[f"{m}_repro"] = _to_num(repro[f"{m}_repro"])
    repro = repro.dropna(subset=["idx"]).copy()
    repro["idx"] = repro["idx"].astype(int)

    merged = paper.merge(repro.drop(columns=["task_repro"]), on="idx", how="inner")
    merged = merged[(merged["idx"] >= 1) & (merged["idx"] <= max_index)]
    merged = merged.sort_values("idx").reset_index(drop=True)
    return merged


def plot_figure3_comparison(df: pd.DataFrame, output: Path, dpi: int = 220) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, (title, lo, hi, y_lim) in zip(axes, PANELS):
        part = df[(df["idx"] >= lo) & (df["idx"] <= hi)].copy()
        if part.empty:
            ax.set_title(f"{title} (no data)")
            continue

        # Use local index inside each group to mimic Figure 3.
        x = (part["idx"] - lo + 1).to_numpy()

        for m in METHODS:
            ax.plot(
                x,
                part[m].to_numpy(),
                color=COLORS[m],
                lw=1.8,
                marker="o",
                ms=3.2,
            )
            ax.plot(
                x,
                part[f"{m}_repro"].to_numpy(),
                color=COLORS[m],
                lw=1.5,
                ls="--",
                marker="x",
                ms=3.0,
                alpha=0.95,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Local Dataset Index")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(1, hi - lo + 1)
        ax.set_ylim(*y_lim)
        ax.grid(alpha=0.25, linewidth=0.7)

    method_handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=m) for m in METHODS]
    style_handles = [
        Line2D([0], [0], color="gray", lw=2, ls="-", marker="o", ms=4, label="Paper"),
        Line2D([0], [0], color="gray", lw=2, ls="--", marker="x", ms=4, label="Reproduced"),
    ]
    fig.legend(
        handles=method_handles + style_handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Accuracy Comparison", y=1.02, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(output, dpi=dpi, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 3 style paper vs reproduced comparison.")
    parser.add_argument("--excel", type=str, default=None, help="Excel file path.")
    parser.add_argument("--sheet-index", type=int, default=0, help="Sheet index, default=0.")
    parser.add_argument("--max-index", type=int, default=36, help="Maximum dataset index included.")
    parser.add_argument("--output", type=str, default="figure3_comparison.png", help="Output image path.")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI.")
    args = parser.parse_args()

    excel_path = Path(args.excel) if args.excel else _find_default_excel()
    data = load_accuracy_data(excel_path, sheet_index=args.sheet_index, max_index=args.max_index)
    plot_figure3_comparison(data, output=Path(args.output), dpi=args.dpi)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
