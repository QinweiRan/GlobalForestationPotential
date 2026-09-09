"""
This script generates Fig. 4 in Ran et al., showing:
- the global forestation potential area (Mha) 
  and living biomass carbon potential (Pg C), 
  starting from a biophysical (structural) baseline and
  progressively constrained by sequential ecological safeguards and
  socioeconomic scenarios.


Requirements:
    - Python >= 3.8
    - matplotlib >= 3.5
    - numpy >= 1.20
    - pandas >= 1.3

Author: QINWEI
Email: qwran@pku.edu.cn
Date: Sep. 2026
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#  SETTINGS
CONFIG = {
    "csv_path": "data/Fig_4_biophys_ecosocio_potential.csv",

    "output_dir": "output/",
    "output_filename": "Fig_4_biophys_ecosocio_potential.png",
    "output_dpi": 300,

    "fig_width": 11,
    "fig_height": 5.5,

    "area_ymax": 680,          
    "carbon_ymax": 45,         
    "carbon_tick_step": 10,

    "subbar_width": 0.34,
    "subbar_offset": 0.20,
}

plt.rcParams["font.sans-serif"] = "Arial"


#  STYLE CONSTANTS
COLOR_BASELINE        = "#1D9E75"   
COLOR_REDUCTION       = "#C6DCF0"   
COLOR_REDUCTION_EDGE  = "#7FA9D4"
COLOR_PIVOT           = "#2E75B6"  
COLOR_PIVOT_EDGE      = "#1F5C99"
COLOR_SCENARIO        = "#F0997B"
COLOR_SCENARIO_EDGE   = "#C77452"
COLOR_COMBINED        = "#C84F2A"
COLOR_COMBINED_EDGE   = "#8C3218"

COLOR_SEPARATOR    = "#9A9A9A"  
COLOR_CONNECTOR    = "#6B6960"   
COLOR_CONNECTOR_C  = "#9A9890"   

CONNECTOR_LINEWIDTH = 0.9
CONNECTOR_LINESTYLE = (0, (3, 2))  

CARBON_LIGHTEN   = 0.55
CARBON_HATCH     = "///"
HATCH_LINEWIDTH  = 0.6

ERRORBAR_COLOR   = "#5F5F5F"    
ERRORBAR_LW      = 1.1
ERRORBAR_CAPSIZE = 3.5
ENDPOINT_COLOR   = "#7A2E18"    
ENDPOINT_SIZE    = 3.6

# ---- Font sizes ----
CARBON_LABEL_FONTSIZE = 10
AXIS_LABEL_FONTSIZE   = 13
TICK_FONTSIZE         = 12
BANNER_FONTSIZE       = 14
LEGEND_FONTSIZE       = 12

mpl.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH


#  FUNCTIONS
def lighten(color, amount):
    """Interpolate a color toward white by `amount` (0-1)."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount)


def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.sort_values("order_in_plot").reset_index(drop=True)
    return df


def compute_geometry(df, value_col):
    """Waterfall geometry: reduction segments are floating bars
    (bottom = current value, top = previous value); all other bars start at 0."""
    n = len(df)
    bottom = np.zeros(n)
    top    = np.zeros(n)
    for i, row in df.iterrows():
        if row["category"] == "pixel_sequential":
            bottom[i], top[i] = row[value_col], df.loc[i - 1, value_col]
        else:
            bottom[i], top[i] = 0, row[value_col]
    return bottom, top


def category_colors(df):
    """Return fill and edge colors of the area bars, by category."""
    fills, edges = [], []
    for _, row in df.iterrows():
        cat = row["category"]
        if cat == "baseline":
            c, e = COLOR_BASELINE, COLOR_BASELINE
        elif cat == "pivot":
            c, e = COLOR_PIVOT, COLOR_PIVOT_EDGE
        elif cat == "pixel_sequential":
            c, e = COLOR_REDUCTION, COLOR_REDUCTION_EDGE
        elif cat == "national_scenario":
            c, e = COLOR_SCENARIO, COLOR_SCENARIO_EDGE
        elif cat == "national_combined":
            c, e = COLOR_COMBINED, COLOR_COMBINED_EDGE
        else:
            c, e = COLOR_BASELINE, COLOR_BASELINE
        fills.append(c)
        edges.append(e)
    return fills, edges


#  PLOTTING FUNCTION
def plot_cascade(df, config):
    FIG_WIDTH     = config["fig_width"]
    FIG_HEIGHT    = config["fig_height"]
    DPI           = config["output_dpi"]
    Y_MAX         = config["area_ymax"]
    SUBBAR_WIDTH  = config["subbar_width"]
    SUBBAR_OFFSET = config["subbar_offset"]
    CARBON_YMAX      = config["carbon_ymax"]
    CARBON_TICK_STEP = config["carbon_tick_step"]

    n = len(df)
    area_bottom, area_top     = compute_geometry(df, "area_mha")
    carbon_bottom, carbon_top = compute_geometry(df, "carbon_PgC")
    area_fill, area_edge      = category_colors(df)
    carbon_fill = [lighten(c, CARBON_LIGHTEN) for c in area_fill]
    carbon_edge = area_edge

    idx_baseline = df.index[df["category"] == "baseline"][0]
    idx_seq      = df.index[df["category"] == "pixel_sequential"].tolist()
    idx_pivot    = df.index[df["category"] == "pivot"][0]
    idx_combined = df.index[df["category"] == "national_combined"][0]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    x = np.arange(n)

    area_x, carbon_x, aw = x - SUBBAR_OFFSET, x + SUBBAR_OFFSET, SUBBAR_WIDTH
    half_w = aw / 2.0

    # ---------- Area bars (left axis) ----------
    ax.bar(area_x, area_top - area_bottom, bottom=area_bottom,
           width=aw, color=area_fill, edgecolor=area_edge,
           linewidth=0.6, zorder=3)

    # ---------- Carbon bars (right axis) ----------
    ax2 = ax.twinx()
    ax2.patch.set_visible(False)
    ax2.set_ylim(0, CARBON_YMAX)
    bars_c = ax2.bar(
        carbon_x, carbon_top - carbon_bottom, bottom=carbon_bottom,
        width=aw, color=carbon_fill, edgecolor=carbon_edge,
        linewidth=0.6, hatch=CARBON_HATCH, zorder=3,
    )
    for b, e in zip(bars_c, carbon_edge):
        b.set_edgecolor(e)

    # ---------- Error bars + endpoint markers ----------
    for i, row in df.iterrows():
        if row["category"] == "pixel_sequential":
            continue
        ax.errorbar(area_x[i], area_top[i], yerr=row["error"],
                    fmt="none", ecolor=ERRORBAR_COLOR, elinewidth=ERRORBAR_LW,
                    capsize=ERRORBAR_CAPSIZE, capthick=ERRORBAR_LW, zorder=5)
        ax.plot(area_x[i], area_top[i], "o", color=ENDPOINT_COLOR,
                markersize=ENDPOINT_SIZE, zorder=6)
        ax2.errorbar(carbon_x[i], carbon_top[i], yerr=row["carbon_error"],
                     fmt="none", ecolor=ERRORBAR_COLOR, elinewidth=ERRORBAR_LW,
                     capsize=ERRORBAR_CAPSIZE, capthick=ERRORBAR_LW, zorder=5)
        ax2.plot(carbon_x[i], carbon_top[i], "o", color=ENDPOINT_COLOR,
                 markersize=ENDPOINT_SIZE, zorder=6)

    # ---------- Waterfall connector lines ----------
    seq_and_baseline = [idx_baseline] + idx_seq + [idx_pivot]
    for k in range(len(seq_and_baseline) - 1):
        i, j = seq_and_baseline[k], seq_and_baseline[k + 1]
        ax.plot([area_x[i] + half_w, area_x[j] - half_w],
                [area_top[j], area_top[j]],
                color=COLOR_CONNECTOR, linewidth=CONNECTOR_LINEWIDTH,
                linestyle=CONNECTOR_LINESTYLE, zorder=2)
        ax2.plot([carbon_x[i] + half_w, carbon_x[j] - half_w],
                 [carbon_top[j], carbon_top[j]],
                 color=COLOR_CONNECTOR_C, linewidth=CONNECTOR_LINEWIDTH,
                 linestyle=CONNECTOR_LINESTYLE, alpha=0.9, zorder=2)

    # ---------- Vertical separator ----------
    ax.axvline(idx_pivot + 0.5, color=COLOR_SEPARATOR, linestyle=(0, (4, 3)),
               linewidth=1.0, alpha=0.7, zorder=2)

    # ---------- Section banners ----------
    ax.set_ylim(0, Y_MAX)
    banner_y = Y_MAX * 0.995

    ax.text(idx_baseline, banner_y,
            "Biophysical\nbaseline",
            ha="center", va="center",
            fontsize=BANNER_FONTSIZE,
            color="#000000",
            style="italic",
            fontweight="bold")

    if idx_seq:
        ax.text((idx_seq[0] + idx_seq[-1]) / 2 + 0.45, banner_y,
                "Ecological safeguards",
                ha="center", va="center",
                fontsize=BANNER_FONTSIZE,
                color="#000000",
                style="italic",
                fontweight="bold")

    ax.text((idx_pivot + idx_combined) / 2 + 0.45, banner_y,
            "Socioeconomic scenarios",
            ha="center", va="center",
            fontsize=BANNER_FONTSIZE,
            color="#000000",
            style="italic",
            fontweight="bold")

    # ---------- X-axis ----------
    ax.set_xticks(x)
    ax.set_xticklabels(df["stage_label"], rotation=35, ha="right",
                       fontsize=TICK_FONTSIZE, color="#000000")

    # ---------- Y-axis (left, area) ----------
    ax.set_ylabel("Potential area (Mha)", fontsize=AXIS_LABEL_FONTSIZE, color="#000000")
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors="#000000")
    step = max(50, int(Y_MAX / 6 / 50) * 50)
    ax.set_yticks(np.arange(0, Y_MAX + 1, step))
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.tick_params(colors="#000000")
    ax.set_axisbelow(True)

    # ---------- Y-axis (right, carbon) ----------
    ax2.set_ylabel("Living biomass carbon potential (Pg C)",
                   fontsize=AXIS_LABEL_FONTSIZE, color="#000000")
    ax2.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors="#000000")
    ax2.set_yticks(np.arange(0, CARBON_YMAX + 1, CARBON_TICK_STEP))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#000000")
    ax.spines["right"].set_visible(False)

    # ---------- Legend ----------
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_BASELINE, edgecolor=COLOR_BASELINE,
                       label="FAO criterion"),
        mpatches.Patch(facecolor=COLOR_PIVOT, edgecolor=COLOR_PIVOT_EDGE,
                       label="Eco safeguard"),
        mpatches.Patch(facecolor="none", edgecolor="#000000", label="Area"),
        mpatches.Patch(facecolor=COLOR_SCENARIO, edgecolor=COLOR_SCENARIO_EDGE,
                       label="Socioeconomic scenario"),
        mpatches.Patch(facecolor=COLOR_COMBINED, edgecolor=COLOR_COMBINED_EDGE,
                       label="All constraints"),
        mpatches.Patch(facecolor="none", edgecolor="#000000",
                       hatch=CARBON_HATCH, label="Carbon"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=LEGEND_FONTSIZE, frameon=False, ncol=2,
              columnspacing=1.6, handlelength=1.4, labelspacing=0.6,
              bbox_to_anchor=(0.98, 0.93))

    plt.tight_layout()
    out_png = os.path.join(config["output_dir"], config["output_filename"])
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}")


#  MAIN FUNCTION
def main():

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(CONFIG["csv_path"])

    print("\nLoaded data:")
    for _, row in df.iterrows():
        print(f"  {row['stage_label']:35s}  {row['area_mha']:7.2f} Mha")
    print()

    plot_cascade(df, CONFIG)


if __name__ == "__main__":
    main()