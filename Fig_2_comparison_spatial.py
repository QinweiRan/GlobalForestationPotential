"""
This script generates Fig. 3 in Ran et al., showing comparisons of forestation potential
estimates from different studies:
- Panels A-E: Spatial comparison maps with latitudinal density distributions
- Panel F: Cumulative bar chart by climate zone

Requirements:
    - Python >= 3.8
    - rasterio >= 1.3
    - numpy >= 1.20
    - matplotlib >= 3.5
    - pandas >= 1.3
    - scipy >= 1.7
    - cartopy >= 0.20

Author: QINWEI
Email: qwran@pku.edu.cn
Date: Jan. 2026
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from pathlib import Path
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#  SETTINGS
CONFIG = {
    # Input data directory
    "data_dir": "data/",
    
    "study_files": {
        "bastin": ("fao_add_bastin.tif", "fao_mns_bastin.tif"),
        "tölgyesi": ("fao_add_tolgyesi.tif", "fao_mns_tolgyesi.tif"),
        "mo": ("fao_add_mo.tif", "fao_mns_mo.tif"),
        "walker": ("fao_add_walker.tif", "fao_mns_walker.tif"),
        "flro": ("fao_add_flro.tif", "fao_mns_flro.tif"),
    },    
   
    # CSV data files
    "bar_data_csv": "Fig3_compare_area.csv",
    "climate_zone_csv": "Fig3_compare_area_climatezone.csv",
    
    # Output settings
    "output_dir": "output/",
    "output_filename": "comparison_spatial",
    "output_formats": ["png"], 
    "output_dpi": 300,
    
    # Figure settings
    "figure_size": (16, 12),
    "figure_dpi": 150,
}

#  Study labels and legend text
STUDY_CONFIG = {
    "labels": ["A", "B", "C", "D", "E"],
    "legend_text": [
        ["Both", "Bastin et al.", "This study"],
        ["Both", "Tölgyesi et al.", "This study"],
        ["Both", "Mo et al.", "This study"],
        ["Both", "Walker et al.", "This study"],
        ["Both", "FLRO-25", "This study"],
    ],
    "y_labels_climate": [
        "This study", "Bastin et al.", "Tölgyesi et al.", 
        "Mo et al.", "Walker et al.", "FLRO-25"
    ],
}

#  Color scheme
COLORS = {
    "add": "#F59B7B",        # Orange - areas in both studies
    "mns_neg": "#A8D3A0",    # Green - areas only in other study
    "mns_pos": "#33ABC1",    # Blue - areas only in this study
    "world": "lightgray",    # Background
    "climate_zones": [
        "#8DD3C7",  # Light cyan
        "#E7B2AC",  # Light red
        "#BEBADA",  # Light purple
        "#FCCDE5",  # Light pink
        "#FFFFB3",  # Light yellow
    ],
}

plt.rcParams['font.family'] = 'Arial'

#  FUNCTIONS
def get_file_path(filename):
    """Get full file path from config data directory."""
    return os.path.join(CONFIG["data_dir"], filename)


def plot_comparison_map(ax_map, ax_bar, file_add, file_mns, bar_data, 
                        bar_index, bar_col_start, legend_labels, label_text):
   
    with rasterio.open(file_add) as src_add:
        data_add = src_add.read(1)
        bounds_add = src_add.bounds

    with rasterio.open(file_mns) as src_mns:
        data_mns = src_mns.read(1)
        bounds_mns = src_mns.bounds
        height, width = data_mns.shape

    # Draw base map
    ax_map.add_feature(cfeature.LAND.with_scale("50m"), 
                       facecolor='lightgray', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN.with_scale("50m"), 
                       facecolor='white', edgecolor='none')
    ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.2)

    extent_add = [bounds_add.left, bounds_add.right, 
                  bounds_add.bottom, bounds_add.top]
    extent_mns = [bounds_mns.left, bounds_mns.right, 
                  bounds_mns.bottom, bounds_mns.top]

    # Plot data layers
    # Add layer (orange - both studies agree)
    masked_add = np.ma.masked_where(data_add != 2, data_add)
    ax_map.imshow(masked_add, extent=extent_add, transform=ccrs.PlateCarree(),
                  origin='upper', cmap=ListedColormap([COLORS["add"]]), 
                  interpolation='nearest')

    # Minus layer (green and blue)
    data_mns_remap = np.where(data_mns == -1, 0, np.where(data_mns == 1, 2, 1))
    masked_mns_neg = np.ma.masked_where(data_mns_remap != 0, data_mns_remap)
    masked_mns_pos = np.ma.masked_where(data_mns_remap != 2, data_mns_remap)

    ax_map.imshow(masked_mns_neg, extent=extent_mns, transform=ccrs.PlateCarree(),
                  origin='upper', cmap=ListedColormap([COLORS["mns_neg"]]),
                  interpolation='nearest', alpha=0.7)
    ax_map.imshow(masked_mns_pos, extent=extent_mns, transform=ccrs.PlateCarree(),
                  origin='upper', cmap=ListedColormap([COLORS["mns_pos"]]),
                  interpolation='nearest', alpha=0.7)

    ax_map.set_global()

    # Add gridlines
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, 
                          color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    # Add panel label
    ax_map.text(-0.05, 0.95, label_text, fontsize=16, fontweight='bold',
                transform=ax_map.transAxes)

    # Inset bar chart
    _draw_inset_bar(ax_map, bar_data, bar_index, bar_col_start)

    # Density plot
    _draw_density_plot(ax_bar, data_add, data_mns, bounds_mns, height)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=legend_labels[0],
               markerfacecolor=COLORS["add"], markersize=10),
        Line2D([0], [0], marker='o', color='w', label=legend_labels[1],
               markerfacecolor=COLORS["mns_neg"], markersize=10),
        Line2D([0], [0], marker='o', color='w', label=legend_labels[2],
               markerfacecolor=COLORS["mns_pos"], markersize=10)
    ]

    ax_map.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.6, 0.05),
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=0.15,
        handletextpad=0.05
    )


def _draw_inset_bar(ax_map, bar_data, bar_index, bar_col_start):
    """Draw inset bar chart on the map."""
    inset_ax = ax_map.inset_axes([0.15, 0.16, 0.12, 0.3])

    values = bar_data.iloc[bar_index, bar_col_start:bar_col_start + 4].values
    bar_width = 0.45

    # First group of bars
    inset_ax.bar(x=0.2, height=values[0], color=COLORS["add"], 
                 width=bar_width, bottom=0, edgecolor='white', linewidth=1)
    inset_ax.bar(x=0.2, height=values[1], color=COLORS["mns_neg"],
                 width=bar_width, bottom=values[0], edgecolor='white', linewidth=0.5)

    # Second group of bars
    inset_ax.bar(x=0.9, height=values[2], color=COLORS["add"],
                 width=bar_width, bottom=0, edgecolor='white', linewidth=1)
    inset_ax.bar(x=0.9, height=values[3], color=COLORS["mns_pos"],
                 width=bar_width, bottom=values[2], edgecolor='white', linewidth=0.5)

    inset_ax.set_xticks([0.2, 0.9])
    inset_ax.set_xticklabels([])
    inset_ax.set_yticks([])

    max_cumulative = max(values[0] + values[1], values[2] + values[3])
    inset_ax.set_ylim(0, max_cumulative * 1.2)
    inset_ax.set_xlim(0, 1.3)

    for spine in inset_ax.spines.values():
        spine.set_visible(False)
    inset_ax.axhline(y=0, color='black', linewidth=1)
    inset_ax.axvline(x=0, color='black', linewidth=1)
    inset_ax.patch.set_alpha(0)

    max_tick = int(np.ceil(max_cumulative / 500) * 500)
    inset_ax.set_yticks([0, max_tick])
    inset_ax.set_yticklabels([f'{int(t)}' for t in [0, max_tick]], 
                              fontsize=10, rotation=90, va='center')
    inset_ax.set_ylabel('Area (Mha)', fontsize=10)


def _draw_density_plot(ax_bar, data_add, data_mns, bounds_mns, height):
    """Draw latitudinal density distribution plot."""
    footstep = 1
    lat_max = bounds_mns.top
    lat_footsteps = np.arange(np.floor(-60), np.ceil(lat_max) + footstep, footstep)
    
    neg_sum = np.zeros(len(lat_footsteps) - 1)
    pos_sum = np.zeros(len(lat_footsteps) - 1)
    both_sum = np.zeros(len(lat_footsteps) - 1)

    for i in range(len(lat_footsteps) - 1):
        row_upper = int(np.round((lat_footsteps[i + 1] - bounds_mns.top) * height / 
                                  (bounds_mns.bottom - bounds_mns.top)))
        row_lower = int(np.round((lat_footsteps[i] - bounds_mns.top) * height / 
                                  (bounds_mns.bottom - bounds_mns.top)))
        row_upper = max(0, min(row_upper, height - 1))
        row_lower = max(0, min(row_lower, height - 1))

        band_mns = data_mns[row_upper:row_lower + 1, :]
        band_add = data_add[row_upper:row_lower + 1, :]

        if band_mns.size > 0:
            total_size = band_mns.size
            neg_sum[i] = np.sum(band_mns == -1) / total_size
            pos_sum[i] = np.sum(band_mns == 1) / total_size
            both_sum[i] = np.sum(band_add == 2) / total_size

    lat_centers = (lat_footsteps[:-1] + lat_footsteps[1:]) / 2
    sigma = 1.0
    neg_smoothed = gaussian_filter1d(neg_sum, sigma=sigma)
    pos_smoothed = gaussian_filter1d(pos_sum, sigma=sigma)
    both_smoothed = gaussian_filter1d(both_sum, sigma=sigma)

    # Plot density curves
    for data, color in [(neg_smoothed, COLORS["mns_neg"]),
                        (pos_smoothed, COLORS["mns_pos"]),
                        (both_smoothed, COLORS["add"])]:
        ax_bar.plot(data, lat_centers, color=color, linewidth=1.5)
        ax_bar.fill_betweenx(lat_centers, 0, data, color=color, alpha=0.8)

    def custom_formatter(x, pos):
        return "0" if abs(x) < 1e-6 else f"{x:.1f}"

    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    ax_bar.set_xlabel('Density', fontsize=10)
    ax_bar.set_ylim(-80, 80)
    ax_bar.set_yticks([-80, -60, -30, 0, 30, 60, 80])
    ax_bar.set_yticklabels(['80°', '60°', '30°', '0°', '30°', '60°', '80°'], fontsize=9)
    ax_bar.set_xlim(0, max(np.max(neg_smoothed), np.max(pos_smoothed), 
                          np.max(both_smoothed)) * 1.2)
    ax_bar.tick_params(labelsize=9, width=0.5)

    for s in ax_bar.spines.values():
        s.set_linewidth(0.5)


def draw_climate_zone_chart(fig, ax_climate, climate_file, reference_ax):
   
    df_climate = pd.read_csv(climate_file, encoding='utf-8')

    climate_zones = df_climate.iloc[:, 0].values
    studies = df_climate.columns[1:]
    data_climate = df_climate.iloc[:, 1:].values

    # Extend colors if needed
    colors_climate = COLORS["climate_zones"]
    if len(climate_zones) > len(colors_climate):
        colors_climate = plt.cm.Set3(np.linspace(0, 1, len(climate_zones)))

    # Draw cumulative horizontal bar chart
    left = np.zeros(len(studies))
    for i, zone in enumerate(climate_zones):
        values = data_climate[i, :]
        ax_climate.barh(studies, values, left=left, label=zone,
                        color=colors_climate[i], edgecolor='white', 
                        linewidth=1, height=0.6)
        left += values

    # Set y-axis labels
    y_labels = STUDY_CONFIG["y_labels_climate"]
    ax_climate.set_yticks(np.arange(len(y_labels)))
    ax_climate.set_yticklabels(y_labels, fontsize=10)

    ax_climate.set_xlabel('Forestation Area (Mha)', fontsize=10)

    # Legend
    leg = ax_climate.legend(
        title='Climate Zone',
        bbox_to_anchor=(1.3, 0.05),
        loc='lower right',
        fontsize=10,
        title_fontsize=10,
        frameon=True
    )
    leg.get_frame().set_edgecolor('#262626')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor('white')

    ax_climate.invert_yaxis()
    ax_climate.tick_params(axis='both', labelsize=10)

    # Add panel label
    ax_climate.text(-0.25, 0.97, 'F', fontsize=16, fontweight='bold',
                    transform=ax_climate.transAxes)

    # Adjust position to align with reference axes
    pos_ref = reference_ax.get_position()
    pos_climate = ax_climate.get_position()
    new_height = pos_ref.height * 0.85
    new_width = pos_climate.width * 0.65
    new_y = pos_ref.y0 + (pos_ref.height - new_height) / 2
    new_x = pos_climate.x0 + 0.04
    ax_climate.set_position([new_x, new_y, new_width, new_height])


# MAIN FUNCTION
def main():
    """Main function to generate the figure."""
    
    # Validate input paths
    study_files = CONFIG["study_files"]
    required_files = []
    for study, (add_file, mns_file) in study_files.items():
        required_files.append(get_file_path(add_file))
        required_files.append(get_file_path(mns_file))
    required_files.append(get_file_path(CONFIG["bar_data_csv"]))
    required_files.append(get_file_path(CONFIG["climate_zone_csv"]))
    
    for filepath in required_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Input file not found: {filepath}\n"
                f"Please update the CONFIG paths at the top of this script."
            )
    
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read bar chart data
    bar_data = pd.read_csv(get_file_path(CONFIG["bar_data_csv"]))
    
    # Create figure
    print("Creating figure...")
    projection = ccrs.EckertIV()
    fig = plt.figure(figsize=CONFIG["figure_size"], dpi=CONFIG["figure_dpi"])
    outer_gs = fig.add_gridspec(3, 2, hspace=-0.39, wspace=0.1)
    
    # Layout: 5 maps in 3x2 grid (position 6 is for climate chart)
    layout = [
        (0, 0),  # A - Bastin
        (0, 1),  # B - Tölgyesi
        (1, 0),  # C - MO
        (1, 1),  # D - Walker
        (2, 0),  # E - FLRO
    ]
    
    study_names = list(study_files.keys())
    
    for idx, study_name in enumerate(study_names):
        print(f"Processing {study_name}...")
        add_file, mns_file = study_files[study_name]
        row, col = layout[idx]
        
        inner_gs = outer_gs[row, col].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.1)
        ax_main = fig.add_subplot(inner_gs[0, 0], projection=projection)
        ax_lat = fig.add_subplot(inner_gs[0, 1])
        
        plot_comparison_map(
            ax_main, ax_lat,
            get_file_path(add_file),
            get_file_path(mns_file),
            bar_data,
            bar_index=0,
            bar_col_start=1 + idx * 4,
            legend_labels=STUDY_CONFIG["legend_text"][idx],
            label_text=STUDY_CONFIG["labels"][idx]
        )
        
        # Adjust density plot height
        pos_main = ax_main.get_position()
        pos_lat = ax_lat.get_position()
        new_height = pos_main.height * 0.9
        new_y = pos_main.y0 + (pos_main.height - new_height) / 2
        ax_lat.set_position([pos_lat.x0, new_y, pos_lat.width, new_height])
    
    # Add climate zone chart (Panel F)
    print("Adding climate zone chart...")
    ax_climate = fig.add_subplot(outer_gs[2, 1])
    draw_climate_zone_chart(
        fig, ax_climate,
        get_file_path(CONFIG["climate_zone_csv"]),
        reference_ax=fig.axes[8]  # Reference to Panel E
    )
    
    # Save figure
    for fmt in CONFIG["output_formats"]:
        output_path = output_dir / f"{CONFIG['output_filename']}.{fmt}"
        print(f"Saving to: {output_path}")
        plt.savefig(output_path, dpi=CONFIG["output_dpi"], 
                    bbox_inches='tight', pad_inches=0.08, facecolor="white")
    
    print("Figure generation complete!")
    plt.show()


if __name__ == "__main__":

    main()

