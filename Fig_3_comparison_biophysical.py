"""
This script generates Fig. 4 in Ran et al., showing comparisons of environmental conditions
(climate and soil) across different studies:
- Column 1: Temperature-Precipitation scatter plots
- Column 2: Aridity index histograms
- Column 3: Soil pH histograms

Requirements:
    - Python >= 3.8
    - rasterio >= 1.3
    - numpy >= 1.20
    - pandas >= 1.3
    - matplotlib >= 3.5

Author: QINWEI
Email: qwran@pku.edu.cn
Date: Jan. 2026
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path
import os

#  SETTINGS
CONFIG = {
    # Input data directory
    "data_dir": "data/",
    
    # Study mask files (add and minus layers for each study)
    "study_masks": [
        {
            "add": "fao_add_bastin.tif",
            "mns": "fao_mns_bastin.tif",
            "title": "Bastin",
            "labels": ["Both", "Bastin et al.", "This study"]
        },
        {
            "add": "fao_add_tolgyesi.tif",
            "mns": "fao_mns_tolgyesi.tif",
            "title": "Tölgyesi",
            "labels": ["Both", "Tölgyesi et al.", "This study"]
        },
        {
            "add": "fao_add_mo.tif",
            "mns": "fao_mns_mo.tif",
            "title": "Mo",
            "labels": ["Both", "Mo et al.", "This study"]
        },
        {
            "add": "fao_add_walker.tif",
            "mns": "fao_mns_walker.tif",
            "title": "Walker",
            "labels": ["Both", "Walker et al.", "This study"]
        },
        {
            "add": "fao_add_wri.tif",
            "mns": "fao_mns_wri.tif",
            "title": "FLRO",
            "labels": ["Both", "FLRO-25", "This study"]
        }
    ],
    
    # Environmental data files
    "env_data": {
        "temperature": "bio_6.tif",      # Min temp of coldest month
        "precipitation": "bio_12.tif",     # Annual precipitation
        "aridity_index": "ai.tif",     # Aridity index
        "soil_ph": "ph.tif"               # Soil pH
    },
    
    # Output settings
    "output_dir": "output/",
    "output_filename": "comparison_biophysical.png",
    "output_dpi": 300,
    
    # Figure settings
    "figure_size": (15, 15),
    "max_sample_points": 10000,  # Maximum points for scatter plot sampling
    
    # Y-axis limits for precipitation (per row)
    "precip_ylim": [4999, 5200, 5600, 5200, 5000],
}

#  Color scheme
COLORS = {
    "SC1": "#F59B7B",  # Both studies agree (orange)
    "SC2": "#A8D3A0",  # Other study only (green)
    "SC3": "#33ABC1",  # This study only (blue)
}

plt.rcParams['font.family'] = 'Arial'


#  FUNCTIONS
def get_file_path(filename):
    """Get full file path from config data directory."""
    return os.path.join(CONFIG["data_dir"], filename)


def load_raster(filepath):
    """Load a raster file and return the data array."""
    with rasterio.open(filepath) as src:
        return src.read(1)


def extract_sampled_xy(mask, label, data_tem, data_pre, max_points=10000):
   
    x = data_tem[mask]
    y = data_pre[mask]
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x = x[valid]
    y = y[valid]
    
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x = x[idx]
        y = y[idx]
    
    return pd.DataFrame({'Tem': x, 'Pre': y, 'SC': label})


def filter_valid_values(data, mask, min_val=0, max_val=1e9):
    """Filter valid values from data within mask."""
    values = data[mask]
    valid = (~np.isnan(values)) & (values > min_val) & (values < max_val)
    return values[valid]


def custom_formatter(x, pos):
    """Format axis tick labels."""
    return "0" if abs(x) < 1e-6 else f"{x:.1f}"


def create_legend_elements(labels):
    """Create legend elements for the three scenarios."""
    return [
        Line2D([0], [0], marker='o', color='w', label=labels[0],
               markerfacecolor=COLORS['SC1'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label=labels[1],
               markerfacecolor=COLORS['SC2'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label=labels[2],
               markerfacecolor=COLORS['SC3'], markersize=15)
    ]


def plot_climate_scatter(ax, df_all, ylim, labels, is_first_row=False):
    """
    Plot temperature-precipitation scatter plot.      
    """
    for sc in ['SC1', 'SC3', 'SC2']:
        subset = df_all[df_all['SC'] == sc]
        if len(subset) > 0:
            ax.hexbin(
                subset['Tem'], subset['Pre'],
                gridsize=50,
                cmap=ListedColormap([COLORS[sc]]),
                mincnt=1,
                alpha=0.8,
                linewidths=0
            )
    
    ax.set_ylim(0, ylim)
    
    if is_first_row:
        ax.set_title("Min Temp. of Coldest Month (°C)", fontsize=16)
    
    ax.set_ylabel("Annual Precip. (mm)", fontsize=16)
    ax.tick_params(labelsize=15)
    
    legend_elements = create_legend_elements(labels)
    ax.legend(handles=legend_elements, loc='upper left', fontsize=15, 
              frameon=False, handletextpad=0.1, labelspacing=0.3)


def plot_histogram(ax, values_list, labels, x_range, title=None, 
                   show_ylabel=True, legend_loc='upper right'):
    """
    Plot histogram for environmental variable. 
    """
    hist_colors = [COLORS['SC1'], COLORS['SC2'], COLORS['SC3']]
    
    # Check if any data exists
    if any(len(v) > 0 for v in values_list):
        ax.hist(values_list,
                bins=100,
                range=x_range,
                density=True,
                histtype='stepfilled',
                color=hist_colors,
                alpha=0.8)
        
        # Add mean lines
        for values, color in zip(values_list, hist_colors):
            if len(values) > 0:
                mean_val = np.mean(values)
                ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5)
    
    if title:
        ax.set_title(title, fontsize=16)
    
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=16)
    
    ax.tick_params(labelsize=15)
    
    legend_elements = create_legend_elements(labels)
    ax.legend(handles=legend_elements, loc=legend_loc, fontsize=15,
              frameon=False, handletextpad=0.1, labelspacing=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))


def add_mean_annotations(ax, values_list, positions, row_idx, col_type):
    """
    Add mean value annotations to histogram.
    """
    hist_colors = [COLORS['SC1'], COLORS['SC2'], COLORS['SC3']]
    
    # Default positions based on column type and row
    if col_type == 'ai':
        default_positions = {
            0: [(0.1, 2.1), (0.01, 2.3), (0.18, 2.5)],
            1: [(0.1, 2.1), (0.01, 2.3), (0.18, 2.5)],
            2: [(0.1, 2.1), (0.01, 2.3), (0.18, 2.5)],
            3: [(0.2, 1.35), (0.25, 1.2), (0.3, 1.0)],
            4: [(0.1, 2.4), (-0.3, 2.4), (0.15, 2.0)],
        }
        fmt = '{:.2f}'
    else:  # ph
        default_positions = {
            0: [(-1, 0.7), (0.1, 0.7), (0.17, 0.65)],
            1: [(0.1, 0.9), (0.1, 1.0), (-1, 1.0)],
            2: [(-1, 0.7), (0.1, 0.7), (0.17, 0.65)],
            3: [(-1.1, 0.75), (0.1, 0.75), (0.2, 0.65)],
            4: [(-1.3, 0.68), (0.1, 0.55), (-1.3, 0.58)],
        }
        fmt = '{:.1f}'
    
    positions = default_positions.get(row_idx, positions)
    
    for i, (values, color, pos) in enumerate(zip(values_list, hist_colors, positions)):
        if len(values) > 0:
            mean_val = np.mean(values)
            if mean_val > 0:
                ax.text(mean_val + pos[0], pos[1], fmt.format(mean_val),
                        color=color, fontsize=13)


# MAIN FUNCTION
def main():
    """Main function to generate the figure."""
    
    # Validate input paths
    required_files = []
    for mask_info in CONFIG["study_masks"]:
        required_files.append(get_file_path(mask_info["add"]))
        required_files.append(get_file_path(mask_info["mns"]))
    for env_file in CONFIG["env_data"].values():
        required_files.append(get_file_path(env_file))
    
    for filepath in required_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Input file not found: {filepath}\n"
                f"Please update the CONFIG paths at the top of this script."
            )
    
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load environmental data
    print("Loading environmental data...")
    data_tem = load_raster(get_file_path(CONFIG["env_data"]["temperature"]))
    data_pre = load_raster(get_file_path(CONFIG["env_data"]["precipitation"]))
    data_ai = load_raster(get_file_path(CONFIG["env_data"]["aridity_index"]))
    data_ph = load_raster(get_file_path(CONFIG["env_data"]["soil_ph"]))
    
    # Mask invalid values
    invalid_mask = ((data_tem > 1e7) | (data_tem < -1e7) | 
                    (data_pre > 1e7) | (data_pre < -1e7))
    data_tem[invalid_mask] = np.nan
    data_pre[invalid_mask] = np.nan
    
    # Create figure
    print("Creating figure...")
    fig, axs = plt.subplots(5, 3, figsize=CONFIG["figure_size"])
    
    for i, mask_info in enumerate(CONFIG["study_masks"]):
        print(f"Processing {mask_info['title']}...")
        
        # Load mask data
        data_add = load_raster(get_file_path(mask_info["add"]))
        data_mns = load_raster(get_file_path(mask_info["mns"]))
        
        # Create masks for each scenario
        mask_sc1 = (data_add == 2)   # Both studies
        mask_sc2 = (data_mns == -1)  # Other study only
        mask_sc3 = (data_mns == 1)   # This study only
        
        # Sample climate data
        df_all = pd.concat([
            extract_sampled_xy(mask_sc1, 'SC1', data_tem, data_pre, 
                             CONFIG["max_sample_points"]),
            extract_sampled_xy(mask_sc2, 'SC2', data_tem, data_pre,
                             CONFIG["max_sample_points"]),
            extract_sampled_xy(mask_sc3, 'SC3', data_tem, data_pre,
                             CONFIG["max_sample_points"])
        ], ignore_index=True)
        
        # Extract environmental values
        vals_ai = [
            filter_valid_values(data_ai, mask_sc1),
            filter_valid_values(data_ai, mask_sc2),
            filter_valid_values(data_ai, mask_sc3, min_val=0.1)
        ]
        
        vals_ph = [
            filter_valid_values(data_ph, mask_sc1),
            filter_valid_values(data_ph, mask_sc2),
            filter_valid_values(data_ph, mask_sc3)
        ]
        
        # Column 1: Temperature-Precipitation scatter
        plot_climate_scatter(
            axs[i, 0], df_all, 
            CONFIG["precip_ylim"][i],
            mask_info["labels"],
            is_first_row=(i == 0)
        )
        
        # Column 2: Aridity index histogram
        plot_histogram(
            axs[i, 1], vals_ai, mask_info["labels"],
            x_range=(0, 3),
            title="Aridity index" if i == 0 else None,
            legend_loc='upper right'
        )
        add_mean_annotations(axs[i, 1], vals_ai, None, i, 'ai')
        
        # Column 3: Soil pH histogram
        plot_histogram(
            axs[i, 2], vals_ph, mask_info["labels"],
            x_range=(0, 10),
            title="Soil pH" if i == 0 else None,
            show_ylabel=False,
            legend_loc='upper left'
        )
        add_mean_annotations(axs[i, 2], vals_ph, None, i, 'ph')
    
    # Set y-axis ticks
    for idx, ax in enumerate(axs.flat):
        row = idx // 3
        col = idx % 3
        if not (row == 1 and col == 2):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    
    axs[1, 2].set_yticks([0, 0.5, 1])
    
    # Add panel labels (A-O)
    letters = [chr(i) for i in range(65, 65 + 15)]
    for idx, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.05, letters[idx], transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='left')
    
    # Adjust layout
    plt.subplots_adjust(hspace=0.17, wspace=0.3)
    for i in range(5):
        pos = axs[i, 2].get_position()
        axs[i, 2].set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.17)
    
    for i in range(5):
        pos = axs[i, 2].get_position()
        axs[i, 2].set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])
    
    # Save figure
    output_path = output_dir / CONFIG["output_filename"]
    print(f"Saving to: {output_path}")
    plt.savefig(output_path, dpi=CONFIG["output_dpi"], bbox_inches='tight')
    
    print("Figure generation complete!")
    plt.show()


if __name__ == "__main__":

    main()

