"""
This script generates Fig. 1 in Ran et al., showing:
- Panel A: Bivariate map of potential tree cover (PTC) and potential canopy height (PCH)
- Panel B: Potential canopy height with latitudinal density distribution
- Panel C: Potential tree cover with latitudinal density distribution

Requirements:
    - Python >= 3.8
    - matplotlib >= 3.5
    - numpy >= 1.20
    - cartopy >= 0.20
    - scipy >= 1.7
    - GDAL (osgeo) >= 3.0

Author: QINWEI
Email: qwran@pku.edu.cn
Date: Sep. 2026
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from osgeo import gdal
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from pathlib import Path

#  SETTINGS
CONFIG = {
    "height_path": "data/PCH_bootstrap_mean.tif",          
    "cover_path": "data/PTC_bootstrap_mean.tif",             
    "height_masked_path": "data/PCH_bootstrap_mean_masked.tif", 
    "cover_masked_path": "data/PTC_bootstrap_mean_masked.tif",    
    
    "output_dir": "output/",
    "output_filename": "Fig_1_forestation_potential_biVar.png",
    "output_dpi": 300,
    
    "piece_num": 5,           
    "lat_bin_size": 0.5,      
    
    "height_min": 5,         
    "height_max": 100,        
    "cover_min": 10,        
    "cover_max": 100,         
}

plt.rcParams["font.sans-serif"] = "Arial"

#  FUNCTIONS
def get_geotiff_data(path):
    
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    gt = ds.GetGeoTransform()
    array = ds.ReadAsArray().astype(np.float32)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    xres, yres = gt[1], gt[5]
    xcor, ycor = gt[0], gt[3]
    lat = np.arange(ycor, ycor + yres * ysize, yres)
    lon = np.arange(xcor, xcor + xres * xsize, xres)
    return lon, lat, array


def preprocess(height, cover, nodata_values=(0, -9999)):
   
    for nodata in nodata_values:
        height = np.where(height == nodata, np.nan, height)
        cover = np.where(cover == nodata, np.nan, cover)

    height = np.where((height < 0) | (height > 100), np.nan, height)
    cover = np.where((cover < 0) | (cover > 100), np.nan, cover)
    return height, cover


#  BIVARIATE COLORMAP 
class BiVarHeatmap:
    # Purple-cyan color scheme (corner colors for bivariate interpolation)
    # Order: [low-low, high-low, low-high, high-high]
    COLOR_SCHEME = np.array([
        [233, 231, 242],  # Low PTC, Low PCH (light purple-gray)
        [78, 174, 209],   # High PTC, Low PCH (cyan)
        [223, 78, 167],   # Low PTC, High PCH (magenta)
        [37, 19, 139]     # High PTC, High PCH (deep purple)
    ]) / 255
    
    def __init__(self, piece_num=5):
        if piece_num < 2:
            raise ValueError("piece_num must be >= 2")
        self.piece_num = piece_num
        self.rgb = self._make_legend(self.COLOR_SCHEME, piece_num)

    def _make_legend(self, color, piece_num):
        """Generate interpolated color grid."""
        def interp(channel):
            points = np.array([
                [piece_num - 1, 0],
                [piece_num - 1, piece_num - 1],
                [0, 0],
                [0, piece_num - 1]
            ])
            values = np.array([
                color[1, channel],
                color[3, channel],
                color[0, channel],
                color[2, channel]
            ])
            grid_x, grid_y = np.mgrid[0:piece_num, 0:piece_num]
            r = griddata(points, values, (grid_x, grid_y), method="cubic")
            mask = np.isnan(r)
            if np.any(mask):
                r[mask] = griddata(points, values, (grid_x, grid_y), method="linear")[mask]
            return r

        r = interp(0)
        g = interp(1)
        b = interp(2)
        return np.dstack((r, g, b))

    def map(self, A, B, bins_A=None, bins_B=None):
       
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape")

        piece_num = self.piece_num
        rgb = self.rgb

        nan_mask = np.isnan(A) | np.isnan(B)
        idx_A = np.full(A.shape, -1, dtype=int)
        idx_B = np.full(B.shape, -1, dtype=int)

        if np.any(~nan_mask):
            A_valid = A[~nan_mask]
            B_valid = B[~nan_mask]

            if bins_A is None:
                bins_A = np.linspace(np.nanmin(A_valid), np.nanmax(A_valid), piece_num + 1)
            if bins_B is None:
                bins_B = np.linspace(np.nanmin(B_valid), np.nanmax(B_valid), piece_num + 1)

            A_valid = np.clip(A_valid, bins_A[0], bins_A[-1])
            B_valid = np.clip(B_valid, bins_B[0], bins_B[-1])

            idx_A[~nan_mask] = np.digitize(A_valid, bins_A, right=True)
            idx_B[~nan_mask] = np.digitize(B_valid, bins_B, right=True)

        idx_A = np.clip(idx_A, 0, piece_num - 1)
        idx_B = np.clip(idx_B, 0, piece_num - 1)

        colorAB = rgb[idx_B, idx_A]
        alpha = np.ones(A.shape, dtype=float)
        alpha[nan_mask] = 0.0
        colorAB = np.concatenate([colorAB, alpha[..., None]], axis=-1)
        return colorAB, rgb


#  PLOTTING FUNCTIONS
def draw_bivar_map(ax, lon, lat, rgba_img, number):
    """Draw a bivariate map on the given axes."""
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.2)

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(
        rgba_img,
        extent=extent,
        transform=ccrs.PlateCarree(),
        origin="upper",
        interpolation="nearest"
    )

    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.xlines = False
    gl.ylines = True
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 91, 30))
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = True
    gl.ylabel_style = {"size": 8}
    gl.ylabel_formatter = mpl.ticker.FuncFormatter(lambda x, pos: f"{int(x)}°" if pos == 0 else "")
    gl.right_labels = False

    ax.text(-0.03, 0.95, number, fontsize=13, fontweight="bold", transform=ax.transAxes)


def draw_lon_plot(ax, data, lat, lat_bin_size=0.5, xlabel="Canopy cover (%)", color='red'):
    
    if data.shape[0] != len(lat):
        print("Warning: data latitude length does not match lat array!")
        return

    lat_bins = np.floor(lat / lat_bin_size) * lat_bin_size
    unique_bins = np.unique(lat_bins)
    
    mean_values = []
    std_values = []
    bin_centers = []
    
    for b in unique_bins:
        bin_indices = np.where(lat_bins == b)[0]
        bin_data = data[bin_indices, :].flatten()
        valid_data = bin_data[(bin_data >= 0) & (~np.isnan(bin_data))]
        
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            mean_values.append(mean_val)
            std_values.append(std_val)
            bin_centers.append(b + lat_bin_size / 2)
    
    if len(mean_values) == 0:
        return
    
    mean_values = np.array(mean_values)
    std_values = np.array(std_values)
    bin_centers = np.array(bin_centers)
    
    upper_bound = mean_values + std_values
    lower_bound = mean_values - std_values
    
    ax.fill_betweenx(bin_centers, lower_bound, upper_bound, 
                     alpha=0.3, color='gray', label='±1 SD')
    ax.plot(mean_values, bin_centers, color=color, linewidth=2, 
            label='Mean', alpha=0.8)
    
    ax.set_xlim(0, max(upper_bound) * 1.05)
    ax.set_ylim(min(bin_centers), max(bin_centers))
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.tick_params(labelsize=8, width=0.5)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)


def draw_bivar_legend_inset(ax, rgb, piece_num,
                            x_range=(0, 100), y_range=(0, 50),
                            anchor=(0.11, 0.255), size=0.3,
                            tick_fs=7, label_fs=8):
    """Draw bivariate legend as an inset in the map axes."""
    axl = inset_axes(
        ax,
        width=f"{size * 100:.1f}%",
        height=f"{size * 100:.1f}%",
        loc="lower left",
        bbox_to_anchor=(anchor[0], anchor[1], 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    axl.imshow(rgb, origin="lower")
    axl.set_xticks([0, piece_num // 2, piece_num - 1])
    axl.set_yticks([0, piece_num // 2, piece_num - 1])

    axl.set_xticklabels(
        [str(x_range[0]), str((x_range[0] + x_range[1]) // 2), str(x_range[1])],
        fontsize=tick_fs
    )
    axl.set_yticklabels(
        [str(y_range[0]), str((y_range[0] + y_range[1]) // 2), str(y_range[1])],
        fontsize=tick_fs
    )
    axl.tick_params(labelsize=tick_fs, width=0.5, length=2, pad=1)
    axl.set_xlabel("PTC (%)", fontsize=label_fs, labelpad=1)
    axl.set_ylabel("PCH (m)", fontsize=label_fs, labelpad=1)

    for s in axl.spines.values():
        s.set_linewidth(0.5)

    axl.set_aspect("equal")
    return axl


def add_inset_hcbar_corner(fig, ax, im, ticks, ticklabels, label,
                           anchor=(0.38, 0.10), width=0.37, height=0.05,
                           tick_fs=8, label_fs=8, label_pad=3,
                           extend="max"):
    """Add a horizontal colorbar as an inset in the corner."""
    cax = inset_axes(
        ax,
        width=f"{width * 100:.1f}%",
        height=f"{height * 100:.1f}%",
        loc="lower left",
        bbox_to_anchor=(anchor[0], anchor[1], 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(size=1, labelsize=tick_fs, width=0.5, length=2, pad=1)
    cbar.outline.set_linewidth(0.2)
    cbar.set_label(label, fontsize=label_fs, labelpad=label_pad, loc="center")
    cbar.ax.xaxis.set_label_position("top")
    return cbar


def draw_height_map(fig, ax, lon, lat, array, number, show_colorbar=True,
                    cb_anchor=(0.38, 0.10), cb_tick_fs=9, cb_label_fs=9, cb_label_pad=3):
    """Draw potential canopy height map."""
    cmap = plt.cm.gnuplot2_r
    norm = Normalize(vmin=5, vmax=30)

    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    array = np.where(array == -9999, np.nan, array)

    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.2)

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    im = ax.imshow(array, extent=extent, transform=ccrs.PlateCarree(),
                   cmap=cmap, norm=norm, origin="upper")

    if show_colorbar:
        add_inset_hcbar_corner(
            fig, ax, im,
            ticks=[5, 10, 20, 30],
            ticklabels=["5", "10", "20", "30"],
            label="Potential canopy height (m)",
            anchor=cb_anchor, width=0.36, height=0.048,
            tick_fs=cb_tick_fs, label_fs=cb_label_fs, label_pad=cb_label_pad,
            extend="max"
        )

    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.xlines = False
    gl.ylines = True
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 91, 30))
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = True
    gl.ylabel_style = {"size": 8}
    gl.ylabel_formatter = mpl.ticker.FuncFormatter(lambda x, pos: f"{int(x)}°" if pos == 0 else "")
    gl.right_labels = False

    ax.text(-0.03, 0.95, number, fontsize=13, fontweight="bold", transform=ax.transAxes)
    return cmap, norm


def draw_cover_map(fig, ax, lon, lat, array, number, show_colorbar=True,
                   cb_anchor=(0.38, 0.10), cb_tick_fs=9, cb_label_fs=9, cb_label_pad=3):
    """Draw potential tree cover map."""
    cmap = plt.cm.YlGnBu
    norm = Normalize(vmin=10, vmax=60)

    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    array = np.where(array == -9999, np.nan, array)
    array_disp = np.clip(array, 10, 60)

    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.2)

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    im = ax.imshow(array_disp, extent=extent, transform=ccrs.PlateCarree(),
                   cmap=cmap, norm=norm, origin="upper")

    if show_colorbar:
        add_inset_hcbar_corner(
            fig, ax, im,
            ticks=[10, 30, 45, 60],
            ticklabels=["10", "30", "45", "60"],
            label="Potential tree cover (%)",
            anchor=cb_anchor, width=0.36, height=0.048,
            tick_fs=cb_tick_fs, label_fs=cb_label_fs, label_pad=cb_label_pad,
            extend="max"
        )

    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.xlines = False
    gl.ylines = True
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 91, 30))
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = True
    gl.ylabel_style = {"size": 8}
    gl.ylabel_formatter = mpl.ticker.FuncFormatter(lambda x, pos: f"{int(x)}°" if pos == 0 else "")
    gl.right_labels = False

    ax.text(-0.03, 0.95, number, fontsize=13, fontweight="bold", transform=ax.transAxes)
    return cmap, norm


def draw_area_plot(ax, data, lat, cmap, norm, lat_bin_size=0.5, min_val=5):
    """Draw latitudinal density distribution plot."""
    if data.shape[0] != len(lat):
        print("Warning: data latitude length does not match lat array!")
        return

    lat_bins = np.floor(lat / lat_bin_size) * lat_bin_size
    unique_bins = np.unique(lat_bins)

    valid_mask_global = (~np.isnan(data)) & (data >= min_val)
    total_valid_pixels = np.sum(valid_mask_global)
    print(f"Total valid pixels (>={min_val}): {total_valid_pixels}")

    densities, mean_values, bin_centers = [], [], []

    for b in unique_bins:
        bin_indices = np.where(lat_bins == b)[0]
        bin_data = data[bin_indices, :].ravel()

        valid_mask = (~np.isnan(bin_data)) & (bin_data >= min_val)
        valid_data = bin_data[valid_mask]

        density = valid_data.size / total_valid_pixels if total_valid_pixels > 0 else 0
        densities.append(density)

        mean_val = np.mean(valid_data) if valid_data.size > 0 else min_val
        mean_values.append(mean_val)

        bin_centers.append(b + lat_bin_size / 2)

    if len(densities) == 0:
        return

    densities = np.array(densities)
    mean_values = np.array(mean_values)
    bin_centers = np.array(bin_centers)

    for i in range(len(bin_centers) - 1):
        color = cmap(norm(mean_values[i]))
        ax.fill_betweenx(
            [bin_centers[i], bin_centers[i + 1]],
            0, [densities[i], densities[i + 1]],
            color=color, alpha=0.8, linewidth=0
        )

    ax.plot(densities, bin_centers, color="black", linewidth=0.8, alpha=0.7)

    ax.set_xlim(0, max(densities) * 1.05 if densities.size else 1)
    ax.set_ylim(min(bin_centers), max(bin_centers))
    ax.set_xlabel("Density", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.tick_params(labelsize=8, width=0.5, length=2)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)

    def custom_formatter(x, pos):
        return "0" if abs(x) < 1e-6 else f"{x:.2f}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))


def draw_histogram(ax, data, mask_threshold=5, x_ticks=[5, 15, 30], 
                   x_tick_labels=['5', '15', '30'], xlabel='', unit='m'):
    """Draw density histogram with median line."""
    data = np.array(data).ravel()
    data = data[~np.isnan(data)]
    data = data[data >= mask_threshold]
    
    if len(data) == 0:
        ax.text(0.5, 0.5, f'No data >={mask_threshold}', ha='center', va='center', fontsize=9)
        return
    
    hist_range = (mask_threshold, max(mask_threshold, np.max(data) * 1.05))
    bins = np.linspace(hist_range[0], hist_range[1], 51)
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for i in range(len(counts)):
        ax.bar(bin_centers[i], counts[i],
               width=(bin_edges[1] - bin_edges[0]) * 0.95,
               color='gray',
               align='center',
               edgecolor='none')
    
    median_val = np.median(data)
    ax.axvline(median_val, color='black', linestyle='--', linewidth=1.2, alpha=0.9)

    ylim = ax.get_ylim()
    ax.text(
        median_val, ylim[1] * 0.85,
        f'Med. {median_val:.0f}{unit}',
        color='black',
        fontsize=8.5,
        ha='left',
        va='top',
        rotation=0
    )
    
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f'{x:.2f}' if x != 0 else '0'))
    
    ax.set_xlim(mask_threshold, np.max(data) * 1.05)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel('Density', fontsize=9, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    ax.tick_params(labelsize=9, width=0.5)
    ax.set_facecolor('none')


#  MAIN FUNCTION
def main():
  
    required_files = [
        CONFIG["height_path"],
        CONFIG["cover_path"],
        CONFIG["height_masked_path"],
        CONFIG["cover_masked_path"]
    ]
    
    for filepath in required_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Input file not found: {filepath}\n"
                f"Please update the CONFIG paths at the top of this script."
            )
    
    # Create output directory if it doesn't exist
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize figure
    projection = ccrs.EckertIV()
    fig = plt.figure(figsize=(10.2, 10), dpi=CONFIG["output_dpi"], facecolor="none")

    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.0, 0.07, 0.07],
        height_ratios=[1, 1, 1],
        wspace=0.015,
        hspace=0.05
    )

    ax1 = fig.add_subplot(gs[0, 0], projection=projection)
    ax2 = fig.add_subplot(gs[1, 0], projection=projection)
    ax3 = fig.add_subplot(gs[2, 0], projection=projection)

    ax1_lon_H = fig.add_subplot(gs[0, 1])
    ax1_lon_C = fig.add_subplot(gs[0, 2])

    ax2_density = fig.add_subplot(gs[1, 1:3])
    ax3_density = fig.add_subplot(gs[2, 1:3])

    # Adjust subplot positions
    for axd in [ax1_lon_H, ax1_lon_C]:
        pos = axd.get_position()
        axd.set_position([
            pos.x0 - 0.05,
            pos.y0 + pos.height * 0.15,
            pos.width * 0.85,
            pos.height * 0.85
        ])

    for axd in [ax2_density, ax3_density]:
        pos = axd.get_position()
        axd.set_position([
            pos.x0 - 0.05,
            pos.y0 + pos.height * 0.15,
            pos.width * 0.85,
            pos.height * 0.85
        ])

    # Initialize bivariate colormap
    piece_num = CONFIG["piece_num"]
    heatmap = BiVarHeatmap(piece_num=piece_num)

    bins_cover = np.linspace(0, 100, piece_num + 1)
    bins_height = np.linspace(0, 50, piece_num + 1)

    # --------- Panel A: Bivariate Map ---------
    print("Loading bivariate map data...")
    lon1, lat1, h1 = get_geotiff_data(CONFIG["height_path"])
    _, _, c1 = get_geotiff_data(CONFIG["cover_path"])
    h1, c1 = preprocess(h1, c1)

    # Panel A
    h1 = np.where(h1 < 0, np.nan, h1)
    c1 = np.where(c1 < 0, np.nan, c1)

    arrayH = np.where((h1 > CONFIG["height_max"]), np.nan, h1)
    arrayC = np.where((c1 > CONFIG["cover_max"]), np.nan, c1)

    rgba1, rgb = heatmap.map(A=h1, B=c1, bins_A=bins_height, bins_B=bins_cover)
    draw_bivar_map(ax1, lon1, lat1, rgba1, number="A")
    
    draw_lon_plot(ax1_lon_H, arrayH, lat1, xlabel="PCH (m)", color='#C81CDE')
    draw_lon_plot(ax1_lon_C, arrayC, lat1, xlabel="PTC (%)", color='#0092D1')
    
    ax1_lon_C.set_ylabel("")
    ax1_lon_C.set_yticklabels([])

    draw_bivar_legend_inset(
        ax1, rgb.transpose(1, 0, 2), piece_num,
        x_range=(0, 100),
        y_range=(0, 50),
        anchor=(0.075, 0.255),
        size=0.23,
        tick_fs=8,
        label_fs=8
    )

    # --------- Panel B: Height Map ---------
    print("Loading height map data...")
    lon2, lat2, array2 = get_geotiff_data(CONFIG["height_masked_path"])
    array2 = np.where((array2 > CONFIG["height_max"]) | (array2 < CONFIG["height_min"]), np.nan, array2)

    cmap2, norm2 = draw_height_map(
        fig, ax2, lon2, lat2, array2, number="B",
        show_colorbar=True,
        cb_anchor=(0.39, 0.10),
        cb_tick_fs=8,
        cb_label_fs=9,
        cb_label_pad=4
    )

    draw_area_plot(ax2_density, array2, lat2, cmap2, norm2, 
                   lat_bin_size=CONFIG["lat_bin_size"], min_val=CONFIG["height_min"])

    ax_hist1 = plt.axes([0.28, 0.45, 0.07, 0.07])
    draw_histogram(ax_hist1, array2, CONFIG["height_min"], 
                   [5, 15, 30], ['5', '15', '30'], xlabel='PCH (m)', unit='m')

    # --------- Panel C: Cover Map ---------
    print("Loading cover map data...")
    lon3, lat3, array3 = get_geotiff_data(CONFIG["cover_masked_path"])
    array3 = np.where((array3 > CONFIG["cover_max"]) | (array3 < CONFIG["cover_min"]), np.nan, array3)

    cmap3, norm3 = draw_cover_map(
        fig, ax3, lon3, lat3, array3, number="C",
        show_colorbar=True,
        cb_anchor=(0.39, 0.10),
        cb_tick_fs=8,
        cb_label_fs=9,
        cb_label_pad=4
    )

    draw_area_plot(ax3_density, array3, lat3, cmap3, norm3, 
                   lat_bin_size=CONFIG["lat_bin_size"], min_val=CONFIG["cover_min"])

    ax_hist2 = plt.axes([0.28, 0.18, 0.07, 0.07])
    draw_histogram(ax_hist2, array3, CONFIG["cover_min"], 
                   [10, 30, 60], ['10', '30', '60'], xlabel='PTC (%)', unit='%')

    # Save output
    output_path = output_dir / CONFIG["output_filename"]
    print(f"Saving figure to: {output_path}")
    plt.savefig(
        output_path,
        dpi=CONFIG["output_dpi"],
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white"
    )
    
    print("Figure generation complete!")
    plt.show()


if __name__ == "__main__":
    main()
