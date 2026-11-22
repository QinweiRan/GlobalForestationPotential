import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "..."
ECOSOCIE_DIR = os.path.join(DATA_DIR, "ecosocie")

# Domain, cover and FAO mask
PATH_DOMAIN_SHP = os.path.join(DATA_DIR, "ecoregion2017_biome.shp")
PATH_C_MEAN = os.path.join(DATA_DIR, "PTC_bootstrap_mean.tif")
PATH_FAO_MASK = os.path.join(DATA_DIR, "FAO_mask.tif")

# Ecological constraints
PATH_ALBEDO = os.path.join(ECOSOCIE_DIR, "NetClimateImpact.tif")
PATH_WATER = os.path.join(ECOSOCIE_DIR, "water_availability.tif")
PATH_BIODIV = os.path.join(ECOSOCIE_DIR, "biodiver.tif")

# Socioeconomic constraints
PATH_DEPRIV = os.path.join(ECOSOCIE_DIR, "livelihood.tif")
PATH_LIBDEM = os.path.join(ECOSOCIE_DIR, "Individual_rights.tif")
PATH_TENURE = os.path.join(ECOSOCIE_DIR, "landtenure.tif")
PATH_COMMIT = os.path.join(ECOSOCIE_DIR, "commitments.tif")

C_SCALE = 100.0


# ============================================================
# HELPERS
# ============================================================
def clean(arr, nodata):
    """NoData/NaN -> 0."""
    arr = np.where(np.isnan(arr), 0.0, arr)
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == nodata, 0.0, arr)
    return arr


def warp_to_grid(path, ref_crs, ref_transform, ref_w, ref_h, resampling):
    """Reproject to reference grid."""
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=ref_crs,
            transform=ref_transform,
            width=ref_w,
            height=ref_h,
            resampling=resampling
        ) as vrt:
            arr = vrt.read(1).astype(np.float32)
            nodata = vrt.nodata

    return clean(arr, nodata)


def rasterize_domain(gdf, ref_crs, ref_transform, ref_shape):
    """Rasterize ecoregion domain."""
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    geoms = [
        g for g in gdf.geometry
        if g is not None and not g.is_empty
    ]

    if not geoms:
        raise ValueError("No usable geometries")

    mask = rasterize(
        ((g, 1) for g in geoms),
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        all_touched=False,
        dtype="uint8"
    )

    return mask.astype(np.float32)


def weighted_area_mha(mask, c_frac, pixel_area_m2):
    """Canopy-cover-weighted area (Mha)."""
    return float((mask * c_frac).sum()) * pixel_area_m2 / 1e10


# ============================================================
# MAIN
# ============================================================
def main():

    # 1. Reference grid
    gdf = gpd.read_file(PATH_DOMAIN_SHP)
    target_crs = gdf.crs

    with rasterio.open(PATH_C_MEAN) as src:
        with WarpedVRT(
            src,
            crs=target_crs,
            resampling=Resampling.nearest
        ) as vrt:
            ref_crs = vrt.crs
            ref_transform = vrt.transform
            ref_w = vrt.width
            ref_h = vrt.height
            cover = vrt.read(1).astype(np.float32)
            c_nodata = vrt.nodata

    ref_shape = (ref_h, ref_w)
    pixel_area_m2 = abs(ref_transform.a) * abs(ref_transform.e)

    # 2. Cover
    cover = clean(cover, c_nodata)
    cover = np.where(
        (cover >= 0) & (cover <= 100),
        cover,
        0.0
    )

    # 3. Domain and FAO mask
    domain = rasterize_domain(
        gdf, ref_crs, ref_transform, ref_shape
    )

    fao = (
        warp_to_grid(
            PATH_FAO_MASK,
            ref_crs, ref_transform, ref_w, ref_h,
            Resampling.nearest
        ) == 1
    ).astype(np.float32)

    c_frac = domain * fao * (cover / C_SCALE)

    area_landcover = weighted_area_mha(
        1.0, c_frac, pixel_area_m2
    )

    del cover, fao, domain

    # 4. Ecological constraints (sequential)
    albedo = warp_to_grid(
        PATH_ALBEDO,
        ref_crs, ref_transform, ref_w, ref_h,
        Resampling.bilinear
    )

    area_albedo = weighted_area_mha(
        albedo, c_frac, pixel_area_m2
    )

    water = warp_to_grid(
        PATH_WATER,
        ref_crs, ref_transform, ref_w, ref_h,
        Resampling.bilinear
    )

    m_water = albedo * water

    area_water = weighted_area_mha(
        m_water, c_frac, pixel_area_m2
    )

    biodiv = warp_to_grid(
        PATH_BIODIV,
        ref_crs, ref_transform, ref_w, ref_h,
        Resampling.nearest
    )

    eco_pivot = m_water * biodiv

    area_biodiv = weighted_area_mha(
        eco_pivot, c_frac, pixel_area_m2
    )

    del albedo, water, biodiv, m_water

    # 5. Socioeconomic scenarios (independent of each other)
    socio_layers = [
        ("S1: Livelihood", PATH_DEPRIV, Resampling.bilinear),
        ("S2: Individual rights", PATH_LIBDEM, Resampling.nearest),
        ("S3: Land tenure", PATH_TENURE, Resampling.nearest),
        ("S4: Commitments", PATH_COMMIT, Resampling.nearest)
    ]

    socio_results = {}
    combined = eco_pivot.copy()

    for label, path, resampling in socio_layers:

        layer = warp_to_grid(
            path,
            ref_crs, ref_transform, ref_w, ref_h,
            resampling
        )

        scenario_mask = eco_pivot * layer

        socio_results[label] = weighted_area_mha(
            scenario_mask, c_frac, pixel_area_m2
        )

        combined *= layer

        del layer, scenario_mask

    socio_results["S5: All combined"] = weighted_area_mha(
        combined, c_frac, pixel_area_m2
    )

    # 6. Print areas
    print("\n" + "=" * 48)
    print("FORESTATION POTENTIAL CASCADE (Mha)")
    print("=" * 48)

    print(f"{'Biophysical potential':30s}: {area_landcover:10.3f}")
    print(f"{'+ Albedo':30s}: {area_albedo:10.3f}")
    print(f"{'+ Downstream water':30s}: {area_water:10.3f}")
    print(f"{'+ Biodiversity (Eco-pivot)':30s}: {area_biodiv:10.3f}")

    print("-" * 48)

    for label, area in socio_results.items():
        print(f"{label:30s}: {area:10.3f}")

    print("=" * 48)


if __name__ == "__main__":
    main()
