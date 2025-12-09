import os
import pickle
from io import StringIO

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
path = os.getcwd()
GEOJSON_LOCAL_FILE = "resources/nyc-zip-code-tabulation-areas-polygons.geojson"
OUTPUT_DIR = "output"
ASTHMA_PKL = "asthma_pd.pkl"
FINAL_DF_PKL = "final_df.pkl"
GEOJSON_LOCAL_PATH = os.path.join(path, GEOJSON_LOCAL_FILE)
N_QUANTILES = 3

# Many-to-many mapping between ZIP3 and UHF42.
UHF_TO_ZIP3_MAP_DATA = """UHF42,zip3
101,104
102,104
103,104
104,104
105,104
106,104
107,104
201,112
202,111
203,111
204,112
205,112
206,111
207,111
208,112
209,111
210,111
211,112
301,100
302,100
303,100
304,100
305,101
306,100
307,100
308,100
309,100
310,102
401,111
402,113
403,113
404,113
405,113
406,113
407,114
408,114
409,110
410,116
501,103
502,103
503,103
504,103
"""

# Color palettes for bivariate maps
BIVARIATE_COLORS = {
    "0-0": "#e8e8e8",
    "0-1": "#b0d5df",
    "0-2": "#64acbe",
    "1-0": "#e4acac",
    "1-1": "#ad9ea5",
    "1-2": "#627f8c",
    "2-0": "#c85a5a",
    "2-1": "#985356",
    "2-2": "#574249",
}


def prepare_data(asthma_path, final_df_path, uhf_zip_map_csv):
    """Loads and processes the data to create a merged ZIP3-level DataFrame."""
    print("--- Preparing Data ---")
    with open(asthma_path, "rb") as f:
        asthma_pd = pickle.load(f)
    with open(final_df_path, "rb") as f:
        final_df = pickle.load(f)

    zip3_map = pd.read_csv(StringIO(uhf_zip_map_csv))
    zip3_map["zip3"] = zip3_map["zip3"].astype(str)

    uhf_data_with_zip3 = pd.merge(final_df, zip3_map, on="UHF42")
    aggregated_uhf_data = (
        uhf_data_with_zip3.groupby("zip3")
        .agg(
            NO2_avg=("NO2", "mean"),
            O3_avg=("O3", "mean"),
            Population_sum=("Population", "sum"),
        )
        .reset_index()
    )

    geo_df = pd.merge(asthma_pd, aggregated_uhf_data, on="zip3", how="left")

    print("\nData preparation complete.")
    return geo_df


def prepare_geospatial_data(geojson_path, data_df):
    """Loads geospatial data and merges it with the processed DataFrame."""
    print("\n--- Preparing Geospatial Data ---")
    try:
        nyc_zcta_gdf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"Could not load geospatial data from {geojson_path}. Error: {e}")
        return None

    print("Creating ZIP3 boundaries...")
    nyc_zcta_gdf["zip3"] = nyc_zcta_gdf["postalCode"].str[:3]

    # Merge data first
    merged_gdf = nyc_zcta_gdf.merge(data_df, on="zip3", how="left")

    return merged_gdf


def plot_choropleth(gdf, column, title, output_path, cmap="plasma", legend_label=""):
    """Generates and saves a single-variable choropleth map."""
    print(f"Generating choropleth map for '{column}'...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        cmap=cmap,
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": legend_label, "orientation": "horizontal"},
    )
    ax.set_title(title, fontdict={"fontsize": "16", "fontweight": "3"})
    ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved map to {output_path}")
    plt.clf()


def plot_bivariate_map(
    gdf, var1_col, var2_col, var1_label, var2_label, colors, title, output_path
):
    """Generates and saves a bivariate choropleth map."""
    print(f"--- Starting Bivariate Choropleth ( {var1_label} vs {var2_label} ) ---")

    bivariate_gdf = gdf.dropna(subset=[var1_col, var2_col]).copy()
    if bivariate_gdf.empty:
        print(f"No data to plot for {title}. Skipping.")
        return

    # Classify data
    try:
        var1_quantiles = pd.qcut(
            bivariate_gdf[var1_col],
            N_QUANTILES,
            labels=range(N_QUANTILES),
            duplicates="drop",
        )
        var2_quantiles = pd.qcut(
            bivariate_gdf[var2_col],
            N_QUANTILES,
            labels=range(N_QUANTILES),
            duplicates="drop",
        )
    except ValueError as e:
        print(
            f"Could not classify data for bivariate map '{title}'. Error: {e}. Skipping."
        )
        return

    bivariate_gdf["bivariate_class"] = (
        var1_quantiles.astype(str) + "-" + var2_quantiles.astype(str)
    )
    bivariate_gdf["color"] = bivariate_gdf["bivariate_class"].map(colors)

    # Plot map
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    bivariate_gdf.plot(
        color=bivariate_gdf["color"], ax=ax, edgecolor="white", linewidth=0.5
    )

    # Create legend
    legend_ax = fig.add_axes([0.05, 0.75, 0.2, 0.2], zorder=5)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_facecolor("#f0f0f0")
    legend_ax.spines[:].set_visible(False)

    for i in range(N_QUANTILES):
        for j in range(N_QUANTILES):
            color = colors.get(f"{i}-{j}")
            if color:
                rect = mpatches.Rectangle(
                    (j / N_QUANTILES, i / N_QUANTILES),
                    1 / N_QUANTILES,
                    1 / N_QUANTILES,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                legend_ax.add_patch(rect)

    legend_ax.text(
        0.5,
        1.05,
        f"High {var2_label} →",
        transform=legend_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    legend_ax.text(
        -0.05,
        0.5,
        f"High {var1_label} →",
        transform=legend_ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        rotation=90,
    )

    # Finalize and save
    ax.set_title(title, fontdict={"fontsize": "18", "fontweight": "3"})
    ax.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved map to {output_path}")
    plt.clf()


def main():
    """Main execution function to prepare data and generate all visualizations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    geo_df = prepare_data(ASTHMA_PKL, FINAL_DF_PKL, UHF_TO_ZIP3_MAP_DATA)

    # --- DEBUGGING PRINTS ---
    print("\n--- DEBUG: Inspecting aggregated data in geo_df ---")
    print(
        geo_df[
            geo_df["zip3"].isin(
                [
                    "100",
                    "101",
                    "102",
                    "104",
                ]
            )
        ],
    )
    print("end")
    # --- END DEBUGGING ---

    # Prepare geospatial data
    map_gdf = prepare_geospatial_data(GEOJSON_LOCAL_PATH, geo_df)

    bivariate_df = map_gdf.drop_duplicates(subset=["zip3"])

    # DEBUG
    print("\n--- DEBUG: Inspecting aggregated data in geo_df ---")

    print("map_gdf:", map_gdf.describe())

    # print NaN or 0 values in O3
    print("O3_avg NaN or 0:")
    print(map_gdf[map_gdf["O3_avg"] == 0])

    print("unique:", map_gdf.groupby("zip3")["O3_avg"].nunique())

    # merged_gdf[merged_gdf["zip3"] == "100"][["postalCode","zip3","O3_avg","NO2_avg","Asthma_Count"]]

    print("end of DEBUG")
    # END DEBUG

    if map_gdf is None:
        print("Exiting due to geospatial data loading failure.")
        return

    plt.style.use("bmh")

    # Generate single-variable maps

    plot_choropleth(
        map_gdf,
        "Asthma_Count",
        "Total Asthma Hospitalizations by ZIP3",
        os.path.join(OUTPUT_DIR, "map_zip3_asthma_count.png"),
        cmap="Reds",
        legend_label="Total Asthma Cases",
    )

    plot_choropleth(
        map_gdf,
        "NO2_avg",
        "Average NO2 Levels by ZIP3",
        os.path.join(OUTPUT_DIR, "map_zip3_no2.png"),
        cmap="viridis",
        legend_label="Average NO2 (ppb)",
    )

    plot_choropleth(
        map_gdf,
        "O3_avg",
        "Average O3 Levels by ZIP3",
        os.path.join(OUTPUT_DIR, "map_zip3_o3.png"),
        cmap="plasma",
        legend_label="Average O3 (ppb)",
    )

    # Generate bivariate maps
    plot_bivariate_map(
        map_gdf,
        "Asthma_Count",
        "NO2_avg",
        "Asthma",
        "NO2",
        BIVARIATE_COLORS,
        "Asthma Hospitalizations vs. NO2 Levels by ZIP3",
        os.path.join(OUTPUT_DIR, "map_zip3_bivariate_asthma_no2.png"),
    )

    plot_bivariate_map(
        map_gdf,
        "Asthma_Count",
        "O3_avg",
        "Asthma",
        "O3",
        BIVARIATE_COLORS,
        "Asthma Hospitalizations vs. O3 Levels by ZIP3",
        os.path.join(OUTPUT_DIR, "map_zip3_bivariate_asthma_o3.png"),
    )

    print("\nAll visualizations complete.")


if __name__ == "__main__":
    main()
