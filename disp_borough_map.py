import os

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
path = os.getcwd()
geojson_borough_file = "resources/boroughs.geojson"
output_dir = "output"
n_quantiles = 2  # Using 2 for 5 boroughs is more stable
FINAL_DATASET_CSV = "final_dataset.csv"

# Color palettes for bivariate maps
bivariate_colors = {
    "0-0": "#e8e8e8",
    "0-1": "#b0d5df",
    "1-0": "#e4acac",
    "1-1": "#ad9ea5",
}


def get_borough_from_uhf(uhf):
    """Maps UHF42 code to a borough name."""
    uhf = int(uhf)
    if 101 <= uhf <= 107:
        return "Bronx"
    elif 201 <= uhf <= 211:
        return "Brooklyn"
    elif 301 <= uhf <= 310:
        return "Manhattan"
    elif 401 <= uhf <= 410:
        return "Queens"
    elif 501 <= uhf <= 504:
        return "Staten Island"
    else:
        return "Unknown"


def prepare_borough_data(csv_path):
    """Loads and aggregates data to the borough level."""
    print("--- Preparing Borough Data ---")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Please run the main notebook to generate it."
        )

    df = pd.read_csv(csv_path)

    # Clean data
    df["Asthma_Count"] = df["Asthma_Count"].fillna(0)
    df = df.dropna(subset=["UHF42", "Population", "NO2", "O3"])

    # Map UHF42 to Borough
    df["Borough"] = df["UHF42"].apply(get_borough_from_uhf)
    df = df[df["Borough"] != "Unknown"]

    # Aggregate data by borough
    borough_summary = (
        df.groupby("Borough")
        .agg(
            NO2_avg=("NO2", "mean"),
            O3_avg=("O3", "mean"),
            Population_sum=("Population", "sum"),
            Asthma_Count_sum=("Asthma_Count", "sum"),
        )
        .reset_index()
    )

    # Calculate asthma rate (per 10,000 people)
    borough_summary["asthma_rate"] = (
        borough_summary["Asthma_Count_sum"] / borough_summary["Population_sum"]
    ) * 10000

    print("Borough data preparation complete.")
    print(borough_summary)
    return borough_summary


def prepare_geospatial_data(geojson_path, data_df):
    """Loads geospatial data and merges it with the processed borough DataFrame."""
    print("\n--- Preparing Geospatial Data for Boroughs ---")
    try:
        borough_gdf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"Could not load geospatial data from {geojson_path}. Error: {e}")
        return None

    # Merge data
    # The geojson uses 'name' for the borough name
    merged_gdf = borough_gdf.merge(
        data_df, left_on="name", right_on="Borough", how="left"
    )

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
    ax.set_axis_off()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved map to {output_path}")
    plt.clf()  # Clear the figure for the next plot


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
            n_quantiles,
            labels=range(n_quantiles),
            duplicates="drop",
        )
        var2_quantiles = pd.qcut(
            bivariate_gdf[var2_col],
            n_quantiles,
            labels=range(n_quantiles),
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

    for i in range(n_quantiles):
        for j in range(n_quantiles):
            color = colors.get(f"{i}-{j}")
            if color:
                rect = mpatches.Rectangle(
                    (j / n_quantiles, i / n_quantiles),
                    1 / n_quantiles,
                    1 / n_quantiles,
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
    """Main execution function to prepare data and generate all borough visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("bmh")

    # Prepare data
    borough_data = prepare_borough_data(FINAL_DATASET_CSV)

    # Prepare geospatial data
    map_gdf = prepare_geospatial_data(geojson_borough_file, borough_data)
    if map_gdf is None:
        print("Exiting due to geospatial data loading failure.")
        return

    # --- Generate Maps ---

    # 1. Asthma Rate Map
    plot_choropleth(
        map_gdf,
        "asthma_rate",
        "Asthma Rate by Borough",
        os.path.join(output_dir, "map_borough_asthma_rate.png"),
        cmap="Reds",
        legend_label="Asthma Cases per 10,000 Residents",
    )

    # 2. NO2 Map
    plot_choropleth(
        map_gdf,
        "NO2_avg",
        "Average NO2 Levels by Borough",
        os.path.join(output_dir, "map_borough_no2.png"),
        cmap="viridis",
        legend_label="Average NO2 (ppb)",
    )

    # 3. O3 Map
    plot_choropleth(
        map_gdf,
        "O3_avg",
        "Average O3 Levels by Borough",
        os.path.join(output_dir, "map_borough_o3.png"),
        cmap="plasma",
        legend_label="Average O3 (ppb)",
    )

    # --- Generate Bivariate Maps ---
    plot_bivariate_map(
        map_gdf,
        "asthma_rate",
        "NO2_avg",
        "Asthma Rate",
        "NO2",
        bivariate_colors,
        "Bivariate Map: Asthma Rate vs. NO2 Levels",
        os.path.join(output_dir, "map_borough_bivariate_asthma_no2.png"),
    )

    plot_bivariate_map(
        map_gdf,
        "asthma_rate",
        "O3_avg",
        "Asthma Rate",
        "O3",
        bivariate_colors,
        "Bivariate Map: Asthma Rate vs. O3 Levels",
        os.path.join(output_dir, "map_borough_bivariate_asthma_o3.png"),
    )

    print("\nAll borough visualizations complete.")


if __name__ == "__main__":
    main()
