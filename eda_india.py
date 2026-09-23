"""
eda_india.py
------------
Generates India-specific exploratory plots, with a focus on the
crop-burning season effect, saved to outputs/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_prep_india import load_and_clean, CROP_BURNING_HOTSPOT_CITIES

OUT = "outputs"


def main():
    df = load_and_clean()
    sns.set_style("whitegrid")

    # 1. Average AQI by month, across all cities -- the Oct/Nov spike
    plt.figure(figsize=(10, 5))
    monthly = df.groupby("month")["AQI"].mean()
    colors = ["crimson" if m in (10, 11) else "steelblue" for m in monthly.index]
    monthly.plot(kind="bar", color=colors)
    plt.title("Average AQI by Month, All Cities (Oct/Nov = crop burning season, in red)")
    plt.xlabel("Month")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(f"{OUT}/india_aqi_by_month.png", dpi=120)
    plt.close()

    # 2. Crop-burning-season vs rest-of-year AQI, hotspot cities vs others
    df["group"] = df["City"].apply(lambda c: c if c in CROP_BURNING_HOTSPOT_CITIES else "Other cities (avg)")
    comparison = (
        df.groupby(["group", "is_crop_burning_season"])["AQI"]
        .mean()
        .unstack()
        .rename(columns={False: "Rest of year", True: "Crop-burning season"})
    )
    # Collapse "Other cities" into a single average row for readability
    other_rows = df[~df["City"].isin(CROP_BURNING_HOTSPOT_CITIES)]
    other_avg = other_rows.groupby("is_crop_burning_season")["AQI"].mean()
    present_hotspots = [c for c in CROP_BURNING_HOTSPOT_CITIES if c in comparison.index]
    comparison = comparison.loc[present_hotspots]
    comparison.loc["All other cities (avg)"] = [other_avg.get(False), other_avg.get(True)]

    comparison.plot(kind="bar", figsize=(11, 6), color=["steelblue", "crimson"])
    plt.title("AQI: Crop-Burning Season vs Rest of Year")
    plt.ylabel("Average AQI")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(f"{OUT}/india_crop_burning_impact.png", dpi=120)
    plt.close()

    # 3. Delhi daily AQI trend across all years, crop-burning windows shaded
    delhi = df[df["City"] == "Delhi"].set_index("Date").sort_index()
    plt.figure(figsize=(14, 5))
    plt.plot(delhi.index, delhi["AQI"], color="steelblue", linewidth=0.8)
    for year in delhi.index.year.unique():
        start = pd.Timestamp(year=year, month=10, day=15)
        end = pd.Timestamp(year=year, month=11, day=30)
        plt.axvspan(start, end, color="crimson", alpha=0.2)
    plt.title("Delhi Daily AQI, 2015-2020 (shaded = crop-burning season)")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(f"{OUT}/india_delhi_aqi_timeline.png", dpi=120)
    plt.close()

    # 4. Most polluted cities specifically during crop-burning season
    plt.figure(figsize=(10, 6))
    season_avg = (
        df[df["is_crop_burning_season"]]
        .groupby("City")["AQI"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    season_avg.plot(kind="barh", color="darkorange")
    plt.title("Top 10 Cities by Average AQI During Crop-Burning Season")
    plt.xlabel("Average AQI (mid-Oct to end-Nov)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUT}/india_top_cities_crop_burning.png", dpi=120)
    plt.close()

    print("Saved 4 plots to outputs/:")
    print(" - india_aqi_by_month.png")
    print(" - india_crop_burning_impact.png")
    print(" - india_delhi_aqi_timeline.png")
    print(" - india_top_cities_crop_burning.png")


if __name__ == "__main__":
    main()
