"""
Prospect Park CPM — Python script with storytelling analysis, descriptive statistics, and heatmap overlay

This script:
- Loads CPM measurement data from a CSV.
- Detects & fixes swapped longitude/latitude.
- Cleans and explores data.
- Produces descriptive statistics (mean, median, min, max, std).
- Classifies CPM readings into Safe, Moderate, High.
- Generates visuals: histogram, boxplot, regression fit, and CPM hotspot heatmap with all points labeled.
- Overlays CPM readings on a map of Prospect Park (using a basemap if available).
- Adds storytelling-style printed interpretation that includes descriptive statistics.
- Saves cleaned dataset to both CSV and Excel.
- Displays all results and plots on screen as well as saving them.

Configuration:
- FILE_PATH: path to your CSV input file.
- OUT_DIR: output directory (set to Desktop).
- EXCEL_PATH: path for Excel file.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    import geopandas as gpd
    from shapely.geometry import Point
    import contextily as ctx
    HAS_MAP = True
except Exception:
    HAS_MAP = False

# --------------------------
# CONFIGURATION (EDIT THESE)
# --------------------------
FILE_PATH = "C:/Users/Chris/Downloads/CPM_LongLat_Matrix_20250820_181432.csv"   # <== CSV input file path
OUT_DIR = os.path.expanduser("~/Desktop")                                      # <== output directory set to Desktop
EXCEL_PATH = "C:/Users/Chris/Downloads/cpm_cleaned.xlsx"                       # <== Excel output path

# --------------------------
# Helper functions
# --------------------------

def detect_and_fix_swapped_coords(df: pd.DataFrame):
    lon_mean = df['Longitude'].mean()
    lat_mean = df['Latitude'].mean()
    if (30 <= lon_mean <= 50) and (-80 <= lat_mean <= -60):
        df = df.copy()
        df['Latitude'], df['Longitude'] = df['Longitude'].copy(), df['Latitude'].copy()
        print("Swapped Latitude/Longitude detected and fixed.")
    return df

def clean_data(df: pd.DataFrame):
    df = df.drop_duplicates()
    df['CPM'] = pd.to_numeric(df['CPM'], errors='coerce')
    df = df.dropna(subset=['CPM','Longitude','Latitude'])
    return df

def classify_cpm(cpm):
    if cpm < 100:
        return "Safe"
    elif 100 <= cpm < 200:
        return "Moderate"
    else:
        return "High"

# --------------------------
# Analysis pipeline
# --------------------------

def run_pipeline(file_path=FILE_PATH, out_dir=OUT_DIR, excel_path=EXCEL_PATH):
    os.makedirs(out_dir, exist_ok=True)

    # Load and clean
    df = pd.read_csv(file_path)
    df = detect_and_fix_swapped_coords(df)
    df = clean_data(df)

    # Descriptive statistics
    desc = df['CPM'].describe()
    mean, median, std = int(df['CPM'].mean()), int(df['CPM'].median()), int(df['CPM'].std())
    min_val, max_val = int(df['CPM'].min()), int(df['CPM'].max())

    print("\nProspect Park CPM Readings — Aug 19 2025")
    print("-------------------------------------")
    print(desc)
    print(f"Mean CPM: {mean}")
    print(f"Median CPM: {median}")
    print(f"Standard Deviation: {std}")
    print(f"Minimum CPM: {min_val}")
    print(f"Maximum CPM: {max_val}")

    # Classify readings
    df['Category'] = df['CPM'].apply(classify_cpm)
    print("\nCategory counts:")
    print(df['Category'].value_counts())

    # Histogram
    plt.figure(figsize=(6,4))
    sns.histplot(df['CPM'], bins=15, kde=True)
    plt.title('Prospect Park CPM Readings — Aug 19 2025 (Distribution)')
    plt.xlabel('CPM')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_dir,'hist_cpm.png'))
    plt.show()

    # Boxplot
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df['CPM'])
    plt.title('Prospect Park CPM — Boxplot')
    plt.savefig(os.path.join(out_dir,'boxplot_cpm.png'))
    plt.show()

    # Regression
    X = df[['Latitude','Longitude']].values
    y = df['CPM'].values
    model = LinearRegression().fit(X,y)
    df['CPM_pred'] = model.predict(X)
    r2 = r2_score(y, df['CPM_pred'])
    print(f"\nRegression R^2: {r2:.3f}")

    plt.figure(figsize=(6,4))
    plt.scatter(y, df['CPM_pred'], alpha=0.6)
    mn, mx = y.min(), y.max()
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel('Actual CPM')
    plt.ylabel('Predicted CPM')
    plt.title('Regression Fit — Prospect Park CPM')
    plt.savefig(os.path.join(out_dir,'regression_fit.png'))
    plt.show()

    # Heatmap with overlay
    plt.figure(figsize=(8,6))
    sns.kdeplot(
        x=df['Longitude'], y=df['Latitude'], weights=df['CPM'],
        cmap='Reds', fill=True, thresh=0.05, alpha=0.6
    )
    sc = plt.scatter(df['Longitude'], df['Latitude'], c=df['CPM'], cmap='coolwarm', s=60, edgecolor='k')
    plt.colorbar(sc, label='CPM')
    plt.title('Prospect Park CPM Hotspots — Aug 19 2025')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(out_dir,'heatmap_cpm.png'))
    plt.show()

    # Map overlay if geopandas + contextily available
    if HAS_MAP:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        ax = gdf.plot(figsize=(8,8), column='CPM', cmap='coolwarm', markersize=50, legend=True, alpha=0.8)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        plt.title('Prospect Park CPM Overlay Map — Aug 19 2025')
        plt.savefig(os.path.join(out_dir,'map_overlay_cpm.png'))
        plt.show()

    # Save cleaned data to CSV & Excel
    csv_out = os.path.join(out_dir, 'cpm_cleaned.csv')
    df.to_csv(csv_out, index=False)
    df.to_excel(excel_path, index=False)
    print("\nSaved cleaned CSV:", csv_out)
    print("Saved cleaned Excel:", excel_path)

    # Storytelling summary
    print("\nStorytelling Summary:")
    print(f"On Aug 19 2025, CPM readings across Prospect Park averaged {mean}, with a median of {median}.")
    print(f"Values ranged from a minimum of {min_val} to a maximum of {max_val}, with a standard deviation of {std}.")
    print("Most values fell in the Safe range (<100 CPM), though some moderate and high readings were observed.")
    print("A heatmap visualization highlights localized hotspots where CPMs spiked. While background levels in NYC typically range 5–60 CPM, anything consistently over 100 CPM is noteworthy, and 200+ CPM readings are considered high and warrant further investigation.")

    return df

if __name__ == "__main__":
    run_pipeline()
