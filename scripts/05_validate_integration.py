import pandas as pd

MASTER_PATH = "data/processed/master_wheat_dataset.csv"

df = pd.read_csv(MASTER_PATH)

print("Master shape:", df.shape)

# ---------------------------------------------------------
# Find rows where coordinates are missing
# ---------------------------------------------------------

bad_rows = df[
    df["latitude"].isna() |
    df["longitude"].isna()
]

print("\nRows with missing coordinates:", len(bad_rows))

print("\nSuspicious rows:")
print(
    bad_rows[
        [
            "latitude",
            "longitude",
            "green_wf",
            "blue_wf",
            "total_wf",
            "planting_doy",
            "maturity_doy",
            "pr_m01",
            "tasmax_m01",
            "soil_DOMSOI_Af"
        ]
    ]
)

print("\nRow indices:")
print(bad_rows.index.tolist())