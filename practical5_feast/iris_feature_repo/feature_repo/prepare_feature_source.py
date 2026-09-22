import pandas as pd
from pathlib import Path

input_file = Path("../../../data/processed/iris_features.csv")
output_file = Path("data/iris_features.parquet")

df = pd.read_csv(input_file)

df.insert(0, "sample_id", range(len(df)))

start_time = pd.Timestamp("2026-08-15 15:20:02", tz="UTC")
df["event_timestamp"] = pd.date_range(
    start=start_time,
    periods=len(df),
    freq="min"
)

df["created_timestamp"] = df["event_timestamp"]

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_file, index=False)

print(f"Wrote {len(df)} rows to {output_file}")
