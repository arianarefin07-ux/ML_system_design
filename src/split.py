import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["split"]

input_path = sys.argv[1]
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

# Drop unnamed index columns if any
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# ── Binary categorical columns (yes/no) ──────────────────────────────────────
binary_cols = ["default", "housing", "loan"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# ── Target column ─────────────────────────────────────────────────────────────
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

# ── Ordinal encoding ──────────────────────────────────────────────────────────
education_order = ["unknown", "primary", "secondary", "tertiary"]
df["education"] = df["education"].map({v: i for i, v in enumerate(education_order)})

# ── One-hot encoding for nominal categoricals ─────────────────────────────────
nominal_cols = ["job", "marital", "contact", "month", "poutcome"]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["deposit"],   # preserve class balance in both splits
)

train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
print(f"Features: {train_df.shape[1] - 1}")
print(f"Train deposit rate: {train_df['deposit'].mean():.2%}")
print(f"Test  deposit rate: {test_df['deposit'].mean():.2%}")
