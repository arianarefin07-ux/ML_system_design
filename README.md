## Part 1: DVC (Data Version Control)

### 1. Setup

DVC is just a Python package. We layer it on top of an existing Git repo.

```bash
# Create and enter the project
mkdir diamond-dvc && cd diamond-dvc

# Python environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install DVC + ML libs
pip install "dvc[all]" pandas scikit-learn pyyaml seaborn

# Initialize Git FIRST — DVC requires a Git repo
git init

# Initialize DVC. This creates .dvc/ and .dvcignore
dvc init

# Commit the DVC scaffolding so teammates get the same setup
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

What `dvc init` creates:

```
.dvc/
├── .gitignore        # Tells Git to ignore the cache
├── config            # DVC config (remotes, cache location)
└── tmp/              # Working state
.dvcignore            # Like .gitignore but for DVC tracking
```

**Configure a remote.** A remote is where the big files actually live. For class, a local folder is easiest; in production, use S3 or GCS.

```bash
# Local remote (good for teaching)
mkdir -p /tmp/dvc-remote-storage
dvc remote add -d localremote /tmp/dvc-remote-storage

# S3 example (don't run, just shown for reference)
# dvc remote add -d s3remote s3://my-bucket/dvc-store
# dvc remote modify s3remote region us-east-1

git add .dvc/config
git commit -m "Configure DVC remote"
```

The `-d` flag means *default*. `dvc push` will send data here; `dvc pull` will retrieve it.

---

### 2. The Special Commands (with deep mental models)

Think of DVC commands as falling into four groups:

#### Group A — Data versioning (Git-LFS-like)

```bash
dvc add data/diamonds.csv
```

This is the most important command to *really understand*. Here's what happens:

1. DVC computes the MD5 hash of `diamonds.csv` — say `a1b2c3...`.
2. It moves the file into `.dvc/cache/files/md5/a1/b2c3...` (content-addressable storage; same content → same path).
3. It creates `data/diamonds.csv.dvc`, a tiny YAML file containing the hash.
4. It adds the original `data/diamonds.csv` to `.gitignore` so Git doesn't track it.
5. The working file `data/diamonds.csv` becomes a **link** (reflink/symlink/copy depending on filesystem) to the cache entry.

The `.dvc` file you'll see:

```yaml
outs:
- md5: a1b2c3d4e5f6...
  size: 2829294
  path: diamonds.csv
```

You commit the `.dvc` file to Git. It's 200 bytes. Anyone who clones the repo runs `dvc pull` to materialize the 3 MB CSV from the remote.

```bash
dvc push                          # Upload cache → remote
dvc pull                          # Download from remote → cache → workspace
dvc status                        # Compare workspace vs cache vs remote
dvc checkout                      # Restore working files from cache (after git checkout)
```

#### Group B — Pipelines

```bash
dvc stage add ...                 # Define a stage (verbose; we usually edit dvc.yaml directly)
dvc repro                         # Run the pipeline; skip stages whose inputs are unchanged
dvc dag                           # ASCII visualization of the pipeline graph
```

#### Group C — Experiments & metrics

```bash
dvc exp run                       # Run pipeline as a tracked experiment
dvc exp show                      # Table of all experiments with params + metrics
dvc exp diff                      # Compare two experiments
dvc metrics show                  # Just show the latest metrics
dvc params diff                   # Compare params across commits
dvc plots show                    # Render plots (e.g., confusion matrix from JSON)
```

#### Group D — Time travel

```bash
git checkout <commit>             # Go back to a code version
dvc checkout                      # Make the data match that code version
```

This combo is why DVC matters. You can return to *any* historical commit and the matching data magically reappears.

---

### 3. Split / Train / Evaluate pipeline

We'll build a real pipeline predicting diamond price.

#### Project layout

```
diamond-dvc/
├── data/
│   └── diamonds.csv              # Raw data (DVC-tracked)
├── src/
│   ├── split.py
│   ├── train.py
│   └── evaluate.py
├── params.yaml                   # All hyperparameters in one place
├── dvc.yaml                      # Pipeline definition
└── .gitignore
```

#### `params.yaml` — single source of truth for hyperparameters

```yaml
split:
  test_size: 0.2
  random_state: 42

train:
  model: random_forest
  n_estimators: 100
  max_depth: 12
  random_state: 42
```

DVC tracks this file specially. Change `n_estimators` from 100 to 200, and DVC knows the *train* stage is stale even though no code changed.

#### `src/split.py`

```python
"""Split diamonds.csv into train/test, encode categoricals."""
import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Read params from the YAML file
params = yaml.safe_load(open("params.yaml"))["split"]

# CLI: python split.py <input_csv> <output_dir>
input_path = sys.argv[1]
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

# Ordinal encoding for the three quality categoricals.
# These have natural ordering — we should not one-hot them blindly.
cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order   = ["J", "I", "H", "G", "F", "E", "D"]            # J worst → D best
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

df["cut"]     = df["cut"].map({v: i for i, v in enumerate(cut_order)})
df["color"]   = df["color"].map({v: i for i, v in enumerate(color_order)})
df["clarity"] = df["clarity"].map({v: i for i, v in enumerate(clarity_order)})

# Drop any unnamed index column some versions of the dataset ship with
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
)

train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
```

#### `src/train.py`

```python
"""Train a RandomForest regressor on the prepared diamond data."""
import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

params = yaml.safe_load(open("params.yaml"))["train"]

train_csv = sys.argv[1]
model_out = Path(sys.argv[2])
model_out.parent.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(train_csv)
X = train_df.drop(columns=["price"])
y = train_df["price"]

model = RandomForestRegressor(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"],
    n_jobs=-1,
)
model.fit(X, y)

joblib.dump(model, model_out)
print(f"Saved model → {model_out}")
```

#### `src/evaluate.py`

```python
"""Evaluate the trained model and write metrics.json."""
import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

model_path   = sys.argv[1]
test_csv     = sys.argv[2]
metrics_path = Path(sys.argv[3])
metrics_path.parent.mkdir(parents=True, exist_ok=True)

model = joblib.load(model_path)
test_df = pd.read_csv(test_csv)

X = test_df.drop(columns=["price"])
y_true = test_df["price"]
y_pred = model.predict(X)

metrics = {
    "mae":  float(mean_absolute_error(y_true, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "r2":   float(r2_score(y_true, y_pred)),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
```

#### `dvc.yaml` — the pipeline definition

This is the file students should stare at until it clicks:

```yaml
stages:
  split:
    cmd: python src/split.py data/diamonds.csv data/prepared
    deps:
      - src/split.py
      - data/diamonds.csv
    params:
      - split
    outs:
      - data/prepared/train.csv
      - data/prepared/test.csv

  train:
    cmd: python src/train.py data/prepared/train.csv models/model.joblib
    deps:
      - src/train.py
      - data/prepared/train.csv
    params:
      - train
    outs:
      - models/model.joblib

  evaluate:
    cmd: python src/evaluate.py models/model.joblib data/prepared/test.csv metrics/metrics.json
    deps:
      - src/evaluate.py
      - models/model.joblib
      - data/prepared/test.csv
    metrics:
      - metrics/metrics.json:
          cache: false
```

Key fields:

- **`deps`** — files this stage reads. If any of them change (by hash), the stage is stale.
- **`params`** — which key in `params.yaml` this stage depends on. Stage-scoped, so changing `train.n_estimators` does *not* invalidate the *split* stage.
- **`outs`** — files this stage produces. DVC takes them over: gitignores them and tracks them in its cache.
- **`metrics`** — like `outs`, but small JSON/YAML files that DVC can read and display in tables. `cache: false` keeps them in Git directly (so you can `git diff` them).

#### Running the pipeline

```bash
# Add the raw data
dvc add data/diamonds.csv
git add data/diamonds.csv.dvc data/.gitignore
git commit -m "Track raw diamonds dataset"

# Reproduce — first run, everything executes
dvc repro

# Inspect
dvc dag
dvc metrics show

# Commit the pipeline lock file (the snapshot of what ran)
git add dvc.yaml dvc.lock params.yaml src/
git commit -m "Add split/train/evaluate pipeline"
dvc push
```

`dvc.lock` is auto-generated and pins the hashes of every dep, param, and output from the last successful run. It's like `package-lock.json` for ML.

#### The magic moment: change one parameter

```bash
# Bump n_estimators in params.yaml from 100 to 300
sed -i 's/n_estimators: 100/n_estimators: 300/' params.yaml

dvc repro
```

Watch the output:

```
Stage 'split' didn't change, skipping
Running stage 'train':
> python src/train.py data/prepared/train.csv models/model.joblib
Running stage 'evaluate':
> python src/evaluate.py models/model.joblib ...
```

**That's the payoff.** Split was correctly skipped — we proved it cost-free to scale parameter sweeps.

#### Experiments

```bash
# Try several values without polluting Git history
dvc exp run --set-param train.n_estimators=50
dvc exp run --set-param train.n_estimators=200
dvc exp run --set-param train.max_depth=20

# Compare them in a table
dvc exp show

# Promote a good one to a Git commit
dvc exp branch <exp-name> better-model
```

---
