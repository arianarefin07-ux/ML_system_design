import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(open("params.yaml"))["train"]

train_csv = sys.argv[1]
model_out = Path(sys.argv[2])
model_out.parent.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(train_csv)
X = train_df.drop(columns=["deposit"])
y = train_df["deposit"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"],
    class_weight="balanced",   
    n_jobs=-1,
)
model.fit(X, y)

joblib.dump(model, model_out)
print(f"Saved model -> {model_out}")


importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10).to_string())
