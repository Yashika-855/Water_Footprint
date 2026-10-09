
import _bootstrap  # noqa
import joblib, json, pandas as pd
from pathlib import Path
from src.config import load_config
from src.models import make_model
from src.evaluation import compute_metrics

cfg = load_config(); proc = Path(cfg["paths"]["processed"]); rows = []
for t in ["green_wf", "blue_wf", "total_wf"]:
    s = joblib.load(proc / f"split_{t}.joblib")
    top = json.load(open(proc / "selected_features.json"))[t]
    m = make_model(cfg).fit(s["X_train"][top], s["y_train"])
    rows.append({"target": t, "clustering": "none (global)",
                 **compute_metrics(s["y_test"], m.predict(s["X_test"][top]))})
out = pd.DataFrame(rows)
out.to_csv(Path(cfg["paths"]["tables"]) / "baseline_global.csv", index=False)
print(out.to_string(index=False))