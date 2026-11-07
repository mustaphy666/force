"""Shared feature engineering + prediction logic.

The same functions are used for training (train_final.py) and inference
(app/app.py), so the app can never drift from the trained model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONF = "FORCE_2020_LITHOFACIES_CONFIDENCE"
BASE = ["GR", "RHOB", "NPHI", "DTC", "RDEP", "PEF"]
WINDOW = 15  # rolling window in samples (0.15 m spacing -> ~2.25 m)

LITH_NAMES = {
    30000: "Sandstone", 65030: "Sand/Shale", 65000: "Shale", 80000: "Marl",
    74000: "Dolomite", 70000: "Limestone", 70032: "Chalk", 88000: "Halite",
    86000: "Anhydrite", 99000: "Tuff", 90000: "Coal", 93000: "Basement",
}

LITH_COLORS = {
    "Sandstone": "#e8c547", "Sand/Shale": "#c9b48a", "Shale": "#7f8c8d",
    "Marl": "#9aa86b", "Dolomite": "#8e6bbf", "Limestone": "#4a90d9",
    "Chalk": "#a8dadc", "Halite": "#f4a6a6", "Anhydrite": "#e07bb0",
    "Tuff": "#d9534f", "Coal": "#222222", "Basement": "#8b5a2b",
}

# Common alternative curve names -> names the model expects
ALIASES = {
    "DEPT": "DEPTH_MD", "DEPTH": "DEPTH_MD", "MD": "DEPTH_MD",
    "DT": "DTC", "DTCO": "DTC", "AC": "DTC",
    "ILD": "RDEP", "LLD": "RDEP", "RT": "RDEP", "RESD": "RDEP",
    "RHOZ": "RHOB", "DEN": "RHOB",
    "NPOR": "NPHI", "TNPH": "NPHI", "CNL": "NPHI",
    "GAMMA": "GR", "GRC": "GR",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case names, map common aliases, make sure DEPTH_MD exists."""
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    rename = {c: ALIASES[c] for c in df.columns if c in ALIASES and ALIASES[c] not in df.columns}
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace([-999.25, -999.0, -9999.0], np.nan)
    if "DEPTH_MD" not in df.columns:
        df["DEPTH_MD"] = np.arange(len(df), dtype="float32") * 0.15
    return df


def feature_names():
    cols = BASE + ["NPHI_RHOB"]
    names = []
    for c in cols:
        names += [c, c + "_mean", c + "_std", c + "_diff"]
    return names


def build_features(df: pd.DataFrame, well_col: str = "WELL"):
    """Return (df_sorted_with_features, feature_list). Missing curves become NaN."""
    d = standardize_columns(df) if "DEPTH_MD" not in df.columns else df.copy()
    if well_col not in d.columns:
        d[well_col] = "well"
    for c in BASE:
        if c not in d.columns:
            d[c] = np.nan
    d = d.sort_values([well_col, "DEPTH_MD"]).reset_index(drop=True)
    d["RDEP"] = np.log10(d["RDEP"].clip(lower=0.01))
    d["NPHI_RHOB"] = d["NPHI"] - (2.65 - d["RHOB"]) / 1.65
    cols = BASE + ["NPHI_RHOB"]
    for c in cols:
        r = d.groupby(well_col)[c].rolling(WINDOW, center=True, min_periods=1)
        d[c + "_mean"] = r.mean().reset_index(level=0, drop=True).astype("float32")
        d[c + "_std"] = r.std().reset_index(level=0, drop=True).astype("float32")
        d[c + "_diff"] = d.groupby(well_col)[c].diff().fillna(0).astype("float32")
    return d, feature_names()


def load_model(model_dir="model"):
    from xgboost import XGBClassifier

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    model = XGBClassifier()
    model.load_model(str(model_dir / "xgb_lithology.json"))
    return model, meta


def predict_well(df: pd.DataFrame, model, meta):
    """Predict lithology for one well. Returns a dataframe with predictions."""
    d, feats = build_features(standardize_columns(df))
    feats = meta["features"]
    proba = model.predict_proba(d[feats].astype("float32"))
    classes = np.array(meta["classes"])
    pred_code = classes[proba.argmax(1)]
    out = d.copy()
    out["PRED_CODE"] = pred_code
    out["PRED_LITHOLOGY"] = [LITH_NAMES.get(int(c), str(c)) for c in pred_code]
    out["PRED_CONFIDENCE"] = proba.max(1)
    return out
