"""
FORCE 2020 Lithology Classification
Final Model Training and Evaluation Pipeline.

Usage:

    python src/train_final.py "C:/path/to/Force_2020"

    python src/train_final.py "C:/path/to/Force_2020" --cv
"""

import argparse
import json
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from features import (
    BASE,
    CONF,
    TARGET,
    LITH_NAMES,
    build_features,
    standardize_columns
)


# ============================================================
# CONFIGURATION
# ============================================================

KEEP = [TARGET, CONF, "DEPTH_MD"] + BASE

MIN_WELLS_PER_CLASS = 5

CLASS_WEIGHT_POWER = 0.5

STEP = 2

N_SPLITS = 5

RANDOM_STATE = 42

PARAMS = dict(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss"
)

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD LAS FILES
# ============================================================

def load_las_folder(root: Path):

    frames = []

    files = sorted(root.rglob("*.las"))

    if not files:
        raise FileNotFoundError(
            f"No LAS files found in {root}"
        )

    for f in files:

        try:

            las = lasio.read(f)

            df = las.df().reset_index()

            df = standardize_columns(df)

            if TARGET not in df.columns:
                continue

            available = [
                c for c in KEEP
                if c in df.columns
            ]

            df = df[available].copy()

            df = df.dropna(subset=[TARGET])

            if df.empty:
                continue

            df = df.astype("float32")

            df["WELL"] = f.stem

            frames.append(df)

        except Exception as e:

            print(f"Skipped {f.name}: {e}")

    if not frames:

        raise ValueError(
            "No labelled LAS files could be loaded."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    print("\nDATA LOADING")
    print("-" * 50)

    print(
        f"Labelled wells: {data['WELL'].nunique()}"
    )

    print(
        f"Total observations: {len(data):,}"
    )

    return data


# ============================================================
# 2. MODEL TRAINING FUNCTION
# ============================================================

def fit_model(X, y, confidence_weights):

    le = LabelEncoder()

    y_encoded = le.fit_transform(y)

    class_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_encoded
    )

    sample_weights = (
        class_weights ** CLASS_WEIGHT_POWER
    ) * confidence_weights

    model = XGBClassifier(**PARAMS)

    model.fit(
        X,
        y_encoded,
        sample_weight=sample_weights
    )

    return model, le


# ============================================================
# 3. CONFIDENCE WEIGHTS
# ============================================================

def get_confidence_weights(data):

    weights = data[CONF].map({
        1: 1.0,
        2: 0.6,
        3: 0.3
    }).fillna(0.5)

    return weights.to_numpy(dtype=np.float32)


# ============================================================
# INVESTIGATION 1: CLASS AND CONFIDENCE DISTRIBUTION
# ============================================================

def investigate_class_distribution(df):

    print("\nCLASS DISTRIBUTION BY WELL")
    print("=" * 60)

    summary = (
        df.groupby(TARGET)
        .agg(
            observations=(TARGET, "size"),
            wells=("WELL", "nunique")
        )
        .reset_index()
    )

    summary["lithology"] = summary[TARGET].map(LITH_NAMES)

    print(
        summary[
            ["lithology", "observations", "wells"]
        ].to_string(index=False)
    )

    print("\nCONFIDENCE DISTRIBUTION")

    confidence = pd.crosstab(
        df[TARGET].map(LITH_NAMES),
        df[CONF],
        normalize="index"
    )

    print(confidence.round(3))

    summary.to_csv(
        MODEL_DIR / "class_distribution.csv",
        index=False
    )

    confidence.to_csv(
        MODEL_DIR / "confidence_distribution.csv"
    )

    print("\nDistribution reports saved.")

# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_true, y_pred, classes):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes
    )

    # Normalize each row.
    cm_normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=classes,
        normalize="true"
    )

    class_names = [
        LITH_NAMES.get(int(c), str(c))
        for c in classes
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(19, 8)
    )

    # Raw counts
    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp1.plot(
        ax=axes[0],
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format="d"
    )

    axes[0].set_title(
        "Confusion Matrix - Counts"
    )

    # Normalized matrix
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=class_names
    )

    disp2.plot(
        ax=axes[1],
        cmap="Blues",
        xticks_rotation=45,
        colorbar=True,
        values_format=".2f"
    )

    axes[1].set_title(
        "Confusion Matrix - Normalized"
    )

    plt.tight_layout()

    output = MODEL_DIR / "confusion_matrix.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nConfusion matrix saved: {output}"
    )

    # Save numerical matrices too.
    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(
        MODEL_DIR / "confusion_matrix_counts.csv"
    )

    pd.DataFrame(
        cm_normalized,
        index=class_names,
        columns=class_names
    ).to_csv(
        MODEL_DIR / "confusion_matrix_normalized.csv"
    )

def investigate_class_distribution(df):

    print("\nCLASS DISTRIBUTION BY WELL")
    print("=" * 60)

    summary = (
        df.groupby(TARGET)
        .agg(
            observations=(TARGET, "size"),
            wells=("WELL", "nunique")
        )
        .reset_index()
    )

    summary["lithology"] = summary[TARGET].map(LITH_NAMES)

    print(
        summary[
            ["lithology", "observations", "wells"]
        ].to_string(index=False)
    )

    print("\nCONFIDENCE DISTRIBUTION")

    confidence = pd.crosstab(
        df[TARGET].map(LITH_NAMES),
        df[CONF],
        normalize="index"
    )

    print(confidence.round(3))

    print("\nOBSERVATIONS PER WELL")

    class_well = pd.crosstab(
        df["WELL"],
        df[TARGET].map(LITH_NAMES)
    )

    print(
        class_well.sum()
        .sort_values(ascending=False)
        .to_string()
    )

    summary.to_csv(
        MODEL_DIR / "class_distribution.csv",
        index=False
    )

    confidence.to_csv(
        MODEL_DIR / "confidence_distribution.csv"
    )
# ============================================================
# 5. CROSS VALIDATION
# ============================================================

def run_cross_validation(X, y, groups, weights):

    print("\nCROSS VALIDATION")
    print("-" * 50)

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    classes = np.sort(np.unique(y))

    # Store predictions from unseen wells.
    oof_predictions = np.empty(
        len(y),
        dtype=np.int64
    )

    f1_scores = []
    accuracy_scores = []

    precision_scores = []
    recall_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=groups),
        start=1
    ):

        print(f"\nFOLD {fold}")
        print("-" * 35)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        w_train = weights[train_idx]

        # Fit only on training wells.
        model, le = fit_model(
            X_train,
            y_train,
            w_train
        )

        predictions_encoded = model.predict(
            X_test
        )

        predictions = le.inverse_transform(
            predictions_encoded.astype(int)
        )

        oof_predictions[test_idx] = predictions

        acc = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        f1_scores.append(f1)

        accuracy_scores.append(acc)

        precision_scores.append(precision)

        recall_scores.append(recall)

        print(f"Accuracy:  {acc:.4f}")

        print(f"Macro F1:  {f1:.4f}")

        print(f"Precision: {precision:.4f}")

        print(f"Recall:    {recall:.4f}")

    # ========================================================
    # AGGREGATED OUT-OF-FOLD EVALUATION
    # ========================================================

    print("\n\nAGGREGATED OUT-OF-FOLD RESULTS")
    print("=" * 60)

    print(
        classification_report(
            y,
            oof_predictions,
            labels=classes,
            target_names=[
                LITH_NAMES.get(int(c), str(c))
                for c in classes
            ],
            zero_division=0
        )
    )

    overall_accuracy = accuracy_score(
        y,
        oof_predictions
    )

    overall_f1 = f1_score(
        y,
        oof_predictions,
        average="macro",
        zero_division=0
    )

    print(
        f"Overall OOF Accuracy: {overall_accuracy:.4f}"
    )

    print(
        f"Overall OOF Macro F1: {overall_f1:.4f}"
    )

    print("\nFOLD SUMMARY")

    print(
        f"Mean accuracy: "
        f"{np.mean(accuracy_scores):.4f} "
        f"+/- {np.std(accuracy_scores):.4f}"
    )

    print(
        f"Mean macro F1: "
        f"{np.mean(f1_scores):.4f} "
        f"+/- {np.std(f1_scores):.4f}"
    )

    # Save classification report.
    report = classification_report(
        y,
        oof_predictions,
        labels=classes,
        target_names=[
            LITH_NAMES.get(int(c), str(c))
            for c in classes
        ],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        MODEL_DIR / "classification_report.csv"
    )

    # Save fold metrics.
    fold_results = pd.DataFrame({
        "fold": range(1, N_SPLITS + 1),
        "accuracy": accuracy_scores,
        "macro_f1": f1_scores,
        "precision": precision_scores,
        "recall": recall_scores
    })

    fold_results.to_csv(
        MODEL_DIR / "fold_metrics.csv",
        index=False
    )

    # Generate confusion matrix using OOF predictions.
    save_confusion_matrix(
        y,
        oof_predictions,
        classes
    )

    # Save OOF predictions.
    oof_df = pd.DataFrame({
        "WELL": groups,
        "TRUE_LITHOLOGY": y,
        "PREDICTED_LITHOLOGY": oof_predictions
    })

    oof_df.to_csv(
        MODEL_DIR / "oof_predictions.csv",
        index=False
    )

    results = {
        "oof_accuracy": float(overall_accuracy),
        "oof_macro_f1": float(overall_f1),

        "fold_accuracy_mean": float(
            np.mean(accuracy_scores)
        ),

        "fold_accuracy_std": float(
            np.std(accuracy_scores)
        ),

        "fold_macro_f1_mean": float(
            np.mean(f1_scores)
        ),

        "fold_macro_f1_std": float(
            np.std(f1_scores)
        ),

        "fold_precision_mean": float(
            np.mean(precision_scores)
        ),

        "fold_recall_mean": float(
            np.mean(recall_scores)
        )
    }

    return results


# ============================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "las_folder",
        help="Folder containing FORCE 2020 LAS files"
    )

    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run five-fold cross-validation"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_las_folder(
        Path(args.las_folder)
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nFEATURE ENGINEERING")
    print("-" * 50)

    d, features = build_features(df)

    print(
        f"Number of engineered features: {len(features)}"
    )

    # --------------------------------------------------------
    # Retain sufficiently represented classes
    # --------------------------------------------------------

    class_well_counts = (
        d.groupby(TARGET)["WELL"].nunique()
    )

    keep = class_well_counts[
        class_well_counts >= MIN_WELLS_PER_CLASS
    ].index

    print("\nRETAINED CLASSES")

    for code in sorted(keep):

        print(
            f"{int(code)} - "
            f"{LITH_NAMES.get(int(code), 'Unknown')}: "
            f"{class_well_counts.loc[code]} wells"
        )

    d = d[d[TARGET].isin(keep)].copy()
    # Investigate class and confidence distribution
# BEFORE downsampling

    investigate_class_distribution(d)

    # --------------------------------------------------------
    # Downsample separately within each well
    # --------------------------------------------------------

    d = d[
        d.groupby("WELL").cumcount() % STEP == 0
    ].reset_index(drop=True)

    print("\nAFTER FILTERING")

    print(
        f"Wells: {d['WELL'].nunique()}"
    )

    print(
        f"Rows: {len(d):,}"
    )

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X = d[features].astype("float32")

    y = d[TARGET].to_numpy(dtype=np.int64)

    groups = d["WELL"].to_numpy()

    weights = get_confidence_weights(d)

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nCLASS DISTRIBUTION")

    print(
        d[TARGET]
        .map(LITH_NAMES)
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Cross validation
    # --------------------------------------------------------

    cv_results = None

    if args.cv:

        cv_results = run_cross_validation(
            X,
            y,
            groups,
            weights
        )

    # --------------------------------------------------------
    # Train final model on all eligible wells
    # --------------------------------------------------------

    print("\nTRAINING FINAL MODEL")
    print("-" * 50)

    model, le = fit_model(
        X,
        y,
        weights
    )
    
# ========================================================
# INVESTIGATION 2: FEATURE IMPORTANCE
# ========================================================

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\nTOP 20 FEATURES")
    print("=" * 50)

    print(
        importance.head(20).to_string(index=False)
    )

    # Save feature importance table

    importance.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False
    )

    # Plot feature importance

    plt.figure(figsize=(10, 8))

    top = importance.head(20).sort_values(
        "importance"
    )

    plt.barh(
        top["feature"],
        top["importance"]
    )

    plt.xlabel("XGBoost Feature Importance")

    plt.title(
        "Top 20 Features for Lithology Classification"
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "feature_importance.png",
        dpi=300
    )

    plt.close()

    print("\nFeature importance files saved.")


    model.save_model(
        str(MODEL_DIR / "xgb_lithology.json")
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "features": features,

        "classes": [
            int(c) for c in le.classes_
        ],

        "class_names": {
            str(int(c)): LITH_NAMES.get(
                int(c),
                "Unknown"
            )
            for c in le.classes_
        },

        "n_wells": int(
            d["WELL"].nunique()
        ),

        "n_observations": int(
            len(d)
        ),

        "n_features": len(features),

        "cv": cv_results,

        "parameters": PARAMS,

        "sampling_step": STEP,

        "minimum_wells_per_class": MIN_WELLS_PER_CLASS

    }

    with open(
        MODEL_DIR / "meta.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\nFINAL MODEL SAVED")

    print(
        f"Model directory: {MODEL_DIR}"
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":

    main()