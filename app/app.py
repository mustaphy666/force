import io
import sys
from pathlib import Path

import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import (LITH_COLORS, LITH_NAMES, TARGET, load_model,  # noqa: E402
                      predict_well, standardize_columns)

st.set_page_config(page_title="Well log lithology predictor", layout="wide")


@st.cache_resource
def get_model():
    return load_model(ROOT / "model")


def read_upload(name: str, raw: bytes) -> pd.DataFrame:
    if name.lower().endswith(".las"):
        las = lasio.read(io.StringIO(raw.decode("utf-8", errors="ignore")))
        return las.df().reset_index()
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")


def log_tracks(res: pd.DataFrame, depth_min: float, depth_max: float):
    r = res[(res["DEPTH_MD"] >= depth_min) & (res["DEPTH_MD"] <= depth_max)]
    tracks = [("GR", "gAPI", False), ("RHOB", "g/cm3", False), ("NPHI", "v/v", False),
              ("DTC", "us/ft", False), ("RDEP", "log10 ohm.m", False)]
    has_true = TARGET in r.columns and r[TARGET].notna().any()
    ncols = len(tracks) + 1 + int(has_true)
    fig, axes = plt.subplots(1, ncols, figsize=(2 * ncols, 9), sharey=True)
    z = r["DEPTH_MD"].values
    for ax, (c, unit, _) in zip(axes, tracks):
        ax.plot(r[c].values, z, lw=0.7, color="#1f3a5f")
        ax.set_title(c, fontsize=10)
        ax.set_xlabel(unit, fontsize=8)
        ax.grid(alpha=0.3)

    names = list(LITH_COLORS.keys())
    cmap = ListedColormap([LITH_COLORS[n] for n in names])
    idx = {n: i for i, n in enumerate(names)}

    def litho_track(ax, labels, title):
        arr = np.array([[idx.get(n, np.nan)] for n in labels], dtype=float)
        ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=len(names) - 1,
                  extent=[0, 1, z.max(), z.min()], interpolation="nearest")
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)

    litho_track(axes[len(tracks)], r["PRED_LITHOLOGY"].values, "Predicted")
    if has_true:
        true_names = [LITH_NAMES.get(int(v), None) if pd.notna(v) else None for v in r[TARGET]]
        litho_track(axes[-1], true_names, "Labelled")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_ylim(z.max(), z.min())
    from matplotlib.patches import Patch
    used = sorted(set(r["PRED_LITHOLOGY"]))
    fig.legend(handles=[Patch(color=LITH_COLORS[n], label=n) for n in used if n in LITH_COLORS],
            loc="lower center", ncol=6, fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    #fig.tight_layout()
    return fig


st.title("Well log lithology predictor")
st.write(
    "Upload a well (LAS or CSV) and get a depth-by-depth lithology prediction from an "
    "XGBoost model trained on the FORCE 2020 North Sea dataset."
)

try:
    model, meta = get_model()
except Exception as e:
    st.error(f"Model files not found in /model. Run `python src/train_final.py <las_folder>` first. ({e})")
    st.stop()

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Well file", type=["las", "csv"])
    sample = ROOT / "data" / "sample"
    sample_files = sorted(sample.glob("*.las")) + sorted(sample.glob("*.csv"))
    use_sample = False
    if sample_files:
        use_sample = st.button("Use bundled sample well")
    st.caption("Expected curves: GR, RHOB, NPHI, DTC, RDEP, PEF. Missing curves are allowed; "
               "accuracy drops when key logs are absent.")

raw, name = None, None
if up is not None:
    raw, name = up.getvalue(), up.name
    st.session_state["sample"] = False
elif use_sample or st.session_state.get("sample"):
    st.session_state["sample"] = True
    raw, name = sample_files[0].read_bytes(), sample_files[0].name

if raw is None:
    st.info("Upload a well file in the sidebar to begin.")
else:
    df = standardize_columns(read_upload(name, raw))
    curves = ["GR", "RHOB", "NPHI", "DTC", "RDEP", "PEF"]
    present = [c for c in curves if c in df.columns and df[c].notna().any()]
    missing = [c for c in curves if c not in present]
    st.subheader(name)
    st.write(f"{len(df):,} samples, depth {df['DEPTH_MD'].min():.0f} to {df['DEPTH_MD'].max():.0f} m. "
             f"Curves found: {', '.join(present) or 'none'}.")
    if "GR" not in present:
        st.error("No gamma ray (GR) curve found, so a prediction would not be reliable.")
        st.stop()
    if missing:
        st.warning(f"Missing curves: {', '.join(missing)}. The model treats them as blank.")

    with st.spinner("Predicting lithology..."):
        res = predict_well(df, model, meta)

    zmin, zmax = float(res["DEPTH_MD"].min()), float(res["DEPTH_MD"].max())
    d0, d1 = st.slider("Depth interval (m)", zmin, zmax, (zmin, min(zmax, zmin + 600.0)))
    left, right = st.columns([3, 1])
    with left:
        st.pyplot(log_tracks(res, d0, d1))
    with right:
        st.markdown("**Predicted lithology share**")
        share = res["PRED_LITHOLOGY"].value_counts(normalize=True).mul(100).round(1)
        st.dataframe(share.rename("%"))
        if TARGET in res.columns and res[TARGET].notna().any():
            lab = res.dropna(subset=[TARGET])
            true_n = lab[TARGET].astype(int).map(LITH_NAMES)
            st.markdown("**Agreement with labels in file**")
            st.metric("Accuracy", f"{accuracy_score(true_n, lab['PRED_LITHOLOGY']):.2f}")
            st.metric("Macro-F1", f"{f1_score(true_n, lab['PRED_LITHOLOGY'], average='macro'):.2f}")

    st.download_button("Download predictions (CSV)",
                       res[["DEPTH_MD", "PRED_LITHOLOGY", "PRED_CONFIDENCE"]].to_csv(index=False),
                       file_name=f"{Path(name).stem}_lithology.csv", mime="text/csv")

with st.expander("About the model and its limits"):
    cv = meta.get("cv")
    st.write(
        f"Trained on {meta.get('n_wells', 'many')} labelled wells from the FORCE 2020 dataset "
        "using rolling-window log features, confidence-weighted samples and partial class balancing."
    )
    if cv:
        st.write(
        f"5-fold cross-validation split by well: "
        f"accuracy {cv['fold_accuracy_mean']:.3f} "
        f"(±{cv['fold_accuracy_std']:.3f}), "
        f"macro-F1 {cv['fold_macro_f1_mean']:.3f} "
        f"(±{cv['fold_macro_f1_std']:.3f})."
    )
    st.write(
        "Reliable for shale, sandstone, coal and limestone. Carbonates are often confused with each "
        "other (chalk and dolomite are mostly predicted as limestone) and transitional units such as "
        "marl, tuff and sand/shale tend to be absorbed by shale. The model learns North Sea log "
        "responses and should not be assumed to transfer to other basins."
    )
    cm = ROOT / "confusion_matrix.png"
    if cm.exists():
        st.image(str(cm), caption="Confusion matrix, 5-fold CV split by well (row-normalised)")
