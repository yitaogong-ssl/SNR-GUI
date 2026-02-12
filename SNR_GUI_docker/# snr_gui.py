# snr_gui.py
import io
import re
import zipfile
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="SNR Calculator (CSV)", layout="wide")

st.title("SNR Calculator (Multiple CSV ROIs)")
st.caption(
    "Compute SNR per marker from per-cell intensity CSVs (one CSV per ROI). "
    "Default metric: mean(top 20 cells) / mean(bottom 10%)."
)

# ---------- Helpers ----------
EXCLUDE_PATTERNS = [
    r"_otsu", r"_ratio", r"_local_", r"^cell_", r"^memref$", r"^size$", r"^x$", r"^y$",
    r"^solidity$", r"^eccentricity$", r"^cell_id$"
]

def is_excluded(col: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, col, flags=re.IGNORECASE):
            return True
    return False

def candidate_marker_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if is_excluded(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def compute_snr_for_vector(
    x: np.ndarray,
    top_mode: str,
    top_n: int,
    top_pct: float,
    bottom_pct: float,
    eps: float,
    cap: float
) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "snr": np.nan,
            "top_mean": np.nan,
            "bottom_mean": np.nan,
            "top_n": 0,
            "bottom_n": 0,
            "overlap_adjusted": False,
            "denom_unstable": True,
        }

    x_sorted = np.sort(x)
    n = x_sorted.size

    # bottom
    b_n = max(1, int(np.floor(n * (bottom_pct / 100.0))))
    # Keep at least one cell for the signal pool when n >= 2.
    if n >= 2:
        b_n = min(b_n, n - 1)
    bottom = x_sorted[:b_n]
    bottom_mean = float(np.mean(bottom))

    # top
    if top_mode == "Top N cells":
        t_n = min(max(1, top_n), n)
    else:
        t_n = max(1, int(np.floor(n * (top_pct / 100.0))))
        t_n = min(t_n, n)

    overlap_adjusted = False
    max_top_without_overlap = max(1, n - b_n)
    if t_n > max_top_without_overlap:
        t_n = max_top_without_overlap
        overlap_adjusted = True

    top = x_sorted[-t_n:]
    top_mean = float(np.mean(top))

    denom = bottom_mean + eps
    snr = top_mean / denom
    if cap is not None and np.isfinite(cap):
        snr = min(snr, cap)
    denom_unstable = (not np.isfinite(bottom_mean)) or (bottom_mean <= (10.0 * eps))
    return {
        "snr": snr,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "top_n": t_n,
        "bottom_n": b_n,
        "overlap_adjusted": overlap_adjusted,
        "denom_unstable": denom_unstable,
    }

def make_barplot(
    summary_df: pd.DataFrame,
    threshold: float,
    log_scale: bool,
    log_floor: float = 1e-6,
    pass_threshold: float = 3.0,
    good_threshold: float = 10.0,
):
    # summary_df columns: Protein, median_snr
    df = summary_df.sort_values("median_snr", ascending=True)
    plot_vals = df["median_snr"].copy()
    if log_scale:
        plot_vals = plot_vals.clip(lower=log_floor)
    fig = plt.figure(figsize=(10, max(3, 0.35 * len(df))))
    plt.barh(df["Protein"], plot_vals)
    if log_scale:
        plt.xscale("log")
        thresh_plot = threshold if threshold > 0 else log_floor
        plt.xlabel("Median SNR across ROIs (log scale)")
    else:
        thresh_plot = threshold
        plt.xlabel("Median SNR across ROIs")
    plt.axvline(thresh_plot, linestyle="--")
    pass_plot = pass_threshold if (not log_scale or pass_threshold > 0) else log_floor
    good_plot = good_threshold if (not log_scale or good_threshold > 0) else log_floor
    plt.axvline(pass_plot, linestyle="--", color="tab:green", label="Pass (SNR >= 3)")
    plt.axvline(good_plot, linestyle="-.", color="tab:blue", label="Good (SNR >= 10)")
    plt.legend(loc="best")
    plt.ylabel("Marker")
    plt.tight_layout()
    return fig



def safe_read_csv(uploaded_file):
    # uploaded_file is a Streamlit UploadedFile
    if uploaded_file is None:
        return None
    if hasattr(uploaded_file, "size") and uploaded_file.size == 0:
        return None
    try:
        # Ensure we read from the start each time.
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except EmptyDataError:
        return None

# ---------- UI ----------
uploaded = st.file_uploader(
    "Upload one or more CSV files (each CSV = one ROI)",
    type=["csv"],
    accept_multiple_files=True
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    top_mode = st.selectbox("Signal definition", ["Top N cells", "Top % of cells"], index=0)
with col2:
    top_n = st.number_input("Top N (if Top N)", min_value=1, max_value=10000, value=20, step=1)
with col3:
    top_pct = st.number_input("Top % (if Top %)", min_value=0.1, max_value=50.0, value=20.0, step=0.5)
with col4:
    bottom_pct = st.number_input("Bottom % (noise)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)

c5, c6, c7 = st.columns(3)
with c5:
    threshold = st.number_input("Pass threshold", min_value=0.0, max_value=1e9, value=10.0, step=1.0)
with c6:
    eps = st.number_input("Pseudocount ε (avoid inf)", min_value=0.0, max_value=1.0, value=1e-6, format="%.6f")
with c7:
    cap = st.number_input("Cap SNR (optional)", min_value=0.0, max_value=1e9, value=1e6, step=1000.0)

c8, c9, c10 = st.columns(3)
with c8:
    min_cells = st.number_input("Min cells per ROI", min_value=1, max_value=1_000_000, value=200, step=10)
with c9:
    tiny_threshold = st.number_input(
        "Near-zero relative threshold (top/median)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.4f"
    )
with c10:
    log_scale = st.checkbox("Log-scale barplot", value=False)

if uploaded:
    # read all valid CSVs once
    dfs = {}
    for idx, f in enumerate(uploaded, start=1):
        df_try = safe_read_csv(f)
        if df_try is None or df_try.empty:
            st.warning(f"Skipped empty/invalid CSV: {f.name}")
            continue
        # Keep each upload as a separate ROI even when filenames are identical.
        roi_name = f"{idx:02d}_{f.name}"
        dfs[roi_name] = df_try

    if not dfs:
        st.error("All uploaded CSVs are empty or invalid.")
        st.stop()

    st.subheader("Marker options")
    marker_mode = st.radio(
        "Marker set across ROIs",
        ["Intersection (common to all)", "Union (all)"],
        index=0,
        horizontal=True
    )

    marker_sets = []
    for _, df in dfs.items():
        marker_sets.append(set(candidate_marker_columns(df)))

    if marker_mode.startswith("Intersection"):
        markers_auto = sorted(set.intersection(*marker_sets)) if marker_sets else []
    else:
        markers_auto = sorted(set.union(*marker_sets)) if marker_sets else []

    if not markers_auto:
        st.error("No marker columns detected with the current filters.")
        st.stop()

    st.subheader("Select marker columns")
    markers = st.multiselect(
        "Auto-detected numeric columns (excluding *_otsu*, *_ratio*, *_local_* etc.)",
        options=markers_auto,
        default=markers_auto
    )

    if not markers:
        st.warning("No marker columns selected.")
        st.stop()

    if st.button("Compute SNR"):
        records = []
        roi_cell_counts = {}
        near_zero_markers = set()
        marker_missing_rois = {m: [] for m in markers}

        for roi, df in dfs.items():
            if len(df) < int(min_cells):
                st.warning(f"Skipped ROI {roi}: only {len(df)} cells (< {int(min_cells)})")
                continue
            roi_cell_counts[roi] = len(df)

            for m in markers:
                if m not in df.columns:
                    marker_missing_rois[m].append(roi)
                    continue
                result = compute_snr_for_vector(
                    df[m].values,
                    top_mode=top_mode,
                    top_n=int(top_n),
                    top_pct=float(top_pct),
                    bottom_pct=float(bottom_pct),
                    eps=float(eps),
                    cap=float(cap)
                )
                marker_vals = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
                marker_vals = marker_vals[np.isfinite(marker_vals)]
                marker_median = float(np.median(marker_vals)) if marker_vals.size > 0 else np.nan
                top_vs_median = (
                    result["top_mean"] / (marker_median + float(eps))
                    if np.isfinite(result["top_mean"]) and np.isfinite(marker_median)
                    else np.nan
                )
                if np.isfinite(top_vs_median) and top_vs_median < float(tiny_threshold):
                    near_zero_markers.add(m)
                records.append({
                    "ROI": roi,
                    "Protein": m,
                    "SNR": result["snr"],
                    "Pass": (result["snr"] >= 3.0) if np.isfinite(result["snr"]) else False,
                    "Good": (result["snr"] >= 10.0) if np.isfinite(result["snr"]) else False,
                    "N_cells": len(df),
                    "Top_mean": result["top_mean"],
                    "Bottom_mean": result["bottom_mean"],
                    "Top_n": result["top_n"],
                    "Bottom_n": result["bottom_n"],
                    "Top_vs_median": top_vs_median,
                    "Denom_unstable": result["denom_unstable"],
                    "Overlap_adjusted": result["overlap_adjusted"],
                })

        snr_df = pd.DataFrame(records)
        if snr_df.empty:
            st.error("No SNR records were computed. Check that the selected marker columns exist in the uploaded CSVs.")
            st.stop()

        if near_zero_markers:
            st.warning(
                "Markers near-zero in at least one ROI by top/median ratio (threshold = "
                f"{float(tiny_threshold)}): {', '.join(sorted(near_zero_markers))}"
            )
        if not snr_df.empty and bool(snr_df["Denom_unstable"].any()):
            unstable_markers = sorted(snr_df.loc[snr_df["Denom_unstable"], "Protein"].unique().tolist())
            st.warning(
                "Unstable denominator detected (bottom_mean <= 10*eps) for markers: "
                + ", ".join(unstable_markers)
            )
        missing_messages = []
        for m, rois in marker_missing_rois.items():
            if rois:
                missing_messages.append(f"{m}: missing in {len(rois)} ROI(s)")
        if missing_messages:
            st.warning("Union mode missing markers by ROI: " + " | ".join(missing_messages))

        st.subheader("SNR per ROI")
        st.dataframe(snr_df, use_container_width=True)

        # summary across ROIs
        n_roi_processed = len(roi_cell_counts)
        summary = (
            snr_df.groupby("Protein")["SNR"]
            .agg(median_snr="median", mean_snr="mean", min_snr="min", max_snr="max")
            .reset_index()
        )
        n_used = (
            snr_df.groupby("Protein")["ROI"]
            .nunique()
            .rename("n_roi_used")
            .reset_index()
        )
        summary = summary.merge(n_used, on="Protein", how="left")
        summary["n_roi_total"] = n_roi_processed
        summary["roi_coverage"] = summary["n_roi_used"] / summary["n_roi_total"].replace(0, np.nan)
        summary["Pass_median"] = summary["median_snr"] >= 3.0
        summary["Good_median"] = summary["median_snr"] >= 10.0

        st.subheader("Summary across ROIs")
        st.dataframe(summary, use_container_width=True)

        fig = make_barplot(summary, float(threshold), log_scale=log_scale)
        st.subheader("Barplot (median SNR across ROIs)")
        st.pyplot(fig)

        # Package outputs
        csv_bytes = snr_df.to_csv(index=False).encode("utf-8")
        sum_bytes = summary.to_csv(index=False).encode("utf-8")

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
        img_buf.seek(0)

        report_text = (
            f"SNR definition: {top_mode} (TopN={top_n}, Top%={top_pct}) / Bottom%={bottom_pct}\n"
            f"Threshold: {threshold}\n"
            f"Pseudocount eps: {eps}, Cap: {cap}\n"
            f"Min cells per ROI: {int(min_cells)}\n"
            f"Marker set mode: {marker_mode}\n"
            f"Near-zero threshold (top/median): {tiny_threshold}\n"
            f"Log-scale barplot: {log_scale}\n"
            f"ROIs uploaded: {len(dfs)}\n"
            f"ROIs processed (after min-cells filter): {n_roi_processed}\n"
            f"Markers: {', '.join(markers)}\n"
        ).encode("utf-8")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("snr_per_roi.csv", csv_bytes)
            z.writestr("snr_summary.csv", sum_bytes)
            z.writestr("snr_barplot.png", img_buf.getvalue())
            z.writestr("report_snippet.txt", report_text)
        zip_buf.seek(0)

        st.download_button(
            "Download results (zip)",
            data=zip_buf,
            file_name="snr_results.zip",
            mime="application/zip"
        )

else:
    st.info("Upload CSV files to begin.")
