import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from PIL import Image

EMBED_MODE = os.environ.get("SNR_EMBED_MODE") == "1"
if not EMBED_MODE:
    st.set_page_config(page_title="SNR OME (segmentation.py)", layout="wide")
    st.title("SNR Calculator (OME-TIFF via segmentation.py)")
else:
    st.subheader("OME-TIFF Mode")
st.caption(
    "Upload OME-TIFF -> split channels -> run segmentation.py -> read cell_data.csv -> compute SNR."
)


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
    cap: float,
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

    b_n = max(1, int(np.floor(n * (bottom_pct / 100.0))))
    if n >= 2:
        b_n = min(b_n, n - 1)
    bottom = x_sorted[:b_n]
    bottom_mean = float(np.mean(bottom))

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


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _build_channel_tokens(channel_names: list[str]) -> tuple[list[str], dict[str, str]]:
    tokens = []
    raw_to_token: dict[str, str] = {}
    used = set()
    for raw in channel_names:
        base = _safe_name(raw).strip("._")
        if not base:
            base = "channel"
        token = base
        suffix = 2
        while token in used:
            token = f"{base}_{suffix}"
            suffix += 1
        used.add(token)
        tokens.append(token)
        if raw not in raw_to_token:
            raw_to_token[raw] = token
    return tokens, raw_to_token


def _read_channel_names_from_ome_xml(ome_xml: str, n_channels: int) -> list[str]:
    names = []
    if ome_xml:
        try:
            root = ET.fromstring(ome_xml)
            channels = root.findall(".//{*}Channel")
            for idx, c in enumerate(channels):
                nm = c.attrib.get("Name") or c.attrib.get("ID") or f"C{idx:02d}"
                names.append(str(nm))
        except Exception:
            names = []
    if len(names) < n_channels:
        names += [f"C{idx:02d}" for idx in range(len(names), n_channels)]
    return names[:n_channels]


def _to_cyx(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = axes.upper()
    if arr.ndim != len(axes):
        raise ValueError(f"Unexpected axes metadata: axes={axes}, ndim={arr.ndim}")

    selector = []
    kept_axes = []
    for i, ax in enumerate(axes):
        if ax in ("Y", "X", "C"):
            selector.append(slice(None))
            kept_axes.append(ax)
        else:
            selector.append(0)

    arr2 = arr[tuple(selector)]
    if "Y" not in kept_axes or "X" not in kept_axes:
        raise ValueError(f"Cannot find Y/X axes in {axes}")
    if "C" not in kept_axes:
        arr2 = np.expand_dims(arr2, axis=0)
        kept_axes = ["C"] + kept_axes

    permute = [kept_axes.index("C"), kept_axes.index("Y"), kept_axes.index("X")]
    return np.transpose(arr2, permute)


def load_ome_to_cyx(uploaded_file) -> tuple[np.ndarray, list[str]]:
    bio = io.BytesIO(uploaded_file.getvalue())
    with tifffile.TiffFile(bio) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = getattr(series, "axes", "")
        cyx = _to_cyx(arr, axes)
        channel_names = _read_channel_names_from_ome_xml(tif.ome_metadata, cyx.shape[0])
    return cyx.astype(np.float32), channel_names


def write_channel_tiffs(cyx: np.ndarray, channel_tokens: list[str], out_dir: str):
    for idx, marker_token in enumerate(channel_tokens):
        channel_path = os.path.join(out_dir, f"{marker_token}.tif")
        tifffile.imwrite(channel_path, np.asarray(cyx[idx]))


def _preview_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1:
        p1 = float(np.min(arr))
        p99 = float(np.max(arr) + 1e-6)
    out = np.clip((arr - p1) / (p99 - p1), 0, 1)
    return (out * 255).astype(np.uint8)


def run_segmentation_py(
    segmentation_py: str,
    work_dir: str,
    nuclei_marker: str,
    nuclei_diameter: int,
    nuclei_expansion: int,
    nuclei_definition: float,
    nuclei_closeness: float,
    nuclei_area_limit: int,
    membrane_marker: str,
    membrane_diameter: int,
    membrane_compactness: float,
    membrane_keep: str,
    measure_markers: list[str],
    pipex_max_resolution: int,
) -> tuple[int, str, str]:
    cmd = [
        sys.executable,
        segmentation_py,
        f"-data={work_dir}",
        f"-nuclei_marker={nuclei_marker}",
        f"-nuclei_diameter={int(nuclei_diameter)}",
        f"-nuclei_expansion={int(nuclei_expansion)}",
        f"-measure_markers={','.join(measure_markers)}",
    ]

    if nuclei_definition > 0:
        cmd.append(f"-nuclei_definition={float(nuclei_definition)}")
    if nuclei_closeness > 0:
        cmd.append(f"-nuclei_closeness={float(nuclei_closeness)}")
    if nuclei_area_limit > 0:
        cmd.append(f"-nuclei_area_limit={int(nuclei_area_limit)}")

    if membrane_marker and membrane_diameter > 0:
        cmd.append(f"-membrane_marker={membrane_marker}")
        cmd.append(f"-membrane_diameter={int(membrane_diameter)}")
        cmd.append(f"-membrane_compactness={float(membrane_compactness)}")
        cmd.append(f"-membrane_keep={membrane_keep}")

    env = os.environ.copy()
    env["PIPEX_MAX_RESOLUTION"] = str(int(pipex_max_resolution))
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    proc = subprocess.run(
        cmd,
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


uploaded = st.file_uploader(
    "Upload one or more OME-TIFF ROI files (each file = one ROI)",
    type=["ome.tif", "ome.tiff", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Upload OME-TIFF files to begin.")
    st.stop()

try:
    _, first_channels = load_ome_to_cyx(uploaded[0])
except Exception as exc:
    st.error(f"Failed to parse OME-TIFF: {uploaded[0].name}")
    st.exception(exc)
    st.stop()

with st.expander("segmentation.py settings", expanded=True):
    segmentation_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmentation.py")
    st.caption(f"Bundled segmentation script: `{segmentation_py}`")

    c2, c3, c4, c5 = st.columns(4)
    with c2:
        nuclei_marker = st.selectbox("nuclei_marker", options=first_channels, index=0)
    with c3:
        nuclei_diameter = st.number_input("nuclei_diameter", min_value=1, max_value=500, value=20, step=1)
    with c4:
        nuclei_expansion = st.number_input("nuclei_expansion", min_value=0, max_value=500, value=8, step=1)
    with c5:
        nuclei_area_limit = st.number_input("nuclei_area_limit (0=off)", min_value=0, max_value=5000000, value=0, step=100)

    c6, c7, c8 = st.columns(3)
    with c6:
        nuclei_definition = st.number_input("nuclei_definition (0=default)", min_value=0.0, max_value=0.999, value=0.75, step=0.01, format="%.3f")
    with c7:
        nuclei_closeness = st.number_input("nuclei_closeness (0=default)", min_value=0.0, max_value=0.999, value=0.10, step=0.01, format="%.3f")
    with c8:
        pipex_max_resolution = st.number_input("PIPEX_MAX_RESOLUTION", min_value=512, max_value=200000, value=30000, step=512)

    st.markdown("Membrane (optional)")
    cm1, cm2, cm3, cm4 = st.columns(4)
    membrane_candidates = ["(disabled)"] + first_channels
    with cm1:
        membrane_marker_ui = st.selectbox("membrane_marker", options=membrane_candidates, index=0)
    with cm2:
        membrane_diameter = st.number_input("membrane_diameter", min_value=0, max_value=500, value=0, step=1)
    with cm3:
        membrane_compactness = st.number_input("membrane_compactness", min_value=0.001, max_value=0.999, value=0.9, step=0.01, format="%.3f")
    with cm4:
        membrane_keep = st.selectbox("membrane_keep", options=["no", "yes"], index=0)

st.subheader("Measure markers (measure_markers)")
default_markers = list(first_channels)
measure_markers = st.multiselect("Select markers for per-cell intensity", options=first_channels, default=default_markers)
if not measure_markers:
    st.warning("measure_markers cannot be empty.")
    st.stop()

st.subheader("SNR settings")
col1, col2, col3, col4 = st.columns(4)
with col1:
    top_mode = st.selectbox("Signal definition", ["Top N cells", "Top % of cells"], index=0)
with col2:
    top_n = st.number_input("Top N (if Top N)", min_value=1, max_value=10000, value=20, step=1)
with col3:
    top_pct = st.number_input("Top % (if Top %)", min_value=0.1, max_value=50.0, value=20.0, step=0.5)
with col4:
    bottom_pct = st.number_input("Bottom % (noise)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)

c9, c10, c11 = st.columns(3)
with c9:
    threshold = st.number_input("Pass threshold", min_value=0.0, max_value=1e9, value=10.0, step=1.0)
with c10:
    eps = st.number_input("Pseudocount eps", min_value=0.0, max_value=1.0, value=1e-6, format="%.6f")
with c11:
    cap = st.number_input("Cap SNR (optional)", min_value=0.0, max_value=1e9, value=1e6, step=1000.0)

min_cells = st.number_input("Min cells per ROI", min_value=1, max_value=1_000_000, value=200, step=10)
log_scale = st.checkbox("Log-scale X axis", value=False)

run_clicked = st.button("Run segmentation.py + SNR")
cache_key = "snr_ome_cached_results"

if run_clicked:
    st.session_state.pop(cache_key, None)
    if not os.path.isfile(segmentation_py):
        st.error(f"segmentation.py not found: {segmentation_py}")
        st.stop()

    membrane_marker = "" if membrane_marker_ui == "(disabled)" else membrane_marker_ui
    per_roi_tables: dict[str, pd.DataFrame] = {}
    seg_qc_rows = []
    runtime_logs = []
    roi_review_images: dict[str, dict[str, np.ndarray]] = {}
    roi_export_files: dict[str, dict[str, bytes]] = {}
    workdirs_to_cleanup = []

    for idx, f in enumerate(uploaded, start=1):
        roi_name = f"{idx:02d}_{f.name}"
        tmpdir = tempfile.mkdtemp(prefix="snr_ome_seg_")
        workdirs_to_cleanup.append(tmpdir)

        try:
            cyx, channel_names = load_ome_to_cyx(f)
            channel_tokens, raw_to_token = _build_channel_tokens(channel_names)
            write_channel_tiffs(cyx=cyx, channel_tokens=channel_tokens, out_dir=tmpdir)

            if nuclei_marker not in channel_names:
                st.warning(f"Skipped {roi_name}: nuclei_marker '{nuclei_marker}' not present")
                continue

            active_measure_markers_raw = [m for m in measure_markers if m in channel_names]
            if not active_measure_markers_raw:
                st.warning(f"Skipped {roi_name}: no selected measure_markers available in this file")
                continue
            active_measure_markers = [raw_to_token[m] for m in active_measure_markers_raw]

            if membrane_marker and membrane_marker not in channel_names:
                st.warning(f"ROI {roi_name}: membrane_marker '{membrane_marker}' missing, disabling membrane path")
                membrane_marker_this = ""
                membrane_diameter_this = 0
            else:
                membrane_marker_this = raw_to_token[membrane_marker] if membrane_marker else ""
                membrane_diameter_this = int(membrane_diameter)

            rc, stdout, stderr = run_segmentation_py(
                segmentation_py=segmentation_py,
                work_dir=tmpdir,
                nuclei_marker=raw_to_token[nuclei_marker],
                nuclei_diameter=int(nuclei_diameter),
                nuclei_expansion=int(nuclei_expansion),
                nuclei_definition=float(nuclei_definition),
                nuclei_closeness=float(nuclei_closeness),
                nuclei_area_limit=int(nuclei_area_limit),
                membrane_marker=membrane_marker_this,
                membrane_diameter=membrane_diameter_this,
                membrane_compactness=float(membrane_compactness),
                membrane_keep=membrane_keep,
                measure_markers=active_measure_markers,
                pipex_max_resolution=int(pipex_max_resolution),
            )

            runtime_logs.append({
                "ROI": roi_name,
                "return_code": rc,
                "stdout_tail": "\n".join(stdout.splitlines()[-20:]) if stdout else "",
                "stderr_tail": "\n".join(stderr.splitlines()[-20:]) if stderr else "",
            })

            if rc != 0:
                st.warning(f"Skipped {roi_name}: segmentation.py failed (code={rc})")
                continue

            cell_csv = os.path.join(tmpdir, "analysis", "cell_data.csv")
            if not os.path.isfile(cell_csv):
                st.warning(f"Skipped {roi_name}: missing output analysis/cell_data.csv")
                continue

            cell_df = pd.read_csv(cell_csv)
            n_cells = len(cell_df)
            seg_qc_rows.append(
                {
                    "ROI": roi_name,
                    "N_cells": n_cells,
                    "Image_shape_CYX": f"{cyx.shape[0]}x{cyx.shape[1]}x{cyx.shape[2]}",
                    "Markers_used": ",".join(active_measure_markers_raw),
                }
            )

            if n_cells < int(min_cells):
                st.warning(f"Skipped ROI {roi_name}: only {n_cells} cells (< {int(min_cells)})")
                continue

            per_roi_tables[roi_name] = cell_df
            dapi_preview = _preview_uint8(cyx[channel_names.index(nuclei_marker)])
            analysis_dir = os.path.join(tmpdir, "analysis")
            show_path_png = os.path.join(analysis_dir, "segmentation_mask_show.png")
            show_path_jpg = os.path.join(analysis_dir, "segmentation_mask_show.jpg")
            show_path = show_path_png if os.path.isfile(show_path_png) else show_path_jpg
            show_preview = np.asarray(Image.open(show_path)) if os.path.isfile(show_path) else dapi_preview
            roi_review_images[roi_name] = {"dapi": dapi_preview, "show": show_preview}

            roi_export_files[roi_name] = {}
            for rel_name, abs_path in [
                ("segmentation_mask.tif", os.path.join(analysis_dir, "segmentation_mask.tif")),
                ("segmentation_binary_mask.tif", os.path.join(analysis_dir, "segmentation_binary_mask.tif")),
                ("segmentation_data.npy", os.path.join(analysis_dir, "segmentation_data.npy")),
                ("segmentation_mask_show.png", os.path.join(analysis_dir, "segmentation_mask_show.png")),
                ("segmentation_mask_show.jpg", os.path.join(analysis_dir, "segmentation_mask_show.jpg")),
                ("segmentation_boundaries.png", os.path.join(analysis_dir, "segmentation_boundaries.png")),
            ]:
                if os.path.isfile(abs_path):
                    with open(abs_path, "rb") as fh:
                        roi_export_files[roi_name][rel_name] = fh.read()

        except Exception as exc:
            st.warning(f"Skipped {roi_name}: {exc}")

    for d in workdirs_to_cleanup:
        shutil.rmtree(d, ignore_errors=True)

    if not per_roi_tables:
        st.error("No ROI passed segmentation and min-cells criteria.")
        if runtime_logs:
            st.subheader("Runtime logs (tail)")
            st.dataframe(pd.DataFrame(runtime_logs), use_container_width=True)
        st.stop()

    st.session_state[cache_key] = {
        "per_roi_tables": per_roi_tables,
        "seg_qc_df": pd.DataFrame(seg_qc_rows),
        "runtime_logs_df": pd.DataFrame(runtime_logs),
        "roi_review_images": roi_review_images,
        "roi_export_files": roi_export_files,
    }

cached = st.session_state.get(cache_key)
if cached is None:
    st.info("Click `Run segmentation.py + SNR` to generate results.")
    st.stop()

per_roi_tables = cached["per_roi_tables"]
seg_qc_df = cached["seg_qc_df"]
runtime_logs_df = cached["runtime_logs_df"]
roi_review_images = cached["roi_review_images"]
roi_export_files = cached["roi_export_files"]

st.subheader("Segmentation QC")
st.dataframe(seg_qc_df, use_container_width=True)

if not runtime_logs_df.empty:
    st.subheader("Runtime logs (tail)")
    st.dataframe(runtime_logs_df, use_container_width=True)

if roi_review_images:
    st.subheader("Segmentation Review")
    review_roi = st.selectbox("ROI for visual check", options=sorted(roi_review_images.keys()))
    rv = roi_review_images[review_roi]
    c1, c2 = st.columns(2)
    with c1:
        st.image(rv["dapi"], caption=f"{review_roi} - DAPI", use_container_width=True)
    with c2:
        st.image(rv["show"], caption=f"{review_roi} - segmentation_mask_show", use_container_width=True)

marker_sets = [set(candidate_marker_columns(df)) for df in per_roi_tables.values()]
markers_auto = sorted(set.intersection(*marker_sets)) if marker_sets else []
if not markers_auto:
    st.error("No marker columns detected after extraction.")
    st.stop()

markers = st.multiselect("Select marker columns", options=markers_auto, default=markers_auto)
if not markers:
    st.warning("No marker columns selected.")
    st.stop()

records = []
for roi, df in per_roi_tables.items():
    for m in markers:
        result = compute_snr_for_vector(
            df[m].values,
            top_mode=top_mode,
            top_n=int(top_n),
            top_pct=float(top_pct),
            bottom_pct=float(bottom_pct),
            eps=float(eps),
            cap=float(cap),
        )
        records.append(
            {
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
                "Denom_unstable": result["denom_unstable"],
                "Overlap_adjusted": result["overlap_adjusted"],
            }
        )

snr_df = pd.DataFrame(records)
if snr_df.empty:
    st.error("No SNR records were computed.")
    st.stop()

summary = (
    snr_df.groupby("Protein")["SNR"]
    .agg(median_snr="median", mean_snr="mean", min_snr="min", max_snr="max")
    .reset_index()
)
summary["Pass_median"] = summary["median_snr"] >= 3.0
summary["Good_median"] = summary["median_snr"] >= 10.0

st.subheader("SNR per ROI")
st.dataframe(snr_df, use_container_width=True)

st.subheader("Summary across ROIs")
st.dataframe(summary, use_container_width=True)

fig = make_barplot(summary, float(threshold), log_scale=log_scale)
st.subheader("Barplot (median SNR across ROIs)")
st.pyplot(fig)

snr_csv = snr_df.to_csv(index=False).encode("utf-8")
summary_csv = summary.to_csv(index=False).encode("utf-8")
seg_qc_csv = seg_qc_df.to_csv(index=False).encode("utf-8")
logs_csv = runtime_logs_df.to_csv(index=False).encode("utf-8")

img_buf = io.BytesIO()
fig.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
img_buf.seek(0)

zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("snr_per_roi.csv", snr_csv)
    z.writestr("snr_summary.csv", summary_csv)
    z.writestr("segmentation_qc.csv", seg_qc_csv)
    z.writestr("runtime_logs_tail.csv", logs_csv)
    z.writestr("snr_barplot.png", img_buf.getvalue())
    for roi, df in per_roi_tables.items():
        z.writestr(f"per_cell/{_safe_name(roi)}.csv", df.to_csv(index=False).encode("utf-8"))
        for rel_name, blob in roi_export_files.get(roi, {}).items():
            z.writestr(f"segmentation/{_safe_name(roi)}/{rel_name}", blob)
zip_buf.seek(0)

st.download_button(
    "Download results (zip)",
    data=zip_buf,
    file_name="snr_ome_results.zip",
    mime="application/zip",
)
