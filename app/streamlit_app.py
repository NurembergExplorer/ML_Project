import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
import numpy as np
import io
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from shapely.geometry import shape, box
from streamlit_folium import st_folium
from folium.plugins import Draw

st.set_page_config(layout="wide", page_title="Nuremberg Land-Cover Explorer")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        height: 100vh;
        overflow: hidden;
    }
    .block-container {
        padding-top: 3.4rem;
        padding-bottom: 0.4rem;
        max-width: 100%;
    }
    [data-testid="stHeader"] {
        background: rgba(238, 252, 231, 0.95);
        backdrop-filter: blur(4px);
    }
    .stApp {
        background: radial-gradient(circle at 15% 15%, #f3ffe8 0%, #e9fbdc 40%, #ddf4cd 100%);
    }
    [data-testid="stSidebar"] {
        background-color: #e1f5d1;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    .panel-card {
        border: 1px solid #d9e8d3;
        border-radius: 14px;
        background: linear-gradient(180deg, #fbfff8 0%, #f5fdf1 100%);
        padding: 10px;
        margin-bottom: 8px;
        box-shadow: 0 8px 24px rgba(53, 102, 45, 0.07);
    }
    .hero {
        border: 1px solid #dcefd4;
        border-radius: 16px;
        background: linear-gradient(120deg, #f8fff6 0%, #edf9e7 45%, #f6fff1 100%);
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.1px;
        color: #244b1f;
    }
    .hero-sub {
        font-size: 0.88rem;
        color: #3f5f3b;
        margin-top: 2px;
    }
    .insight-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .pop-chip-wrap {
        display:flex;
        gap:8px;
        flex-wrap:wrap;
        margin-top:4px;
    }
    .pop-chip {
        border:1px solid #d9e8d3;
        border-radius:10px;
        background:#f8fff5;
        padding:6px 10px;
        min-width:86px;
        text-align:center;
    }
    .pop-year {
        font-size:0.72rem;
        color:#557052;
    }
    .pop-val {
        font-size:0.9rem;
        font-weight:700;
        color:#23401f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _normalize_feature(drawing: dict | None) -> dict | None:
    if not isinstance(drawing, dict):
        return None

    if "geometry" in drawing and isinstance(drawing["geometry"], dict):
        geometry = drawing["geometry"]
    elif drawing.get("type") in {"Polygon", "MultiPolygon", "LineString"}:
        geometry = drawing
    else:
        return None

    if geometry.get("type") not in {"Polygon", "MultiPolygon", "LineString"}:
        return None

    return {"type": "Feature", "geometry": geometry, "properties": drawing.get("properties", {})}


def _geometry_has_enough_points(geometry: dict) -> bool:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return False

    if gtype == "Polygon":
        if len(coords) == 0 or len(coords[0]) < 3:
            return False
    elif gtype == "MultiPolygon":
        if len(coords) == 0 or len(coords[0]) == 0 or len(coords[0][0]) < 3:
            return False
    else:
        return False

    return True


def _ensure_closed_ring(ring: list) -> list:
    if not ring:
        return ring
    if ring[0] != ring[-1]:
        return ring + [ring[0]]
    return ring


def _sanitize_polygon_geometry(geometry: dict) -> dict | None:
    if not isinstance(geometry, dict):
        return None

    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    if gtype == "Polygon" and isinstance(coords, list) and len(coords) > 0:
        fixed_rings = []
        for ring in coords:
            if not isinstance(ring, list) or len(ring) < 3:
                continue
            fixed_rings.append(_ensure_closed_ring(ring))
        if not fixed_rings:
            return None
        return {"type": "Polygon", "coordinates": fixed_rings}

    if gtype == "MultiPolygon" and isinstance(coords, list) and len(coords) > 0:
        fixed_polygons = []
        for polygon in coords:
            if not isinstance(polygon, list) or len(polygon) == 0:
                continue
            fixed_rings = []
            for ring in polygon:
                if not isinstance(ring, list) or len(ring) < 3:
                    continue
                fixed_rings.append(_ensure_closed_ring(ring))
            if fixed_rings:
                fixed_polygons.append(fixed_rings)
        if not fixed_polygons:
            return None
        return {"type": "MultiPolygon", "coordinates": fixed_polygons}

    if gtype == "LineString" and isinstance(coords, list) and len(coords) >= 3:
        ring = _ensure_closed_ring(coords)
        if len(ring) < 4:
            return None
        return {"type": "Polygon", "coordinates": [ring]}

    return None


def _pick_latest_valid_polygon(output: dict | None, previous_feature: dict | None = None) -> dict | None:
    if not output:
        return previous_feature

    candidates = []
    # Prefer committed drawings, but also allow last active polygon/rectangle so selections register.
    all_drawings = output.get("all_drawings") or []
    if all_drawings:
        candidates.extend(reversed(all_drawings))

    last_active = output.get("last_active_drawing")
    if isinstance(last_active, dict):
        raw_geom = last_active.get("geometry") if "geometry" in last_active else last_active
        raw_type = raw_geom.get("type") if isinstance(raw_geom, dict) else None
        # Ignore transient line events; accept only polygon-like active shapes.
        if raw_type in {"Polygon", "MultiPolygon"}:
            candidates.append(last_active)

    for candidate in candidates:
        feat = _normalize_feature(candidate)
        if not feat:
            continue
        geometry = _sanitize_polygon_geometry(feat.get("geometry", {}))
        if geometry is None:
            continue
        if not _geometry_has_enough_points(geometry):
            continue
        feat["geometry"] = geometry
        return feat

    return previous_feature


def _round_nested_coords(value, digits: int = 6):
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    if isinstance(value, list):
        return [_round_nested_coords(v, digits=digits) for v in value]
    return value


def _geometry_signature(geometry: dict) -> str:
    if not isinstance(geometry, dict):
        return ""
    stable = {
        "type": geometry.get("type"),
        "coordinates": _round_nested_coords(geometry.get("coordinates"), digits=6),
    }
    return json.dumps(stable, sort_keys=True)


# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = PROJECT_ROOT / "data/processed/grid/grid_100m.gpkg"
PRED_DIR = PROJECT_ROOT / "data/processed/predictions"
POP_DIR = PROJECT_ROOT / "data/processed/population"
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_KEYS = ["built", "veg", "water", "other"]
CLASS_LABELS = {
    "built": "Built-up",
    "veg": "Vegetation",
    "water": "Water",
    "other": "Other",
}
CLASS_COLORS = {
    "built": "#8a9a5b",
    "veg": "#7fcb4d",
    "water": "#84c7d0",
    "other": "#b7c89b",
}

# Map-only palette requested by user.
MAP_CLASS_COLORS = {
    "built": "#e53935",
    "veg": "#43a047",
    "water": "#1e88e5",
    "other": "#757575",
}


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def load_grid() -> gpd.GeoDataFrame:
    grid = gpd.read_file(GRID_PATH)
    if grid.crs is None:
        raise ValueError("Grid CRS is missing.")
    if "cell_id" not in grid.columns:
        raise ValueError("Grid file must contain 'cell_id' column.")

    grid = grid.to_crs(epsg=4326)
    grid_m = grid.to_crs(epsg=3857)
    grid["cell_area_m2"] = grid_m.geometry.area.values
    return grid


@st.cache_data(show_spinner=False)
def list_prediction_years() -> list[int]:
    years = []
    for fp in PRED_DIR.glob("pred_*.parquet"):
        try:
            years.append(int(fp.stem.split("_")[1]))
        except Exception:
            continue
    return sorted(set(years))


@st.cache_data(show_spinner=False)
def list_population_years() -> list[int]:
    years = []
    for fp in POP_DIR.glob("pop_*_by_cell_id.csv"):
        parts = fp.stem.split("_")
        if len(parts) >= 2:
            try:
                years.append(int(parts[1]))
            except Exception:
                continue
    return sorted(set(years))


@st.cache_data(show_spinner=False)
def load_prediction(year: int) -> pd.DataFrame:
    return pd.read_parquet(PRED_DIR / f"pred_{year}.parquet")


@st.cache_data(show_spinner=False)
def load_population(year: int) -> pd.DataFrame:
    df = pd.read_csv(POP_DIR / f"pop_{year}_by_cell_id.csv")
    pop_col = f"pop_{year}_sum"
    if pop_col not in df.columns:
        raise ValueError(f"Population column missing: {pop_col}")
    return df[["cell_id", pop_col]].rename(columns={pop_col: "population"})


@st.cache_data(show_spinner=False)
def load_model_context() -> dict:
    out = {
        "mae": {},
        "r2": {},
        "unc_mean": {},
        "unc_p90": {},
    }

    metrics_path = MODELS_DIR / "metrics_spatial_cv_summary.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        lgbm = df[df["model"].astype(str).str.lower() == "lgbm"]
        for _, row in lgbm.iterrows():
            tgt = str(row.get("target", ""))
            out["mae"][tgt] = float(row.get("mae_mean", np.nan))
            out["r2"][tgt] = float(row.get("r2_mean", np.nan))

    unc_path = MODELS_DIR / "uncertainty_summary.csv"
    if unc_path.exists():
        udf = pd.read_csv(unc_path)
        for _, row in udf.iterrows():
            tgt = str(row.get("target", ""))
            out["unc_mean"][tgt] = float(row.get("mean_uncertainty", np.nan))
            out["unc_p90"][tgt] = float(row.get("p90_uncertainty", np.nan))

    return out


def compute_weighted_composition(selection: gpd.GeoDataFrame, pred_df: pd.DataFrame) -> dict[str, float]:
    merged = selection[["cell_id", "weight"]].merge(pred_df, on="cell_id", how="inner")
    total_weight = merged["weight"].sum()
    if total_weight <= 0:
        return {k: 0.0 for k in CLASS_KEYS}

    out: dict[str, float] = {}
    for key in CLASS_KEYS:
        col = f"{key}_pred_lgbm"
        vals = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        out[key] = float((vals * merged["weight"]).sum() / total_weight * 100.0)
    return out


def compute_population_for_year(selection: gpd.GeoDataFrame, year: int) -> float:
    base = selection[["cell_id", "weight"]].copy()
    pop = load_population(year)
    merged = base.merge(pop, on="cell_id", how="left")
    merged["population"] = pd.to_numeric(merged["population"], errors="coerce").fillna(0.0)
    return float((merged["population"] * merged["weight"]).sum())


def compute_population_series(selection: gpd.GeoDataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        rows.append({"year": int(year), "population": float(compute_population_for_year(selection, int(year)))})
    return pd.DataFrame(rows).sort_values("year")


def build_selection_report_frames(
    selection: gpd.GeoDataFrame,
    mode: str,
    composition: dict[str, float] | None,
    selected_year: int | None,
    start_year: int | None,
    end_year: int | None,
    pop_years: list[int],
) -> str:
    summary_rows = []
    summary_rows.append({"metric": "mode", "value": mode})
    summary_rows.append({"metric": "selected_area_km2", "value": float(selection["intersection_area_m2"].sum() / 1_000_000.0)})
    summary_rows.append({"metric": "equivalent_100m_cells", "value": float(selection["intersection_area_m2"].sum() / 10_000.0)})

    if mode == "composition" and composition is not None and selected_year is not None:
        summary_rows.append({"metric": "year", "value": int(selected_year)})
        for k in CLASS_KEYS:
            summary_rows.append({"metric": f"{k}_share_percent", "value": float(composition.get(k, 0.0))})
    if mode == "change" and start_year is not None and end_year is not None:
        summary_rows.append({"metric": "start_year", "value": int(start_year)})
        summary_rows.append({"metric": "end_year", "value": int(end_year)})

    pop_series = compute_population_series(selection, pop_years)
    for _, row in pop_series.iterrows():
        summary_rows.append({"metric": f"population_{int(row['year'])}", "value": float(row["population"])})

    summary_df = pd.DataFrame(summary_rows)
    detail_cols = ["cell_id", "weight", "intersection_area_m2"]
    details_df = selection[detail_cols].copy().sort_values("weight", ascending=False)
    return summary_df, details_df


def build_selection_report_csv(
    selection: gpd.GeoDataFrame,
    mode: str,
    composition: dict[str, float] | None,
    selected_year: int | None,
    start_year: int | None,
    end_year: int | None,
    pop_years: list[int],
) -> str:
    summary_df, details_df = build_selection_report_frames(
        selection=selection,
        mode=mode,
        composition=composition,
        selected_year=selected_year,
        start_year=start_year,
        end_year=end_year,
        pop_years=pop_years,
    )
    summary_csv = summary_df.to_csv(index=False)
    details_csv = details_df.to_csv(index=False)
    return summary_csv + "\n" + "cell_details" + "\n" + details_csv


def build_executive_summary_pdf(
    title: str,
    mode: str,
    narrative_lines: list[str],
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    key_points: list[str],
) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")

        y = 0.98
        ax.text(0.02, y, title, fontsize=15, fontweight="bold", color="#244b1f", va="top")
        y -= 0.04
        ax.text(0.02, y, f"Mode: {mode}", fontsize=10, color="#2f4a2b", va="top")
        y -= 0.025
        ax.text(0.02, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, color="#4f6a49", va="top")

        y -= 0.05
        ax.text(0.02, y, "Selection Narrative", fontsize=12, fontweight="bold", color="#1f3f1a", va="top")
        y -= 0.03
        for line in narrative_lines:
            ax.text(0.03, y, f"- {line}", fontsize=10, color="#233f1f", va="top")
            y -= 0.022

        y -= 0.02
        ax.text(0.02, y, "Key Highlights", fontsize=12, fontweight="bold", color="#1f3f1a", va="top")
        y -= 0.03
        for point in key_points:
            ax.text(0.03, y, f"- {point}", fontsize=10, color="#233f1f", va="top")
            y -= 0.022

        y -= 0.02
        ax.text(0.02, y, "Summary Metrics (from CSV export)", fontsize=12, fontweight="bold", color="#1f3f1a", va="top")
        y -= 0.03
        for _, row in summary_df.head(22).iterrows():
            ax.text(0.03, y, f"{row['metric']}: {row['value']}", fontsize=9, color="#2b3c2a", va="top")
            y -= 0.018
            if y < 0.08:
                break

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig2 = plt.figure(figsize=(8.27, 11.69))
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        ax2.text(0.02, 0.98, "Top Cell Details (from CSV export)", fontsize=13, fontweight="bold", color="#1f3f1a", va="top")
        top_details = details_df.head(30).copy()
        if not top_details.empty:
            top_details["weight"] = top_details["weight"].map(lambda x: f"{float(x):.4f}")
            top_details["intersection_area_m2"] = top_details["intersection_area_m2"].map(lambda x: f"{float(x):.2f}")
            table_text = top_details.to_string(index=False)
            ax2.text(0.02, 0.94, table_text, fontsize=8, family="monospace", color="#2b3c2a", va="top")
        else:
            ax2.text(0.02, 0.94, "No cell details available.", fontsize=10, color="#2b3c2a", va="top")
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    return buf.getvalue()


def _zoom_to_tile_size_m(zoom_level: float) -> float:
    if zoom_level >= 14:
        return 220.0
    if zoom_level >= 13:
        return 400.0
    if zoom_level >= 12:
        return 700.0
    return 1100.0


def aggregate_selection_for_map(merged: gpd.GeoDataFrame, zoom_level: float) -> gpd.GeoDataFrame:
    if merged.empty:
        return merged

    tile_size_m = _zoom_to_tile_size_m(zoom_level)
    centroids = merged.to_crs(epsg=3857).geometry.centroid
    work = merged.copy()
    work["tile_x"] = np.floor(centroids.x / tile_size_m).astype(int)
    work["tile_y"] = np.floor(centroids.y / tile_size_m).astype(int)

    bounds = work.geometry.bounds
    work["minx"] = bounds["minx"]
    work["miny"] = bounds["miny"]
    work["maxx"] = bounds["maxx"]
    work["maxy"] = bounds["maxy"]

    agg_rows = []
    for _, grp in work.groupby(["tile_x", "tile_y"], dropna=False):
        w = pd.to_numeric(grp["weight"], errors="coerce").fillna(0.0)
        wsum = float(w.sum())
        if wsum <= 0:
            w = pd.Series(np.ones(len(grp)), index=grp.index, dtype=float)
            wsum = float(w.sum())

        row = {
            "geometry": box(float(grp["minx"].min()), float(grp["miny"].min()), float(grp["maxx"].max()), float(grp["maxy"].max())),
            "cell_id": f"tile_{int(grp['tile_x'].iloc[0])}_{int(grp['tile_y'].iloc[0])}",
            "weight": wsum,
            "built_pred_lgbm": float((pd.to_numeric(grp["built_pred_lgbm"], errors="coerce").fillna(0.0) * w).sum() / wsum),
            "veg_pred_lgbm": float((pd.to_numeric(grp["veg_pred_lgbm"], errors="coerce").fillna(0.0) * w).sum() / wsum),
            "water_pred_lgbm": float((pd.to_numeric(grp["water_pred_lgbm"], errors="coerce").fillna(0.0) * w).sum() / wsum),
            "other_pred_lgbm": float((pd.to_numeric(grp["other_pred_lgbm"], errors="coerce").fillna(0.0) * w).sum() / wsum),
        }
        agg_rows.append(row)

    return gpd.GeoDataFrame(agg_rows, geometry="geometry", crs=merged.crs)


def scenario_projection(comp_end: dict[str, float], built_adj: float, veg_adj: float, water_adj: float) -> tuple[dict[str, float], float]:
    other_adj = -(built_adj + veg_adj + water_adj)
    simulated = {
        "built": max(0.0, comp_end["built"] + built_adj),
        "veg": max(0.0, comp_end["veg"] + veg_adj),
        "water": max(0.0, comp_end["water"] + water_adj),
        "other": max(0.0, comp_end["other"] + other_adj),
    }
    total = float(sum(simulated.values()))
    if total <= 0:
        return comp_end.copy(), other_adj
    simulated = {k: float(v / total * 100.0) for k, v in simulated.items()}
    return simulated, other_adj


def fit_density_relation_for_selection(
    selection: gpd.GeoDataFrame,
    years: list[int],
    selected_area_km2: float,
) -> dict:
    rows = []
    for year in years:
        try:
            pred = load_prediction(int(year))
            comp = compute_weighted_composition(selection, pred)
            pop = compute_population_for_year(selection, int(year))
            density = pop / selected_area_km2 if selected_area_km2 > 0 else np.nan
            rows.append(
                {
                    "year": int(year),
                    "built": float(comp.get("built", 0.0)),
                    "veg": float(comp.get("veg", 0.0)),
                    "water": float(comp.get("water", 0.0)),
                    "density": float(density),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3:
        return {
            "intercept": 0.0,
            "built": 0.0,
            "veg": 0.0,
            "water": 0.0,
            "r2": np.nan,
            "n_obs": int(len(df)),
            "ok": False,
        }

    X = df[["built", "veg", "water"]].to_numpy(dtype=float)
    y = df["density"].to_numpy(dtype=float)

    A = np.column_stack([np.ones(len(X), dtype=float), X])
    ridge_lambda = 0.2
    reg = np.eye(A.shape[1], dtype=float)
    reg[0, 0] = 0.0

    try:
        beta = np.linalg.solve(A.T @ A + ridge_lambda * reg, A.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A.T @ A + ridge_lambda * reg) @ (A.T @ y)

    y_hat = A @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "intercept": float(beta[0]),
        "built": float(beta[1]),
        "veg": float(beta[2]),
        "water": float(beta[3]),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "n_obs": int(len(df)),
        "ok": True,
    }


def predict_density_from_relation(comp: dict[str, float], relation: dict) -> float:
    return float(
        relation.get("intercept", 0.0)
        + relation.get("built", 0.0) * float(comp.get("built", 0.0))
        + relation.get("veg", 0.0) * float(comp.get("veg", 0.0))
        + relation.get("water", 0.0) * float(comp.get("water", 0.0))
    )


def reliability_label_from_uncertainty(mean_unc: float, p90_unc: float) -> str:
    if not np.isfinite(mean_unc) or not np.isfinite(p90_unc) or p90_unc <= 0:
        return "n/a"
    ratio = mean_unc / p90_unc
    if ratio <= 0.45:
        return "higher confidence"
    if ratio <= 0.70:
        return "moderate confidence"
    return "cautious confidence"


def render_compact_bar(label: str, value: float, color: str) -> None:
    pct = max(0.0, min(100.0, float(value)))
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; font-size:0.95rem;">
                <span><b>{label}</b></span>
                <span>{pct:.2f}%</span>
            </div>
            <div style="width:100%; background:#e9f3e4; border-radius:999px; height:10px; margin-top:6px;">
                <div style="width:{pct}%; background:{color}; height:10px; border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_delta_card(label: str, start_val: float, end_val: float, delta_pp: float) -> None:
    arrow = "↑" if delta_pp > 0 else ("↓" if delta_pp < 0 else "→")
    tone = "#2f855a" if delta_pp > 0 else ("#b45309" if delta_pp < 0 else "#6b7280")
    st.markdown(
        f"""
        <div style="padding:12px; border:1px solid #d9e8d3; border-radius:12px; background:#f8fff5;">
            <div style="font-weight:600; margin-bottom:6px;">{label}</div>
            <div style="font-size:0.9rem; color:#4b5563;">{start_val:.2f}% → {end_val:.2f}%</div>
            <div style="font-size:1.1rem; color:{tone}; font-weight:700; margin-top:4px;">{arrow} {delta_pp:+.2f} pp</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_pill(text: str, bg: str = "#e6f4de", color: str = "#2d5a25") -> None:
    st.markdown(
        f"<span class='insight-pill' style='background:{bg}; color:{color};'>{text}</span>",
        unsafe_allow_html=True,
    )


def composition_diversity(composition: dict[str, float]) -> float:
    probs = np.array([max(v, 0.0) for v in composition.values()], dtype=float)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(len(probs))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy * 100.0)


def render_story_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class='panel-card' style='padding:8px 10px;'>
            <div style='font-size:0.78rem; color:#4f6a49; font-weight:600;'>{title}</div>
            <div style='font-size:1.1rem; font-weight:800; color:#1f3f1a; margin-top:2px;'>{value}</div>
            <div style='font-size:0.76rem; color:#597453; margin-top:2px;'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_selection_from_feature(feature: dict | None, grid_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    if not feature or "geometry" not in feature:
        return None
    try:
        aoi_geom = shape(feature["geometry"])
        if not aoi_geom.is_valid:
            aoi_geom = aoi_geom.buffer(0)

        aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")
        selection = gpd.overlay(grid_df[["cell_id", "geometry", "cell_area_m2"]], aoi, how="intersection")
        if selection.empty:
            return None

        selection_m = selection.to_crs(epsg=3857)
        selection["intersection_area_m2"] = selection_m.geometry.area.values
        selection["weight"] = (selection["intersection_area_m2"] / selection["cell_area_m2"]).clip(lower=0.0, upper=1.0)
        return selection
    except Exception:
        return None


def _split_geom_by_proportions(geom, built: float, veg: float, water: float, other: float) -> list[tuple[object, str]]:
    vals = np.array([max(built, 0.0), max(veg, 0.0), max(water, 0.0), max(other, 0.0)], dtype=float)
    total = float(vals.sum())
    if total <= 0:
        return [(geom, "other")]
    vals = vals / total

    minx, miny, maxx, maxy = geom.bounds
    width = float(maxx - minx)
    if width <= 0:
        dominant_idx = int(np.argmax(vals))
        return [(geom, CLASS_KEYS[dominant_idx])]

    pieces: list[tuple[object, str]] = []
    cursor = float(minx)
    for idx, key in enumerate(CLASS_KEYS):
        frac = float(vals[idx])
        if frac <= 0:
            continue
        x2 = float(maxx) if idx == len(CLASS_KEYS) - 1 else float(cursor + width * frac)
        strip = box(cursor, miny, x2, maxy)
        part = geom.intersection(strip)
        if not part.is_empty:
            pieces.append((part, key))
        cursor = x2

    return pieces if pieces else [(geom, "other")]


def add_grid_mix_overlay(
    m: folium.Map,
    selection: gpd.GeoDataFrame,
    pred_df: pd.DataFrame,
    fill_mode: str = "Proportional (striped)",
    max_detailed_cells: int = 1000,
) -> dict:
    cols = ["cell_id", "built_pred_lgbm", "veg_pred_lgbm", "water_pred_lgbm", "other_pred_lgbm"]
    merged = selection.merge(pred_df[cols], on="cell_id", how="left")
    if merged.empty:
        return {"cell_count": 0, "render_mode": "none", "downgraded": False}

    merged["dominant"] = merged[["built_pred_lgbm", "veg_pred_lgbm", "water_pred_lgbm", "other_pred_lgbm"]].idxmax(axis=1)
    merged["dominant"] = merged["dominant"].str.replace("_pred_lgbm", "", regex=False)

    use_dominant = fill_mode == "Dominant class" or len(merged) > max_detailed_cells

    def outline_style(_: dict) -> dict:
        return {
            "color": "#f5f5f5",
            "weight": 0.30,
            "fillOpacity": 0.0,
        }

    if use_dominant:
        def dom_style(feat: dict) -> dict:
            cls = feat.get("properties", {}).get("dominant", "other")
            return {
                "color": "#f5f5f5",
                "weight": 0.30,
                "fillColor": MAP_CLASS_COLORS.get(cls, MAP_CLASS_COLORS["other"]),
                "fillOpacity": 0.52,
            }

        folium.GeoJson(
            merged[["geometry", "dominant", "cell_id"]],
            style_function=dom_style,
            name="Selected grids (dominant)",
        ).add_to(m)
        return {
            "cell_count": int(len(merged)),
            "render_mode": "dominant",
            "downgraded": fill_mode != "Dominant class",
        }

    part_rows = []
    for _, row in merged.iterrows():
        geom = row.get("geometry")
        if geom is None or geom.is_empty:
            continue
        parts = _split_geom_by_proportions(
            geom,
            float(row.get("built_pred_lgbm", 0.0)),
            float(row.get("veg_pred_lgbm", 0.0)),
            float(row.get("water_pred_lgbm", 0.0)),
            float(row.get("other_pred_lgbm", 0.0)),
        )
        for part_geom, cls in parts:
            part_rows.append({"geometry": part_geom, "cls": cls, "cell_id": row.get("cell_id")})

    if not part_rows:
        return {"cell_count": int(len(merged)), "render_mode": "none", "downgraded": False}

    parts_gdf = gpd.GeoDataFrame(part_rows, geometry="geometry", crs=selection.crs)

    def part_style(feat: dict) -> dict:
        cls = feat.get("properties", {}).get("cls", "other")
        return {
            "color": MAP_CLASS_COLORS.get(cls, MAP_CLASS_COLORS["other"]),
            "weight": 0.0,
            "fillColor": MAP_CLASS_COLORS.get(cls, MAP_CLASS_COLORS["other"]),
            "fillOpacity": 0.54,
        }

    folium.GeoJson(
        parts_gdf[["geometry", "cls", "cell_id"]],
        style_function=part_style,
        name="Selected grids (proportions)",
    ).add_to(m)

    folium.GeoJson(
        merged[["geometry", "cell_id"]],
        style_function=outline_style,
        name="Selected grid outlines",
    ).add_to(m)

    return {
        "cell_count": int(len(merged)),
        "render_mode": "proportional",
        "downgraded": False,
    }


def render_composition_donut(composition: dict[str, float]) -> None:
    built = max(0.0, composition["built"])
    veg = max(0.0, composition["veg"])
    water = max(0.0, composition["water"])
    other = max(0.0, composition["other"])

    total = built + veg + water + other
    if total <= 0:
        st.info("No composition values to display.")
        return

    built = built / total * 100.0
    veg = veg / total * 100.0
    water = water / total * 100.0
    other = other / total * 100.0

    b2 = built
    b3 = b2 + veg
    b4 = b3 + water

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:18px; margin-bottom:8px;">
            <div style="
                width: 150px;
                height: 150px;
                border-radius: 50%;
                background: conic-gradient(
                    {CLASS_COLORS['built']} 0% {b2:.2f}%,
                    {CLASS_COLORS['veg']} {b2:.2f}% {b3:.2f}%,
                    {CLASS_COLORS['water']} {b3:.2f}% {b4:.2f}%,
                    {CLASS_COLORS['other']} {b4:.2f}% 100%
                );
                position: relative;
            ">
                <div style="
                    position:absolute;
                    inset:30px;
                    background:#f8fff5;
                    border-radius:50%;
                "></div>
            </div>
            <div style="font-size:0.9rem; line-height:1.75;">
                <div><span style="display:inline-block; width:10px; height:10px; background:{CLASS_COLORS['built']}; border-radius:2px;"></span> Built-up</div>
                <div><span style="display:inline-block; width:10px; height:10px; background:{CLASS_COLORS['veg']}; border-radius:2px;"></span> Vegetation</div>
                <div><span style="display:inline-block; width:10px; height:10px; background:{CLASS_COLORS['water']}; border-radius:2px;"></span> Water</div>
                <div><span style="display:inline-block; width:10px; height:10px; background:{CLASS_COLORS['other']}; border-radius:2px;"></span> Other</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# VALIDATION
# -----------------------------
if not GRID_PATH.exists():
    st.error(f"Missing grid file: {GRID_PATH}")
    st.stop()
if not PRED_DIR.exists():
    st.error(f"Missing predictions directory: {PRED_DIR}")
    st.stop()
if not POP_DIR.exists():
    st.error(f"Missing population directory: {POP_DIR}")
    st.stop()


grid = load_grid()
pred_years = list_prediction_years()
pop_years = list_population_years()

if not pred_years:
    st.error("No prediction files found (pred_YYYY.parquet).")
    st.stop()

if not pop_years:
    st.error("No population files found (pop_YYYY_by_cell_id.csv).")
    st.stop()

analysis_years = sorted(set(pred_years))
model_context = load_model_context()


# -----------------------------
# PAGE
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🌿 Nuremberg Land-Cover Intelligence</div>
        <div class="hero-sub">Draw a polygon to unlock live composition, change, and population insights — all in one screen.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "selected_feature" not in st.session_state:
    st.session_state["selected_feature"] = None
if "selected_feature_signature" not in st.session_state:
    st.session_state["selected_feature_signature"] = None
if "map_nonce" not in st.session_state:
    st.session_state["map_nonce"] = 0
if "show_map_color_info" not in st.session_state:
    st.session_state["show_map_color_info"] = False


# -----------------------------
# TOP CONTROLS
# -----------------------------
c_mode, c_y1, c_y2, c_map = st.columns([1.0, 1.0, 1.0, 1.0])
with c_mode:
    mode = st.selectbox("Mode", ["composition", "change"], index=0)

with c_map:
    map_fill_mode = st.selectbox("Map fill", ["Proportional (striped)", "Dominant class"], index=0)

if mode == "composition":
    with c_y1:
        selected_year = st.selectbox("Year", analysis_years, index=len(analysis_years) - 1)
    with c_y2:
        st.markdown("<div style='font-size:0.95rem; color:#3f5f3b; margin-bottom:0.25rem;'>Summary</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-card' style='margin-bottom:0; padding:10px 12px;'>Data for the selected year and area</div>",
            unsafe_allow_html=True,
        )
else:
    with c_y1:
        start_year = st.selectbox("Start year", analysis_years[:-1], index=0)
    with c_y2:
        end_options = [y for y in analysis_years if y > start_year]
        end_year = st.selectbox("End year", end_options, index=len(end_options) - 1)


# -----------------------------
# LAYOUT: MAP + DASHBOARD
# -----------------------------
left_col, right_col = st.columns([1.25, 1.0], gap="medium")

city_center = [49.4521, 11.0767]
m = folium.Map(location=city_center, zoom_start=12, control_scale=True, double_click_zoom=False)

# Render selected AOI grids on the map.
preview_feature = st.session_state.get("selected_feature")
preview_selection = compute_selection_from_feature(preview_feature, grid)
overlay_meta = {"cell_count": 0, "render_mode": "none", "downgraded": False}
if preview_selection is not None and len(preview_selection) > 0:
    overlay_year = selected_year if mode == "composition" else end_year
    overlay_pred = load_prediction(overlay_year)
    overlay_meta = add_grid_mix_overlay(
        m,
        preview_selection,
        overlay_pred,
        fill_mode=map_fill_mode,
    )
    minx, miny, maxx, maxy = preview_selection.total_bounds
    if np.isfinite([minx, miny, maxx, maxy]).all():
        m.fit_bounds([[float(miny), float(minx)], [float(maxy), float(maxx)]])

Draw(
    export=False,
    draw_options={
        "polyline": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "polygon": True,
        "rectangle": True,
    },
    edit_options={"edit": True, "remove": True},
).add_to(m)

with left_col:
    st.markdown("<div class='panel-card' style='padding:6px;'>", unsafe_allow_html=True)
    map_h1, map_h2 = st.columns([0.9, 0.1])
    with map_h1:
        st.markdown("<div style='font-weight:700; color:#2f4a2b; padding:2px 0 6px 2px;'>Selection map</div>", unsafe_allow_html=True)
    with map_h2:
        if st.button("i", key="map_color_info_btn", help="Map color representation", use_container_width=True):
            st.session_state["show_map_color_info"] = not st.session_state["show_map_color_info"]

    if st.session_state["show_map_color_info"]:
        st.markdown(
            """
            <div class='panel-card' style='padding:8px; margin-top:0; margin-bottom:6px;'>
                <b>Map color representation</b><br>
                Red = Built-up, Green = Vegetation, Blue = Water, Gray = Other.<br>
                Proportional mode splits each selected grid into colored strips by class percentage.<br>
                Dominant mode colors each selected grid by its largest class only.
            </div>
            """,
            unsafe_allow_html=True,
        )

    output = st_folium(
        m,
        height=500,
        use_container_width=True,
        key=f"aoi_map_{st.session_state['map_nonce']}",
        returned_objects=["last_active_drawing", "all_drawings"],
    )

    if output:
        has_active = output.get("last_active_drawing") is not None
        drawings_count = len(output.get("all_drawings") or [])
        latest = _pick_latest_valid_polygon(output, None)
        gtype = latest.get("geometry", {}).get("type", "none") if latest else "none"
        status = f"Draw status → active: {'yes' if has_active else 'no'}, saved shapes: {drawings_count}, captured: {gtype}"
        st.caption(status)
    if overlay_meta.get("downgraded"):
        st.caption(
            f"Rendered {overlay_meta['cell_count']} cells in dominant mode for smooth performance. "
            "Choose a smaller AOI to see proportional strips."
        )
    st.markdown("</div>", unsafe_allow_html=True)
    clear_clicked = st.button("Clear polygon", use_container_width=True)


# -----------------------------
# POLYGON SELECTION
# -----------------------------
if clear_clicked:
    st.session_state["selected_feature"] = None
    st.session_state["selected_feature_signature"] = None
    st.session_state["map_nonce"] = int(st.session_state.get("map_nonce", 0)) + 1
    st.rerun()

latest_feature = _pick_latest_valid_polygon(output, st.session_state["selected_feature"])

if latest_feature is not None:
    signature = _geometry_signature(latest_feature.get("geometry", {}))
    if signature != st.session_state["selected_feature_signature"]:
        st.session_state["selected_feature"] = latest_feature
        st.session_state["selected_feature_signature"] = signature
        # Force a fresh map mount so fit_bounds is applied reliably.
        st.session_state["map_nonce"] = int(st.session_state.get("map_nonce", 0)) + 1
        st.rerun()

feature = st.session_state["selected_feature"]
selection = None
reliability_card = None
change_post_section = None

if feature and feature.get("geometry"):
    try:
        aoi_geom = shape(feature["geometry"])
        if not aoi_geom.is_valid:
            aoi_geom = aoi_geom.buffer(0)

        aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")
        selection = gpd.overlay(grid[["cell_id", "geometry", "cell_area_m2"]], aoi, how="intersection")
    except Exception:
        st.warning("Polygon could not be processed. Please redraw the polygon clearly and close it.")
        st.stop()

    if selection.empty:
        st.warning("Please select an area within Nuremberg.")
        st.stop()

    selection_m = selection.to_crs(epsg=3857)
    selection["intersection_area_m2"] = selection_m.geometry.area.values
    selection["weight"] = (selection["intersection_area_m2"] / selection["cell_area_m2"]).clip(lower=0.0, upper=1.0)

    selected_area_km2 = float(selection["intersection_area_m2"].sum() / 1_000_000.0)
    equivalent_cells = float(selection["intersection_area_m2"].sum() / 10_000.0)

    with right_col:
        a1, a2 = st.columns(2)
        a1.metric("Selected area (km²)", f"{selected_area_km2:.3f}")
        a2.metric("Equivalent 100m cells", f"{equivalent_cells:.1f}")


# -----------------------------
# ANALYTICS
# -----------------------------
if selection is not None:
    with right_col:
        if mode == "composition":
            pred = load_prediction(selected_year)
            composition = compute_weighted_composition(selection, pred)
            population_selected_year = compute_population_for_year(selection, selected_year)

            dominant_key = max(composition, key=composition.get)
            green_blue = composition["veg"] + composition["water"]
            urban_natural_gap = composition["built"] - green_blue
            sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)
            dominance_margin = sorted_comp[0][1] - sorted_comp[1][1] if len(sorted_comp) > 1 else sorted_comp[0][1]
            diversity = composition_diversity(composition)
            pop_density = population_selected_year / selected_area_km2 if selected_area_km2 > 0 else 0.0

            # Determine diversity/balance assessment
            if diversity > 70:
                diversity_desc = "highly diverse and mixed"
            elif diversity > 50:
                diversity_desc = "fairly balanced"
            else:
                diversity_desc = "dominated by one main cover type"
            
            # Density context
            if pop_density < 100:
                density_context = "rural or sparse"
            elif pop_density < 500:
                density_context = "moderate suburban"
            else:
                density_context = "dense urban"
            
            # Built-to-green assessment
            if urban_natural_gap > 20:
                balance_text = f"with {urban_natural_gap:.1f}pp more built-up than green space"
            elif urban_natural_gap > 0:
                balance_text = f"with slightly more urban than natural cover"
            else:
                balance_text = f"with more natural than urban cover"
            
            comp_narrative_lines = [
                f"In {selected_year}, {CLASS_LABELS[dominant_key]} is the dominant land cover at {composition[dominant_key]:.1f}%.",
                f"Green plus water accounts for {green_blue:.1f}% ({balance_text}), and the area is {diversity_desc}.",
                f"Estimated population is {population_selected_year:,.0f}, with density around {pop_density:,.0f}/km² ({density_context}).",
            ]
            st.markdown(
                "<div class='panel-card'><b>Selection narrative</b><br>"
                f"{comp_narrative_lines[0]}<br>{comp_narrative_lines[1]}<br>{comp_narrative_lines[2]}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='panel-card'><b>Composition ({selected_year})</b></div>", unsafe_allow_html=True)

            i1, i2, i3 = st.columns(3)
            i1.metric("Dominant cover", f"{CLASS_LABELS[dominant_key]} ({composition[dominant_key]:.1f}%)")
            i2.metric("Green + Water", f"{green_blue:.1f}%")
            i3.metric("Built − (Green+Water)", f"{urban_natural_gap:+.1f} pp")

            s1, s2, s3 = st.columns(3)
            with s1:
                render_story_card("Land diversity", f"{diversity:.1f}/100", "Higher = more mixed land cover")
            with s2:
                render_story_card("Dominance margin", f"{dominance_margin:.1f} pp", "Gap between top two classes")
            with s3:
                render_story_card("Population density", f"{pop_density:,.0f}/km²", f"Estimated for {selected_year}")

            d_col, b_col = st.columns([0.9, 1.1], gap="small")
            with d_col:
                render_composition_donut(composition)
            with b_col:
                for k in CLASS_KEYS:
                    render_compact_bar(CLASS_LABELS[k], composition[k], CLASS_COLORS[k])

            pop_series = compute_population_series(selection, pop_years)
            base_year = int(pop_series.iloc[0]["year"]) if not pop_series.empty else selected_year
            base_pop = float(pop_series.iloc[0]["population"]) if not pop_series.empty else population_selected_year
            pop_delta_base = population_selected_year - base_pop
            pop_delta_base_pct = (pop_delta_base / base_pop * 100.0) if base_pop > 0 else 0.0
            latest_year = int(pop_series.iloc[-1]["year"]) if not pop_series.empty else selected_year
            latest_pop = float(pop_series.iloc[-1]["population"]) if not pop_series.empty else population_selected_year
            since_latest = population_selected_year - latest_pop if selected_year != latest_year else 0.0
            since_latest_pct = (since_latest / latest_pop * 100.0) if latest_pop > 0 and selected_year != latest_year else 0.0

            st.markdown("<div class='panel-card'><b>Population context</b></div>", unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            p1.metric(f"Population ({selected_year})", f"{population_selected_year:,.0f}")
            p2.metric("Per 100m cell", f"{(population_selected_year / equivalent_cells if equivalent_cells > 0 else 0):.1f}")
            p3.metric(f"Δ vs {base_year}", f"{pop_delta_base:+,.0f} ({pop_delta_base_pct:+.1f}%)")

            st.markdown("<div style='margin-top:2px;'>", unsafe_allow_html=True)
            if selected_year != latest_year:
                render_insight_pill(
                    f"Vs latest ({latest_year}): {since_latest:+,.0f} ({since_latest_pct:+.1f}%)",
                    "#eef6ff",
                    "#21486b",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            dom_target = f"{dominant_key}_prop"
            dom_mae = model_context.get("mae", {}).get(dom_target, np.nan)
            dom_r2 = model_context.get("r2", {}).get(dom_target, np.nan)
            dom_unc_mean = model_context.get("unc_mean", {}).get(dom_target, np.nan)
            dom_unc_p90 = model_context.get("unc_p90", {}).get(dom_target, np.nan)

            reliability_card = {
                "m1_label": "OOF MAE (dominant class)",
                "m1_value": "n/a" if not np.isfinite(dom_mae) else f"{dom_mae:.3f}",
                "m2_label": "OOF R² (dominant class)",
                "m2_value": "n/a" if not np.isfinite(dom_r2) else f"{dom_r2:.3f}",
                "m3_label": "Confidence",
                "m3_value": reliability_label_from_uncertainty(dom_unc_mean, dom_unc_p90),
            }

            summary_df, details_df = build_selection_report_frames(
                selection=selection,
                mode="composition",
                composition=composition,
                selected_year=selected_year,
                start_year=None,
                end_year=None,
                pop_years=pop_years,
            )
            export_csv = summary_df.to_csv(index=False) + "\n" + "cell_details" + "\n" + details_df.to_csv(index=False)
            executive_pdf = build_executive_summary_pdf(
                title="Nuremberg Land-Cover Executive Summary",
                mode=f"composition ({selected_year})",
                narrative_lines=comp_narrative_lines,
                summary_df=summary_df,
                details_df=details_df,
                key_points=[
                    f"Dominant cover: {CLASS_LABELS[dominant_key]} ({composition[dominant_key]:.1f}%)",
                    f"Green + Water share: {green_blue:.1f}%",
                    f"Population ({selected_year}): {population_selected_year:,.0f}",
                    f"Density: {pop_density:,.0f}/km²",
                ],
            )

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="Download selection report (CSV)",
                    data=export_csv,
                    file_name=f"selection_report_composition_{selected_year}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="Download executive summary (PDF)",
                    data=executive_pdf,
                    file_name=f"selection_executive_summary_{selected_year}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        else:
            pred_start = load_prediction(start_year)
            pred_end = load_prediction(end_year)
            pop_start = compute_population_for_year(selection, start_year)
            pop_end = compute_population_for_year(selection, end_year)
            pop_delta = pop_end - pop_start

            comp_start = compute_weighted_composition(selection, pred_start)
            comp_end = compute_weighted_composition(selection, pred_end)

            change_df = pd.DataFrame(
                {
                    "class": [CLASS_LABELS[k] for k in CLASS_KEYS],
                    "start_percent": [comp_start[k] for k in CLASS_KEYS],
                    "end_percent": [comp_end[k] for k in CLASS_KEYS],
                    "change_pp": [comp_end[k] - comp_start[k] for k in CLASS_KEYS],
                }
            )

            max_gain = change_df.sort_values("change_pp", ascending=False).iloc[0]
            max_loss = change_df.sort_values("change_pp", ascending=True).iloc[0]
            built_delta = comp_end["built"] - comp_start["built"]
            total_shift = float(change_df["change_pp"].abs().sum() / 2.0)
            density_start = pop_start / selected_area_km2 if selected_area_km2 > 0 else 0.0
            density_end = pop_end / selected_area_km2 if selected_area_km2 > 0 else 0.0

            # Characterize overall trend
            if built_delta > 5:
                trend = "strong urbanization"
            elif built_delta > 1:
                trend = "moderate urban growth"
            elif built_delta > -1:
                trend = "stable composition"
            elif built_delta > -5:
                trend = "moderate greening"
            else:
                trend = "significant greening"
            
            # Population trend context
            if pop_delta > 0 and built_delta > 0:
                pop_context = f"population grew by {pop_delta:+,.0f}, accompanying urbanization"
            elif pop_delta > 0:
                pop_context = f"population grew by {pop_delta:+,.0f} while cover remained stable"
            elif pop_delta < 0 and built_delta > 0:
                pop_context = f"population declined by {pop_delta:,.0f} despite urbanization"
            else:
                pop_context = f"population changed by {pop_delta:+,.0f}"
            
            change_narrative_lines = [
                f"From {start_year} to {end_year}, this area shows {trend}, with built-up changing by {built_delta:+.2f} pp.",
                f"The largest increase is in {max_gain['class']} ({float(max_gain['change_pp']):+.2f} pp), while {max_loss['class']} has the largest decline ({float(max_loss['change_pp']):+.2f} pp).",
                f"Overall land reallocation is {total_shift:.2f} pp, and {pop_context}.",
            ]
            st.markdown(
                "<div class='panel-card'><b>Selection narrative</b><br>"
                f"{change_narrative_lines[0]}<br>{change_narrative_lines[1]}<br>{change_narrative_lines[2]}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"<div class='panel-card'><b>Change ({start_year} → {end_year})</b></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                render_delta_card("Built-up", comp_start["built"], comp_end["built"], comp_end["built"] - comp_start["built"])
                render_delta_card("Vegetation", comp_start["veg"], comp_end["veg"], comp_end["veg"] - comp_start["veg"])
            with c2:
                render_delta_card("Water", comp_start["water"], comp_end["water"], comp_end["water"] - comp_start["water"])
                render_delta_card("Other", comp_start["other"], comp_end["other"], comp_end["other"] - comp_start["other"])

            st.bar_chart(change_df.set_index("class")["change_pp"], height=180, use_container_width=True)

            st.markdown("<div style='margin-top:2px;'>", unsafe_allow_html=True)
            tg_col, tl_col = st.columns(2)
            with tg_col:
                render_insight_pill(
                    f"Top gain: {max_gain['class']} ({float(max_gain['change_pp']):+.2f} pp)",
                    "#daf2d1",
                    "#21501b",
                )
            with tl_col:
                render_insight_pill(
                    f"Top loss: {max_loss['class']} ({float(max_loss['change_pp']):+.2f} pp)",
                    "#f8ead6",
                    "#7a4a11",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div class='panel-card'><b>What-if simulator</b><br>"
                "Use the sliders to test planning scenarios by changing land-cover shares in percentage points (pp). "
                "When you increase one class, the model auto-adjusts <i>Other</i> so total share stays at 100%. "
                "Projected density shows the directional impact of your scenario.</div>",
                unsafe_allow_html=True,
            )
            sim1, sim2, sim3 = st.columns(3)
            with sim1:
                built_adj = st.slider("Built-up adj (pp)", -15.0, 15.0, 0.0, 0.5, key="sim_built")
            with sim2:
                veg_adj = st.slider("Vegetation adj (pp)", -15.0, 15.0, 0.0, 0.5, key="sim_veg")
            with sim3:
                water_adj = st.slider("Water adj (pp)", -15.0, 15.0, 0.0, 0.5, key="sim_water")

            sim_comp, sim_other_adj = scenario_projection(comp_end, built_adj, veg_adj, water_adj)
            overlap_years = sorted(set(analysis_years).intersection(set(pop_years)))
            relation = fit_density_relation_for_selection(selection, overlap_years, selected_area_km2)
            if relation.get("ok", False):
                pred_base = predict_density_from_relation(comp_end, relation)
                pred_sim = predict_density_from_relation(sim_comp, relation)
                rel_delta = pred_sim - pred_base
                rel_delta = float(np.clip(rel_delta, -0.6 * max(density_end, 1.0), 0.6 * max(density_end, 1.0)))
                projected_density = max(0.0, density_end + rel_delta)
            else:
                projected_density = max(0.0, density_end)

            ws1, ws2, ws3 = st.columns(3)
            ws1.metric("Projected density", f"{projected_density:,.0f}/km²", f"{(projected_density - density_end):+,.0f}/km²")
            ws2.metric("Projected built-up", f"{sim_comp['built']:.1f}%", f"{(sim_comp['built'] - comp_end['built']):+.1f} pp")
            ws3.metric("Auto-balanced other adj", f"{sim_other_adj:+.1f} pp")

            tradeoffs = sorted(
                [
                    ("Built-up", built_adj),
                    ("Vegetation", veg_adj),
                    ("Water", water_adj),
                    ("Other", sim_other_adj),
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            top_up = tradeoffs[0]
            top_down = sorted(tradeoffs, key=lambda x: x[1])[0]
            tpos_col, tneg_col = st.columns(2)
            with tpos_col:
                render_insight_pill(f"Top positive shift: {top_up[0]} ({top_up[1]:+.1f} pp)", "#daf2d1", "#21501b")
            with tneg_col:
                render_insight_pill(f"Top negative shift: {top_down[0]} ({top_down[1]:+.1f} pp)", "#f8ead6", "#7a4a11")
            if relation.get("ok", False):
                r2_txt = "n/a" if not np.isfinite(relation.get("r2", np.nan)) else f"{relation['r2']:.2f}"
                st.caption(
                    f"Note: scenario density is estimated using a fitted multi-year relation in this AOI "
                    #f"(built/vegetation/water → density, n={relation.get('n_obs', 0)}, R²={r2_txt})."
                )
            else:
                st.caption("Note: not enough historical points to calibrate relation; projected density stays close to current level.")

            pop_series = compute_population_series(selection, pop_years)
            pop_delta_pct = (pop_delta / pop_start * 100.0) if pop_start > 0 else 0.0



            unc_vals = []
            unc_map = model_context.get("unc_mean", {})
            for tgt in ["delta_built", "delta_veg", "delta_water", "delta_other"]:
                val = unc_map.get(tgt, np.nan)
                if np.isfinite(val):
                    unc_vals.append(float(val))
            mean_change_unc = float(np.mean(unc_vals)) if unc_vals else np.nan

            mae_vals = [float(v) for v in model_context.get("mae", {}).values() if np.isfinite(v)]
            r2_vals = [float(v) for v in model_context.get("r2", {}).values() if np.isfinite(v)]
            mean_mae = float(np.mean(mae_vals)) if mae_vals else np.nan
            mean_r2 = float(np.mean(r2_vals)) if r2_vals else np.nan
            reliability_card = {
                "m1_label": "Mean OOF MAE",
                "m1_value": "n/a" if not np.isfinite(mean_mae) else f"{mean_mae:.3f}",
                "m2_label": "Mean OOF R²",
                "m2_value": "n/a" if not np.isfinite(mean_r2) else f"{mean_r2:.3f}",
                "m3_label": "Change uncertainty",
                "m3_value": "n/a" if not np.isfinite(mean_change_unc) else f"{mean_change_unc:.4f}",
            }

            summary_df, details_df = build_selection_report_frames(
                selection=selection,
                mode="change",
                composition=None,
                selected_year=None,
                start_year=start_year,
                end_year=end_year,
                pop_years=pop_years,
            )
            export_csv = summary_df.to_csv(index=False) + "\n" + "cell_details" + "\n" + details_df.to_csv(index=False)
            executive_pdf = build_executive_summary_pdf(
                title="Nuremberg Land-Cover Executive Summary",
                mode=f"change ({start_year} to {end_year})",
                narrative_lines=change_narrative_lines,
                summary_df=summary_df,
                details_df=details_df,
                key_points=[
                    f"Built-up shift: {built_delta:+.2f} pp",
                    f"Top gain: {max_gain['class']} ({float(max_gain['change_pp']):+.2f} pp)",
                    f"Top loss: {max_loss['class']} ({float(max_loss['change_pp']):+.2f} pp)",
                    f"Population change: {pop_delta:+,.0f}",
                ],
            )
            change_post_section = {
                "start_year": int(start_year),
                "end_year": int(end_year),
                "pop_start": float(pop_start),
                "pop_end": float(pop_end),
                "pop_delta": float(pop_delta),
                "pop_delta_pct": float(pop_delta_pct),
                "csv": export_csv,
                "pdf": executive_pdf,
            }

    if reliability_card is not None:
        with left_col:
            st.markdown("<div class='panel-card'><b>Model reliability context</b></div>", unsafe_allow_html=True)
            lr1, lr2, lr3 = st.columns(3)
            lr1.metric(reliability_card["m1_label"], reliability_card["m1_value"])
            lr2.metric(reliability_card["m2_label"], reliability_card["m2_value"])
            lr3.metric(reliability_card["m3_label"], reliability_card["m3_value"])

    if mode == "change" and change_post_section is not None:
        with left_col:
            st.markdown("<div class='panel-card'><b>Population context</b></div>", unsafe_allow_html=True)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric(f"{change_post_section['start_year']}", f"{change_post_section['pop_start']:,.0f}")
            p2.metric(f"{change_post_section['end_year']}", f"{change_post_section['pop_end']:,.0f}")
            p3.metric("Δ", f"{change_post_section['pop_delta']:+,.0f}")
            p4.metric("Δ %", f"{change_post_section['pop_delta_pct']:+.1f}%")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="Download selection report (CSV)",
                    data=change_post_section["csv"],
                    file_name=f"selection_report_change_{change_post_section['start_year']}_{change_post_section['end_year']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="Download executive summary (PDF)",
                    data=change_post_section["pdf"],
                    file_name=f"selection_executive_summary_{change_post_section['start_year']}_{change_post_section['end_year']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

else:
    with right_col:
        st.markdown(
            """
            <div class="panel-card" style="height:520px; display:flex; align-items:center; justify-content:center; text-align:center;">
                <div>
                    <div style="font-size:1.2rem; font-weight:700; margin-bottom:6px;">Draw your area to begin</div>
                    <div style="color:#4b5563;">Use polygon or rectangle tool on the map. Results will appear here instantly.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
