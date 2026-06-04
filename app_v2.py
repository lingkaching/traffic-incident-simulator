"""
risk_simulator_v12.py
=====================
Fleet Risk Simulator — Integrated UI + Risk Engine
Requires data_layer.py in the same directory.

Run:
    streamlit run risk_simulator_v12.py
"""

import streamlit as st
import osmnx as ox
import networkx as nx
import math
import copy
import pickle
import os
import random
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import time

# ── data layer ────────────────────────────────────────────────────────────────
from data_layer_v2 import (
    DriverDB, VehicleDB, TripLogDB, EnvironmentSim,
    RouteFeatureExtractor, RouteDB, build_task_context,
    VEHICLE_ELIGIBILITY, ROAD_CLASS_LABELS, estimate_duration_h,
)
# ── allocator ─────────────────────────────────────────────────────────────────
from allocator import Allocator, Task as AllocTask, AllocationResult

st.set_page_config(layout="wide", page_title="Traffic Risk Assessment", page_icon="🛡️")

# ══════════════════════════════════════════════════════════════════════════════
# CSS  — clean light gov theme (unchanged from v11)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f5f6f8;
    color: #1a1d23;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0.5rem 0.8rem 1rem !important; max-width: 100% !important; }

.app-header {
    background: #ffffff; border-bottom: 2px solid #e2e5ea;
    padding: 10px 18px; margin-bottom: 14px;
    display: flex; align-items: center; justify-content: space-between;
}
.app-title   { font-size: 17px; font-weight: 700; color: #1a1d23; letter-spacing: -.01em; }
.app-subtitle{ font-size: 11px; color: #6b7280; letter-spacing: .04em; text-transform: uppercase; margin-top: 1px; }
.status-chip { font-family: 'DM Mono',monospace; font-size: 10px; background: #ecfdf5;
               color: #059669; border: 1px solid #6ee7b7; border-radius: 20px; padding: 3px 10px; }

.panel-label { font-size: 14px; font-weight: 600; color: #6b7280; text-transform: uppercase;
               letter-spacing: .1em; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1.5px solid #e2e5ea; }
.section-label { font-size: 10px; font-weight: 600; color: #9ca3af; text-transform: uppercase;
                 letter-spacing: .09em; margin: 14px 0 6px 0; }

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTimeInput > div > div > input {
    background-color: #ffffff !important; border: 1.5px solid #d1d5db !important;
    color: #1a1d23 !important; border-radius: 6px !important;
    font-family: 'DM Sans',sans-serif !important; font-size: 13px !important;
}
.stButton > button[kind="primary"] {
    background: #1d4ed8 !important; color: #fff !important; border: none !important;
    border-radius: 6px !important; font-family: 'DM Sans',sans-serif !important;
    font-weight: 600 !important; width: 100% !important; padding: 10px 0 !important;
    font-size: 13px !important; letter-spacing: .02em !important; margin-top: 10px !important;
}
.stButton > button[kind="primary"]:hover { background: #1e40af !important; }
.stButton > button:not([kind="primary"]) {
    background: #fff !important; color: #374151 !important; border: 1.5px solid #d1d5db !important;
    border-radius: 5px !important; font-size: 11px !important; font-weight: 500 !important; padding: 4px 10px !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #f0f5ff !important; border-color: #3b82f6 !important; color: #1d4ed8 !important;
}

.risk-pill  { display:inline-block; padding:2px 8px; border-radius:20px;
              font-size:10px; font-weight:600; letter-spacing:.06em; }
.pill-low   { background:#ecfdf5; color:#059669; border:1px solid #6ee7b7; }
.pill-med   { background:#fffbeb; color:#d97706; border:1px solid #fcd34d; }
.pill-high  { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5; }

.driver-profile-box { background:#fff; border:1.5px solid #e2e5ea; border-radius:8px;
                      padding:14px 16px; margin-bottom:10px; }
.profile-name { font-size:15px; font-weight:700; color:#111827; }
.profile-id   { font-family:'DM Mono',monospace; font-size:10px; color:#9ca3af; margin-top:1px; }

.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin:10px 0 8px; }
.metric-box  { background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
               padding:8px 10px; text-align:center; }
.metric-val  { font-family:'DM Mono',monospace; font-size:16px; font-weight:700; color:#1d4ed8; }
.metric-lbl  { font-size:9px; color:#9ca3af; text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }

.factor-row   { display:flex; justify-content:space-between; align-items:center;
                padding:7px 0; border-bottom:1px solid #f3f4f6; font-size:12px; }
.factor-label { color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.07em; }
.factor-value { font-family:'DM Mono',monospace; font-size:11px; color:#111827; }
.factor-value.good   { color:#059669; }
.factor-value.warn   { color:#d97706; }
.factor-value.danger { color:#dc2626; }

.hdivider { border:none; border-top:1px solid #e5e7eb; margin:10px 0; }
[data-testid="stDataFrame"] { border:1px solid #e2e5ea !important; border-radius:6px !important; }
.stSlider > div { padding:0 !important; }
.empty-state { text-align:center; padding:70px 20px; color:#9ca3af; }

/* ── Collapsible factor sections ── */
div[data-testid="stExpander"] {
    border: 1px solid #e2e5ea !important;
    border-radius: 6px !important;
    background: #ffffff !important;
    margin-bottom: 6px !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    padding: 8px 12px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: #f9fafb !important;
    border-radius: 6px !important;
}
div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    padding: 2px 12px 8px !important;
    border-top: 1px solid #f3f4f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
BASE_LAMBDA  = 1e-6

MILEAGE_MULT = {"<20k": 1.4, "20k-100k": 1.1, "100k-300k": 0.95, ">300k": 0.9}
PURPOSE_MULT = {"Admin": 1.0, "Training": 1.05, "Operation": 1.2, "Emergency": 1.6}
VTYPE_MULT   = {"Car": 1.0, "Light Truck": 1.2, "5T": 1.5, "10T": 2.0}
ROAD_MULT    = {"motorway": 0.7, "trunk": 0.9, "primary": 1.2,
                "secondary": 1.5, "tertiary": 1.8, "residential": 2.2}

FATIGUE_RATE       = 0.08
CRASH_MULT         = 1.5
ROUTE_EXP_THRESHOLD = 10   # trips on this specific route → High familiarity
ROUTE_EXP_BONUS    = 0.80  # High route familiarity
ROUTE_EXP_MED_MULT = 1.00  # Medium route familiarity
ROUTE_EXP_LOW_MULT = 1.35  # Low route familiarity
# Vehicle-type experience: graduated by trips with the assigned vehicle type.
# Mirrors the route familiarity structure — three tiers based on trip count.
VEH_TYPE_EXP_HIGH_THRESH = 100   # ≥50 trips with this type → High (experienced)
VEH_TYPE_EXP_MED_THRESH  = 50   # ≥10 trips               → Medium
                                 # <10 trips               → Low  (novice on this type)
VEH_TYPE_EXP_HIGH_MULT   = 0.85 # experienced: reduced risk
VEH_TYPE_EXP_MED_MULT    = 1.00 # neutral
VEH_TYPE_EXP_LOW_MULT    = 1.30 # novice on this vehicle type: elevated risk
MAINT_POOR_MULT    = 1.5   # Poor condition multiplier       (was 1.3 — understated unsafe vehicles)
TECH_PENALTY       = 0.05
# NOTE: no separate NIGHT_VIS_MULT — night is already captured by effective_visibility_km()
# which caps visibility at _NIGHT_VIS_CAP_KM (4 km) at night, directly raising vis_m.
# Adding a second night multiplier on top would double-count the effect.
NARROW_ROAD_MULT   = 1.3
SHARP_TURN_MULT    = 1.4
HOTSPOT_MULT_VAL   = 2.5

# ══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE FUNCTIONS
# All inputs are typed data-layer objects — no raw dicts from old design
# ══════════════════════════════════════════════════════════════════════════════

def risk_category(prob: float) -> str:
    # Low  : prob < 0.03%  (routine assignments under normal conditions)
    # Medium: 0.03% – 0.10% (elevated but manageable — wider band than before)
    # High : prob ≥ 0.10%  (genuinely dangerous; reserve for worst-case combos)
    if prob < 0.05:  return "Low"
    if prob < 0.10:  return "Medium"
    return "High"

def risk_pill_html(level: str) -> str:
    cls = {"Low": "pill-low", "Medium": "pill-med", "High": "pill-high"}[level]
    return f'<span class="risk-pill {cls}">{level}</span>'

def frow(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="factor-row">'
            f'<span class="factor-label">{label}</span>'
            f'<span class="factor-value {cls}">{value}</span>'
            f'</div>')

# ══════════════════════════════════════════════════════════════════════════════
# INDEX FUNCTIONS
# P, T, A depend only on their own domain inputs — no segment data needed.
# E and R are length-weighted averages over route segments; they are computed
# inside compute_route_risk() in a single pass so we never iterate edges twice.
# ══════════════════════════════════════════════════════════════════════════════

# ── Index P : Driver ──────────────────────────────────────────────────────────
def compute_P(driver_ctx: dict) -> float:
    """
    P = mileage × crash × fatigue

    Fatigue uses a piecewise non-linear curve:
      0–4 h  : 0.05/hr  (normal working hours)
      4–10 h : 0.10/hr  (fatigue accumulating)
      >10 h  : 0.18/hr  (real impairment zone)
    """
    profile = driver_ctx["profile"]
    fatigue = driver_ctx["fatigue_hours"]

    if fatigue <= 4:
        fatigue_mult = 1 + fatigue * 0.05
    elif fatigue <= 10:
        fatigue_mult = 1 + 4 * 0.05 + (fatigue - 4) * 0.10
    else:
        fatigue_mult = 1 + 4 * 0.05 + 6 * 0.10 + (fatigue - 10) * 0.18

    return (MILEAGE_MULT[profile.mileage_band]
            * (CRASH_MULT if profile.crash_history else 1.0)
            * fatigue_mult)


# ── Index T : Task ────────────────────────────────────────────────────────────
def compute_T(driver_ctx: dict, purpose: str, route_idx: int) -> float:
    """
    T = purpose × route_familiarity × veh_type_exp

    route_familiarity: how many times the driver has taken this specific
    route index between the O-D pair (High / Medium / Low).

    veh_type_exp: trips with the assigned vehicle type — graduated into
    three tiers (High / Medium / Low), same structure as route familiarity.
    Replaces the old binary veh_mult (yes/no) and the weak total-trips
    exp_mult, which was too coarse to be meaningful.
    """
    route_exp      = driver_ctx["route_experience"]
    veh_type_trips = driver_ctx["veh_type_trips"]

    route_fam = TripLogDB.per_route_familiarity(route_exp, route_idx)
    fam_mult  = {"High": ROUTE_EXP_BONUS,
                 "Medium": ROUTE_EXP_MED_MULT,
                 "Low":    ROUTE_EXP_LOW_MULT}[route_fam]

    if veh_type_trips >= VEH_TYPE_EXP_HIGH_THRESH:
        veh_type_exp_mult = VEH_TYPE_EXP_HIGH_MULT
        veh_type_level    = "High"
    elif veh_type_trips >= VEH_TYPE_EXP_MED_THRESH:
        veh_type_exp_mult = VEH_TYPE_EXP_MED_MULT
        veh_type_level    = "Medium"
    else:
        veh_type_exp_mult = VEH_TYPE_EXP_LOW_MULT
        veh_type_level    = "Low"

    return PURPOSE_MULT[purpose] * fam_mult * veh_type_exp_mult


# ── Index A : Vehicle ─────────────────────────────────────────────────────────
def compute_A(vehicle_ctx: dict, weather: "WeatherSnapshot") -> float:
    """
    A = vehicle_type × condition × (1 − safety_tech_discount) × braking_mult

    condition: Good → 1.0 / Fair → 1.1 / Poor → MAINT_POOR_MULT
    braking_mult: weather-driven wet-road stopping-distance penalty.
    """
    profile   = vehicle_ctx["profile"]
    condition = vehicle_ctx["condition"]
    tech_disc = len(profile.safe_tech) * TECH_PENALTY
    cond_mult = (MAINT_POOR_MULT if condition == "Poor" else
                 1.1             if condition == "Fair" else 1.0)
    return (VTYPE_MULT.get(profile.vehicle_type, 1.0)
            * cond_mult
            * (1 - tech_disc)
            * weather.braking_mult)


# ── Per-edge roadway multiplier (helper used inside compute_route_risk) ────────
def _roadway_mult_for_edge(edge: dict) -> float:
    """
    R_i = hotspot × road_class × narrow × sharp, capped at 4.0.
    Cap prevents a single residential-hotspot-narrow-sharp edge (~10×)
    from dominating the entire route lambda.
    """
    hw = edge.get("highway", "residential")
    if isinstance(hw, list):
        hw = hw[0] if hw else "residential"
    mult = edge.get("hotspot_mult", 1.0)
    mult *= ROAD_MULT.get(hw, 1.5)
    if edge.get("is_narrow"): mult *= NARROW_ROAD_MULT
    if edge.get("is_sharp"):  mult *= SHARP_TURN_MULT
    return min(mult, 4.0)


# ── Full route risk ────────────────────────────────────────────────────────────
def compute_route_risk(route_features, P: float, T: float, A: float,
                       env_ctx: dict, dep_time: "time", DG) -> dict:
    """
    Single-pass over route edges. Computes per-segment lambda and derives
    the route-level E and R indices as length-weighted averages:

        E_i = vis_m × traffic_m(i)
        R_i = hotspot × road_class × narrow × sharp   [capped 4.0]

        E = Σ_i  E_i × len(i) / total_len
        R = Σ_i  R_i × len(i) / total_len

        λ_i = BASE_LAMBDA × len(i) × P × T × A × E_i × R_i
        prob = 1 − exp(−Σ λ_i)

    vis_m is computed by EnvironmentSim.visibility_mult(), which calls
    effective_visibility_km() — this already caps visibility at _NIGHT_VIS_CAP_KM
    (4 km) at night, so the night penalty is fully captured inside vis_m.
    No separate night multiplier is needed or applied.
    """
    weather_snap = env_ctx["weather"]
    vis_m        = EnvironmentSim.visibility_mult(weather_snap, dep_time)
    # Night effect is already inside vis_m via the 4 km visibility cap.

    traffic_lookup: dict[tuple, float] = {
        (s.u, s.v): s.density for s in route_features.segment_traffic
    }

    nodes          = route_features.node_sequence
    total_lambda   = 0.0
    total_len      = 0.0
    weighted_E     = 0.0   # Σ E_i × len(i)
    weighted_R     = 0.0   # Σ R_i × len(i)
    segment_data   = []
    hotspot_coords = []

    for u, v in zip(nodes[:-1], nodes[1:]):
        edge      = DG.get_edge_data(u, v) or {}
        length    = edge.get("length", 0.0)

        traffic_m = traffic_lookup.get((u, v), 3.0) / 3.0
        E_i       = vis_m * traffic_m
        R_i       = _roadway_mult_for_edge(edge)

        weighted_E += E_i * length
        weighted_R += R_i * length
        total_len  += length

        if edge.get("hotspot_mult", 1.0) > 1.0:
            hotspot_coords.append((DG.nodes[u]["y"], DG.nodes[u]["x"]))

        lam = BASE_LAMBDA * length * P * T * A * E_i * R_i
        total_lambda += lam
        segment_data.append((u, v, lam))          # store raw λ_i; fraction computed below

    denom = max(total_len, 1e-9)
    E = weighted_E / denom
    R = weighted_R / denom

    # Express each segment's lambda as a fraction of total_lambda.
    # This directly connects map colour to the final prob = 1 − exp(−Σλ_i):
    # red segments are the ones most responsible for the route's risk number.
    total_lambda_safe = max(total_lambda, 1e-30)
    segments_with_frac = [(u, v, lam_i / total_lambda_safe)
                          for u, v, lam_i in segment_data]

    return {
        "prob":           1 - math.exp(-total_lambda),
        "total_lambda":   total_lambda,
        "dist_km":        route_features.dist_km,
        "segments":       segments_with_frac,      # (u, v, frac_i) where frac_i = λ_i/Σλ
        "hotspot_coords": hotspot_coords,
        "P": P, "T": T, "A": A, "E": E, "R": R,
        "vis_m":         vis_m,
        "avg_traffic_m": weighted_E / (vis_m * denom) if vis_m > 0 else 1.0,
    }

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH LOADING
# ══════════════════════════════════════════════════════════════════════════════
GRAPH_PATH = "sg_drive.pkl"

@st.cache_resource
def load_graph():
    if not os.path.exists(GRAPH_PATH):
        G_raw = ox.graph_from_place("Singapore", network_type="drive")
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(G_raw, f)
    else:
        with open(GRAPH_PATH, "rb") as f:
            G_raw = pickle.load(f)

    G_raw = ox.bearing.add_edge_bearings(G_raw)
    DG    = nx.DiGraph()
    for node, data in G_raw.nodes(data=True):
        DG.add_node(node, x=data["x"], y=data["y"])

    all_candidates = []
    for u, v, k, data in G_raw.edges(keys=True, data=True):
        cd = {k2: (v2[0] if isinstance(v2, list) else v2) for k2, v2 in data.items()}
        w_raw = str(cd.get("width", "6.0")).replace("m", "").split(",")[0].strip()
        width = float(w_raw) if w_raw.replace(".", "", 1).isdigit() else 6.0
        lr    = str(cd.get("lanes", "2")).split(",")[0].strip()
        lanes = int(lr) if lr.isdigit() else 2
        cd["is_narrow"]    = (width < 6.0) or (lanes < 2)
        cd["hotspot_mult"] = 1.0
        cd["is_sharp"]     = False
        if G_raw.degree[u] >= 4 or cd.get("highway") == "motorway_link" or cd.get("bridge") == "yes":
            all_candidates.append((u, v))
        DG.add_edge(u, v, **cd)

    rng = random.Random(42)
    for u, v in rng.sample(all_candidates, min(len(all_candidates), 1000)):
        DG[u][v]["hotspot_mult"] = HOTSPOT_MULT_VAL

    for node in DG.nodes():
        for _, _vi, d_in in DG.in_edges(node, data=True):
            for _, v_out, _ in DG.out_edges(node, data=True):
                b1, b2 = d_in.get("bearing", 0), DG.get_edge_data(node, v_out).get("bearing", 0)
                diff = abs(b1 - b2)
                if min(diff, 360 - diff) > 90 and rng.random() < 0.2:
                    DG[node][v_out]["is_sharp"] = True

    for u, v in DG.edges():
        if DG[u][v].get("is_narrow") and rng.random() > 0.2:
            DG[u][v]["is_narrow"] = False

    return G_raw, DG

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER SINGLETONS  (cached so they're built once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_data_layer():
    driver_db  = DriverDB()
    vehicle_db = VehicleDB()
    route_db   = RouteDB("route.json")
    trip_db    = TripLogDB(driver_db, vehicle_db, route_db)
    env_sim    = EnvironmentSim(seed=42)
    return driver_db, vehicle_db, trip_db, env_sim, route_db

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS = {
    "results":       None,
    "sel_idx":       0,
    "task_params":   {},
    "app_mode":      "Single Task",       # "Single Task" | "Batch Allocation"
    "batch_tasks":   [],                  # list of AllocTask dicts (serialisable)
    "alloc_result":  None,                # AllocationResult
    "alloc_sel_idx": 0,                   # selected assignment index in alloc view
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# PATHFINDING  — find up to 3 diverse routes
# ══════════════════════════════════════════════════════════════════════════════
def find_candidate_routes(DG, orig: int, dest: int) -> list[list[int]]:
    routes, temp = [], DG.copy()
    for _ in range(3):
        try:
            path = nx.shortest_path(temp, orig, dest, weight="length")
            routes.append(path)
            for u, v in zip(path[:-1], path[1:]):
                temp[u][v]["length"] *= 2.5
        except nx.NetworkXNoPath:
            break
    return routes

# ══════════════════════════════════════════════════════════════════════════════
# RESULT BUILDER  — combine task_context + risk engine into result rows
# ══════════════════════════════════════════════════════════════════════════════
def build_results(task_ctx: dict, DG) -> list[dict]:
    """
    For each eligible driver × each candidate route, compute full risk.
    Returns list of result dicts sorted by prob ascending.
    """
    rows        = []
    vehicle_ctx = task_ctx["vehicle"]
    env_ctx     = task_ctx["environment"]
    weather     = env_ctx["weather"]
    dep_time    = task_ctx["dep_time"]
    A           = compute_A(vehicle_ctx, weather)

    for driver_ctx in task_ctx["drivers"]:
        P = compute_P(driver_ctx)
        for rf in task_ctx["routes"]:
            T    = compute_T(driver_ctx, task_ctx["purpose"], rf.route_index)
            risk = compute_route_risk(rf, P, T, A, env_ctx, dep_time, DG)
            rows.append({
                "driver_ctx":     driver_ctx,
                "vehicle_ctx":    vehicle_ctx,
                "env_ctx":        env_ctx,
                "route_features": rf,
                "risk":           risk,
                "prob":           risk["prob"],
                "purpose":        task_ctx["purpose"],
                "dep_time":       dep_time,
            })

    rows.sort(key=lambda x: x["prob"])
    return rows



def top3_by_risk(rows: list[dict]) -> list[dict]:
    """
    Pick the 3 lowest-risk combinations from the full driver × route matrix.
 
    Only constraint: no duplicate (driver_id, route_index) pair.
    No forced driver diversity, no forced route diversity — the risk score
    is the sole ranking criterion. Different drivers naturally produce
    different scores because P varies per driver; different routes produce
    different scores because T (familiarity) and E (per-segment traffic)
    vary per route.
    """
    seen: set[tuple] = set()
    top3: list[dict] = []
 
    for row in rows:          # already sorted lowest prob first
        key = (
            row["driver_ctx"]["profile"].driver_id,
            row["route_features"].route_index,
        )
        if key not in seen:
            seen.add(key)
            top3.append(row)
        if len(top3) == 3:
            break
 
    return top3

# ══════════════════════════════════════════════════════════════════════════════
# LIVE RECOMPUTE  — used when fatigue slider changes
# ══════════════════════════════════════════════════════════════════════════════
def compute_risk_for_result(result: dict, DG) -> dict:
    """Compute risk for a stored result using its driver_ctx fatigue as-is."""
    driver_ctx  = result["driver_ctx"]
    vehicle_ctx = result["vehicle_ctx"]
    env_ctx     = result["env_ctx"]
    rf          = result["route_features"]
    purpose     = result["purpose"]
    dep_time    = result["dep_time"]
    weather     = env_ctx["weather"]

    P = compute_P(driver_ctx)
    T = compute_T(driver_ctx, purpose, rf.route_index)
    A = compute_A(vehicle_ctx, weather)
    return compute_route_risk(rf, P, T, A, env_ctx, dep_time, DG)

# ══════════════════════════════════════════════════════════════════════════════
# MAP RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_map(G_raw, result: dict, risk: dict) -> folium.Map:
    nodes  = result["route_features"].node_sequence
    mid_nd = nodes[len(nodes) // 2]
    m = folium.Map(location=[G_raw.nodes[mid_nd]["y"], G_raw.nodes[mid_nd]["x"]],
                   zoom_start=13, tiles="CartoDB positron")

    legend_html = """
    <div style="position:fixed;bottom:24px;left:24px;background:#fff;
                border:1px solid #d1d5db;border-radius:8px;padding:10px 14px;
                font-size:11px;font-family:sans-serif;z-index:9999;
                box-shadow:0 2px 6px rgba(0,0,0,.1);">
        <b>Contribution of Route Risk</b><br>
        <span style="color:#16a34a;">●</span> &lt;1% &nbsp;
        <span style="color:#d97706;">●</span> 1–5% &nbsp;
        <span style="color:#dc2626;">●</span> &gt;5%<br>
        <span style="font-size:10px;color:#6b7280;">Hover for exact % of total risk</span><br>
        <span>⚠️</span> Collision Hotspot
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
        

    # frac_i = λ_i / Σλ: the fraction of total route risk from this segment.
    # Colour by direct percentage — no normalisation needed:
    #   green  < 1%  : minor contributor
    #   amber  1–5%  : notable contributor
    #   red    > 5%  : this one segment accounts for >5% of the route's total
    #                  incident probability — a genuine hotspot
    # The tooltip shows the exact % so map and risk number are on the same scale.
    for u, v, frac in risk["segments"]:
        pct   = frac * 100
        color = "#dc2626" if pct > 5.0 else "#d97706" if pct > 1.0 else "#16a34a"
        pts   = [[G_raw.nodes[u]["y"], G_raw.nodes[u]["x"]],
                 [G_raw.nodes[v]["y"], G_raw.nodes[v]["x"]]]
        folium.PolyLine(pts, color=color, weight=5, opacity=0.85,
                        tooltip=f"{pct:.2f}% of route risk").add_to(m)

    for lat, lon in set(risk.get("hotspot_coords", [])):
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html='<div style="font-size:16px;">⚠️</div>'),
            tooltip="Collision Hotspot",
        ).add_to(m)

    folium.Marker(
        [G_raw.nodes[nodes[0]]["y"],  G_raw.nodes[nodes[0]]["x"]],
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        tooltip="Origin",
    ).add_to(m)
    folium.Marker(
        [G_raw.nodes[nodes[-1]]["y"], G_raw.nodes[nodes[-1]]["x"]],
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        tooltip="Destination",
    ).add_to(m)
    return m

# ══════════════════════════════════════════════════════════════════════════════
# APP HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛡️ Driver × Vehicle Risk Assessment</div>
    </div>
    <div class="status-chip">● System Ready</div>
</div>
""", unsafe_allow_html=True)

# ── Mode toggle ───────────────────────────────────────────────────────────────
mode_col, _ = st.columns([2, 5])
with mode_col:
    app_mode = st.radio(
        "Mode", ["Single Task", "Batch Allocation"],
        horizontal=True, label_visibility="collapsed",
        index=0 if st.session_state.app_mode == "Single Task" else 1,
    )
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode = app_mode
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading road network and data layer…"):
    G, DG = load_graph()
    driver_db, vehicle_db, trip_db, env_sim, route_db = get_data_layer()

RANK_LABEL     = ["#1", "#2", "#3"]
RANK_EMOJI     = ["🥇", "🥈", "🥉"]
DRIVER_AVATARS = ["👨‍✈️", "👩‍✈️", "🧑‍✈️", "👨‍🔧", "👩‍🔧"]

# ══════════════════════════════════════════════════════════════════════════════
# SHARED RIGHT-PANEL RENDERER
# Called by both Single Task mode and Batch Allocation mode.
# ══════════════════════════════════════════════════════════════════════════════

def _render_right_panel(result_dict, DG, driver_ctx, vehicle_ctx, env_ctx,
                        rf, purpose_v, dep_time_v):
    """Render the driver profile box + all P/T/A/E/R factor expanders."""
    profile      = driver_ctx["profile"]
    vprofile     = vehicle_ctx["profile"]
    weather_snap = env_ctx["weather"]

    # Re-run risk with the provided driver_ctx (may have chained fatigue injected)
    live_risk  = compute_risk_for_result({
        "driver_ctx":     driver_ctx,
        "vehicle_ctx":    vehicle_ctx,
        "env_ctx":        env_ctx,
        "route_features": rf,
        "purpose":        purpose_v,
        "dep_time":       dep_time_v,
    }, DG)
    live_prob  = live_risk["prob"]
    live_level = risk_category(live_prob)

    avatar         = DRIVER_AVATARS[int(profile.driver_id[1:]) % len(DRIVER_AVATARS)]
    crash_txt      = "🔴 Yes" if profile.crash_history else "🟢 None"
    fam            = TripLogDB.per_route_familiarity(driver_ctx["route_experience"], rf.route_index)
    fam_trips      = driver_ctx["route_experience"]["by_route"].get(rf.route_index, 0)
    veh_type_trips = driver_ctx["veh_type_trips"]
    total_trips    = driver_ctx["total_trips"]
    fat_h          = driver_ctx["fatigue_hours"]

    if veh_type_trips >= VEH_TYPE_EXP_HIGH_THRESH:
        veh_type_level = "High";   veh_type_m = VEH_TYPE_EXP_HIGH_MULT
    elif veh_type_trips >= VEH_TYPE_EXP_MED_THRESH:
        veh_type_level = "Medium"; veh_type_m = VEH_TYPE_EXP_MED_MULT
    else:
        veh_type_level = "Low";    veh_type_m = VEH_TYPE_EXP_LOW_MULT

    st.markdown(f"""
    <div class="driver-profile-box">
        <div style="font-size:28px;margin-bottom:4px;">{avatar}</div>
        <div class="profile-name">{profile.name}</div>
        <div class="profile-id">{profile.driver_id} · Category {profile.category}
             · Shift {profile.shift_start_time.strftime("%H:%M")}</div>
        <div class="hdivider"></div>
        <div class="metric-grid">
            <div class="metric-box"><div class="metric-val">{profile.mileage_band}</div>
                <div class="metric-lbl">Mileage</div></div>
            <div class="metric-box"><div class="metric-val">{total_trips}</div>
                <div class="metric-lbl">Total Trips</div></div>
            <div class="metric-box"><div class="metric-val">{fam_trips}</div>
                <div class="metric-lbl">This Route</div></div>
            <div class="metric-box"><div class="metric-val">{veh_type_trips}</div>
                <div class="metric-lbl">{vprofile.vehicle_type} Trips</div></div>
        </div>
        <div style="font-size:11px;color:#6b7280;line-height:1.8;">Crash History: {crash_txt}</div>
        <div class="hdivider"></div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:11px;color:#6b7280;">Incident Probability</span>
            <span style="font-family:'DM Mono',monospace;font-size:13px;
                         font-weight:600;color:#1d4ed8;">{live_prob*100:.3f}%</span>
        </div>
        <div style="margin-top:4px;text-align:right;">{risk_pill_html(live_level)}</div>
    </div>""", unsafe_allow_html=True)

    # P
    mil_m = MILEAGE_MULT[profile.mileage_band]
    mil_c = "good" if mil_m <= 0.95 else ("warn" if mil_m <= 1.1 else "danger")
    if fat_h <= 4:    fat_m = 1 + fat_h * 0.05
    elif fat_h <= 10: fat_m = 1 + 0.20 + (fat_h - 4) * 0.10
    else:             fat_m = 1 + 0.20 + 0.60 + (fat_h - 10) * 0.18
    fat_c = "danger" if fat_h > 10 else ("warn" if fat_h > 4 else "good")
    cr_c  = "danger" if profile.crash_history else "good"
    with st.expander("Driver", expanded=True):
        st.markdown(
            frow("Mileage Band",            f"{profile.mileage_band}  →  {mil_m:.2f}×", mil_c) +
            frow("Hours Since Shift Start", f"{fat_h:.1f} h  →  {fat_m:.2f}×",         fat_c) +
            frow("Crash History",           "Yes  →  1.50×" if profile.crash_history else "None  →  1.00×", cr_c) +
            frow("Driver Index P",          f"{live_risk['P']:.4f}"),
            unsafe_allow_html=True)

    # T
    pur_m  = PURPOSE_MULT[purpose_v]
    pur_c  = "danger" if purpose_v == "Emergency" else ("warn" if purpose_v == "Operation" else "")
    fam_m  = {"High": ROUTE_EXP_BONUS, "Medium": ROUTE_EXP_MED_MULT, "Low": ROUTE_EXP_LOW_MULT}[fam]
    fam_c  = "good" if fam == "High" else ("warn" if fam == "Medium" else "danger")
    veh_tc = "good" if veh_type_level == "High" else ("warn" if veh_type_level == "Medium" else "danger")
    with st.expander("Task", expanded=False):
        st.markdown(
            frow("Task Purpose",      f"{purpose_v}  →  {pur_m:.2f}×",                                           pur_c) +
            frow("Route Familiarity", f"R{rf.route_index+1}: {fam} ({fam_trips} trips)  →  {fam_m:.2f}×",        fam_c) +
            frow("Veh. Type Exp.",    f"{vprofile.vehicle_type}: {veh_type_level} ({veh_type_trips} trips)  →  {veh_type_m:.2f}×", veh_tc) +
            frow("Task Index T",      f"{live_risk['T']:.4f}"),
            unsafe_allow_html=True)

    # A
    vt_m     = VTYPE_MULT.get(vprofile.vehicle_type, 1.0)
    vt_c     = "danger" if vprofile.vehicle_type == "10T" else (
                "warn" if vprofile.vehicle_type in ["5T","Light Truck"] else "good")
    cond     = vehicle_ctx["condition"]
    cond_m   = MAINT_POOR_MULT if cond == "Poor" else (1.1 if cond == "Fair" else 1.0)
    cond_c   = "danger" if cond == "Poor" else ("warn" if cond == "Fair" else "good")
    td       = len(vprofile.safe_tech) * TECH_PENALTY
    last_svc = max(vprofile.maintenance_records).isoformat() if vprofile.maintenance_records else "N/A"
    bk_m     = weather_snap.braking_mult
    bk_c     = "danger" if bk_m >= 1.55 else ("warn" if bk_m > 1.0 else "good")
    with st.expander("Vehicle", expanded=False):
        st.markdown(
            frow("Vehicle",           f"{vprofile.vehicle_number} ({vprofile.vehicle_type})", "") +
            frow("Vehicle Age",       f"{vehicle_ctx['age']} yrs",
                 "danger" if vehicle_ctx["age"] >= 10 else ("warn" if vehicle_ctx["age"] >= 6 else "good")) +
            frow("Last Service",      last_svc, "good" if last_svc != "N/A" else "warn") +
            frow("Condition",         f"{cond}  →  {cond_m:.2f}×",                      cond_c) +
            frow("Vehicle Type",      f"{vprofile.vehicle_type}  →  {vt_m:.2f}×",       vt_c) +
            frow("Braking (weather)", f"{weather_snap.condition}  →  {bk_m:.2f}×",      bk_c) +
            frow("Safety Tech",       f"{', '.join(vprofile.safe_tech) or 'None'}  →  −{td:.0%}",
                 "good" if vprofile.safe_tech else "warn") +
            frow("Vehicle Index A",   f"{live_risk['A']:.4f}"),
            unsafe_allow_html=True)

    # E
    eff_vis = EnvironmentSim.effective_visibility_km(weather_snap, dep_time_v)
    vis_m   = live_risk["vis_m"]
    avg_tm  = live_risk["avg_traffic_m"]
    vis_c   = "danger" if vis_m >= 1.8 else ("warn" if vis_m >= 1.3 else "good")
    tod     = "Night" if env_ctx["is_night"] else "Day"
    tod_c   = "warn" if env_ctx["is_night"] else "good"
    tr_c    = "warn" if avg_tm >= 4/3 else "good"
    with st.expander("Environment", expanded=False):
        st.markdown(
            frow("Weather",          f"{weather_snap.condition}", "") +
            frow("Time of Day",      tod,                                                                  tod_c) +
            frow("Visibility mult",  f"{eff_vis:.1f} km effective  →  {vis_m:.2f}×",                     vis_c) +
            frow("Avg traffic mult", f"density {avg_tm*3:.1f}/5  →  {avg_tm:.2f}×",                      tr_c) +
            frow("Environment Index E", f"{live_risk['E']:.4f}"),
            unsafe_allow_html=True)

    # R
    h_c    = "danger" if rf.hotspot_count > 5 else ("warn" if rf.hotspot_count > 2 else "good")
    s_c    = "warn" if rf.sharp_turn_count > 3 else "good"
    n_c    = "warn" if rf.narrow_road_count > 3 else "good"
    rc_str = "  ".join(f"{k}: {v*100:.0f}%" for k, v in rf.road_class_breakdown.items())
    with st.expander("Roadway", expanded=False):
        st.markdown(
            frow("Hotspots",        f"{rf.hotspot_count} segments",     h_c) +
            frow("Sharp Turns",     f"{rf.sharp_turn_count} segments",  s_c) +
            frow("Narrow Roads",    f"{rf.narrow_road_count} segments", n_c) +
            frow("Road Classes",    rc_str or "N/A") +
            frow("Route Dist.",     f"{rf.dist_km:.2f} km") +
            frow("Roadway Index R", f"{live_risk['R']:.4f}"),
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALLOCATION EXPLANATION GENERATOR
# Translates backend risk numbers into plain-English justification cards.
# ══════════════════════════════════════════════════════════════════════════════

def _driver_selection_reason(asgn, all_assignments: list, score_matrix: dict = None) -> str:
    """
    One sentence explaining why this driver was selected.
    If the score matrix is available, compares the assigned risk against the
    best possible for this task to distinguish between:
      - Genuinely the best driver for this task
      - A deliberate fleet trade-off (better driver was needed elsewhere)
    """
    d    = asgn.driver_ctx["profile"]
    fat  = asgn.fatigue_at_dep
    prob = asgn.risk["prob"]

    parts = []

    # Experience
    veh_trips = asgn.driver_ctx["veh_type_trips"]
    vtype     = asgn.vehicle_ctx["profile"].vehicle_type
    if veh_trips >= VEH_TYPE_EXP_HIGH_THRESH:
        parts.append(f"highly experienced with {vtype} ({veh_trips} trips)")
    elif veh_trips >= VEH_TYPE_EXP_MED_THRESH:
        parts.append(f"moderately experienced with {vtype} ({veh_trips} trips)")
    else:
        parts.append(f"limited {vtype} experience ({veh_trips} trips)")

    # Fatigue
    if fat <= 2:
        parts.append("low fatigue at departure")
    elif fat <= 5:
        parts.append(f"moderate fatigue ({fat:.1f} h into shift)")
    else:
        parts.append(f"elevated fatigue ({fat:.1f} h into shift)")

    if d.crash_history:
        parts.append("has prior crash history")

    fam = asgn.driver_ctx["route_experience"]["familiarity"]
    if fam == "High":
        parts.append("familiar with this route")
    elif fam == "Low":
        parts.append("unfamiliar with this route")

    detail = "; ".join(parts) + "."

    # Determine qualifier from score matrix
    if score_matrix:
        best_possible = min(
            (v["prob"] for (tid, did, ridx), v in score_matrix.items()
             if tid == asgn.task.task_id),
            default=prob,
        )
        gap = prob - best_possible
        if gap < 0.005:
            qualifier = "Best available driver for this task — "
        else:
            qualifier = (f"Fleet trade-off — a better option existed for this task "
                         f"(best possible {best_possible*100:.3f}% vs assigned {prob*100:.3f}%), "
                         f"but was allocated elsewhere to protect the fleet. ")
    else:
        all_probs = [a.risk["prob"] for a in all_assignments]
        qualifier = "Best available — " if prob == min(all_probs) else ""

    return qualifier + detail


def _route_selection_reason(asgn) -> str:
    """One sentence on why this route index was chosen."""
    rf    = asgn.route_features
    ridx  = rf.route_index
    risk  = asgn.risk

    flags = []
    if rf.hotspot_count == 0:
        flags.append("no collision hotspots")
    elif rf.hotspot_count <= 2:
        flags.append(f"{rf.hotspot_count} hotspot segment{'s' if rf.hotspot_count>1 else ''}")
    else:
        flags.append(f"{rf.hotspot_count} hotspot segments")

    if rf.narrow_road_count == 0 and rf.sharp_turn_count == 0:
        flags.append("clear road geometry")
    else:
        if rf.narrow_road_count > 0:
            flags.append(f"{rf.narrow_road_count} narrow segment{'s' if rf.narrow_road_count>1 else ''}")
        if rf.sharp_turn_count > 0:
            flags.append(f"{rf.sharp_turn_count} sharp turn{'s' if rf.sharp_turn_count>1 else ''}")

    hw_breakdown = rf.road_class_breakdown
    dominant_hw  = max(hw_breakdown, key=hw_breakdown.get) if hw_breakdown else "mixed"

    return (f"Route {ridx + 1} selected ({rf.dist_km:.1f} km, "
            f"predominantly {dominant_hw.lower()} roads): "
            + ", ".join(flags) + ".")


def _risk_flags(asgn) -> list[tuple[str, str]]:
    """
    Return list of (severity, message) tuples for notable risk factors.
    severity: 'high' | 'medium' | 'low'
    """
    flags  = []
    risk   = asgn.risk
    d      = asgn.driver_ctx["profile"]
    vc     = asgn.vehicle_ctx
    env    = asgn.env_ctx
    fat    = asgn.fatigue_at_dep
    rf     = asgn.route_features
    vtype  = vc["profile"].vehicle_type

    # Fatigue
    if fat > 10:
        flags.append(("high",   f"⚠️ Critical fatigue: {fat:.1f} h since shift start"))
    elif fat > 6:
        flags.append(("medium", f"⚡ Elevated fatigue: {fat:.1f} h since shift start"))

    # Crash history
    if d.crash_history:
        flags.append(("medium", "🔴 Driver has prior crash history (+50% risk)"))

    # Vehicle condition
    cond = vc["condition"]
    if cond == "Poor":
        flags.append(("high",   f"⚠️ Vehicle in poor condition (+50% risk)"))
    elif cond == "Fair":
        flags.append(("medium", f"⚡ Vehicle in fair condition (+10% risk)"))

    # Weather
    weather = env["weather"]
    if weather.condition == "Heavy Rain":
        flags.append(("high",   f"⚠️ Heavy rain — reduced visibility and braking (+55% braking risk)"))
    elif weather.condition == "Rain":
        flags.append(("medium", f"⚡ Rainy conditions — reduced visibility (+25% braking risk)"))

    # Night
    if env["is_night"]:
        flags.append(("medium", "🌙 Night driving — visibility capped at 4 km"))

    # Hotspots
    if rf.hotspot_count > 5:
        flags.append(("high",   f"⚠️ Route passes {rf.hotspot_count} collision hotspot segments"))
    elif rf.hotspot_count > 2:
        flags.append(("medium", f"⚡ Route passes {rf.hotspot_count} collision hotspot segments"))

    # Vehicle type
    if vtype == "10T":
        flags.append(("medium", "🚛 10T vehicle — highest base risk category (2.0×)"))
    elif vtype == "5T":
        flags.append(("low",    "🚚 5T vehicle — elevated base risk category (1.5×)"))

    # Low route familiarity
    fam = asgn.driver_ctx["route_experience"]["familiarity"]
    if fam == "Low":
        flags.append(("low", "📍 Driver has low familiarity with this route (+35% task risk)"))

    # Purpose
    purpose = asgn.task.purpose
    if purpose == "Emergency":
        flags.append(("high",   "🚨 Emergency task — highest purpose multiplier (1.6×)"))
    elif purpose == "Operation":
        flags.append(("medium", "⚡ Operational task — elevated purpose multiplier (1.2×)"))

    return flags


def _fleet_insight(result) -> list[str]:
    """
    2–4 fleet-level observations about the overall allocation.
    For the bottleneck task, distinguishes between two cases:
      - Task is inherently hard (best possible driver still gives high risk)
      - Driver was a fleet-optimal trade-off (a better driver existed but
        was needed elsewhere to protect another task)
    """
    insights    = []
    assignments = result.assignments
    score_matrix = getattr(result, "score_matrix", {})
    if not assignments:
        return insights

    probs = [a.risk["prob"] for a in assignments]
    max_a = max(assignments, key=lambda a: a.risk["prob"])
    spread = max(probs) - min(probs)

    # ── Bottleneck analysis ───────────────────────────────────────────────────
    bt_task = max_a.task
    bt_prob = max_a.risk["prob"]

    # Find the best possible prob for this task across ALL (driver, route)
    # options in the score matrix, regardless of conflicts
    best_possible = min(
        (v["prob"] for (tid, did, ridx), v in score_matrix.items()
         if tid == bt_task.task_id),
        default=bt_prob,
    )
    assigned_prob = bt_prob
    gap = assigned_prob - best_possible

    # gap < 0.5% → assigned driver is essentially the best available
    # gap ≥ 0.5% → a better driver existed; was withheld for fleet reasons
    if gap < 0.005:
        bottleneck_note = (
            f"🎯 Bottleneck: {bt_task.task_id} ({bt_task.origin[:22]} → "
            f"{bt_task.destination[:22]}) at {bt_prob*100:.3f}%. "
            f"The assigned driver is the best available for this task "
            f"(best possible: {best_possible*100:.3f}%). "
            f"The risk is driven by the task's own conditions — vehicle, route, or environment."
        )
    else:
        bottleneck_note = (
            f"🎯 Bottleneck: {bt_task.task_id} ({bt_task.origin[:22]} → "
            f"{bt_task.destination[:22]}) at {bt_prob*100:.3f}%. "
            f"A better driver existed for this task (best possible: {best_possible*100:.3f}%), "
            f"but was assigned elsewhere to minimise the fleet's worst-case risk. "
            f"This is an intentional fleet-level trade-off."
        )
    insights.append(bottleneck_note)

    # ── Spread ────────────────────────────────────────────────────────────────
    if spread < 0.0005:
        insights.append("✅ Risk is well-balanced across all tasks — no single assignment is disproportionately dangerous.")
    elif spread > 0.005:
        insights.append(f"⚡ Large risk spread ({spread*100:.3f}%) between best and worst task — consider whether the bottleneck task can be rescheduled or replanned.")

    # ── Multi-task drivers ────────────────────────────────────────────────────
    driver_task_count: dict[str, int] = {}
    for a in assignments:
        did = a.driver_ctx["profile"].driver_id
        driver_task_count[did] = driver_task_count.get(did, 0) + 1
    multi = [(did, n) for did, n in driver_task_count.items() if n > 1]
    if multi:
        names = [f"{driver_db.get(did).name} ({n} tasks)" for did, n in multi]
        insights.append(f"👤 Multi-task drivers: {', '.join(names)}. Fatigue carry-forward has been applied to their later assignments.")

    # ── Unassigned ────────────────────────────────────────────────────────────
    if result.unassigned:
        insights.append(f"❌ {len(result.unassigned)} task(s) could not be assigned — no eligible driver was available without a schedule conflict.")

    return insights



left, middle, right = st.columns([1.1, 2.4, 1.3], gap="medium")

# ─────────────────────────────────────────────────────────────────────────────
# ███  SINGLE TASK MODE  (original behaviour, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.app_mode == "Single Task":

    # ── LEFT: task inputs ────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="panel-label">Task Configuration</div>', unsafe_allow_html=True)
        start_loc = st.text_input("Origin", "Orchard Road, Singapore")
        end_loc   = st.text_input("Destination", "Changi Airport, Singapore")
        c1, c2    = st.columns(2)
        with c1:  dep_time = st.time_input("Departure Time", value=time(10, 0))
        with c2:  purpose  = st.selectbox("Purpose", ["Admin", "Training", "Operation", "Emergency"])
        vehicle_options = {v.vehicle_number: f"{v.vehicle_number} ({v.vehicle_type})"
                           for v in vehicle_db.all()}
        vehicle_number  = st.selectbox("Vehicle", options=list(vehicle_options.keys()),
                                       format_func=lambda k: vehicle_options[k])
        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
        st.caption("Drivers pre-filtered by vehicle eligibility.")
        analyze_btn = st.button("🔍  Analyze Routes", type="primary", use_container_width=True)

    # ── Single-task analysis ─────────────────────────────────────────────────
    if analyze_btn:
        try:
            with st.spinner("Geocoding and computing routes…"):
                start_coords    = ox.geocoder.geocode(start_loc)
                end_coords      = ox.geocoder.geocode(end_loc)
                orig            = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
                dest            = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
                route_sequences = find_candidate_routes(DG, orig, dest)
                if not route_sequences:
                    st.error("No route found between the two locations.")
                    st.stop()
                task_ctx = build_task_context(
                    vehicle_number=vehicle_number, origin=start_loc,
                    destination=end_loc, purpose=purpose, departure_time=dep_time,
                    route_sequences=route_sequences, DG=DG,
                    driver_db=driver_db, vehicle_db=vehicle_db,
                    trip_log_db=trip_db, env_sim=env_sim,
                )
                task_ctx["purpose"]  = purpose
                task_ctx["dep_time"] = dep_time
                if not task_ctx["drivers"]:
                    st.error(f"No eligible drivers for {task_ctx['vehicle']['profile'].vehicle_type}.")
                    st.stop()
                all_rows = build_results(task_ctx, DG)
                top3     = top3_by_risk(all_rows)
            st.session_state.results   = top3
            st.session_state.sel_idx   = 0
            st.session_state.task_params = {
                "origin": start_loc, "destination": end_loc,
                "purpose": purpose, "dep_time": dep_time,
                "vehicle_number": vehicle_number,
            }
        except Exception as err:
            st.error(f"Analysis failed: {err}")
            raise

    # ── MIDDLE: results + map ────────────────────────────────────────────────
    with middle:
        st.markdown('<div class="panel-label">Risk Assessment Results</div>', unsafe_allow_html=True)
        if not st.session_state.results:
            st.markdown("""
            <div class="empty-state">
                <div style="font-size:44px;margin-bottom:10px;">🗺️</div>
                <div style="font-size:13px;font-weight:600;color:#6b7280;margin-bottom:4px;">No results yet</div>
                <div style="font-size:12px;">Configure a task and press <b>Analyze Routes</b>.</div>
            </div>""", unsafe_allow_html=True)
        else:
            results  = st.session_state.results
            computed = [compute_risk_for_result(r, DG) for r in results]
            st.markdown('<div class="section-label">Top Driver × Route Recommendations</div>',
                        unsafe_allow_html=True)
            for i, (res, risk) in enumerate(zip(results, computed)):
                p        = res["driver_ctx"]["profile"]
                rf       = res["route_features"]
                prob     = risk["prob"]
                lvl      = risk_category(prob)
                fam      = TripLogDB.per_route_familiarity(res["driver_ctx"]["route_experience"], rf.route_index)
                fat      = res["driver_ctx"]["fatigue_hours"]
                fam_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[fam]
                lvl_icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[lvl]
                selected = (st.session_state.sel_idx == i)
                border   = "2px solid #3b82f6" if selected else "1.5px solid #e2e5ea"
                bg       = "#eff6ff"           if selected else "#ffffff"
                col_info, col_pick = st.columns([11, 1])
                with col_info:
                    st.markdown(f"""
                    <div style="background:{bg};border:{border};border-radius:8px;
                                padding:11px 14px;margin-bottom:2px;
                                box-shadow:{'0 0 0 3px rgba(59,130,246,.08)' if selected else 'none'};">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:16px;font-weight:700;color:#111827;">
                                {RANK_EMOJI[i]}&nbsp;{p.name}
                            </span>
                            <span style="font-family:'DM Mono',monospace;font-size:12px;
                                         background:{'#dbeafe' if selected else '#f3f4f6'};
                                         color:{'#1d4ed8' if selected else '#6b7280'};
                                         padding:2px 8px;border-radius:20px;">
                                {lvl_icon} {lvl}
                            </span>
                        </div>
                        <div style="font-family:'DM Mono',monospace;font-size:12px;
                                    color:#6b7280;margin-top:4px;line-height:1.7;">
                            {p.driver_id} · Cat {p.category}
                            &nbsp;·&nbsp; Route {rf.route_index + 1}
                            &nbsp;·&nbsp; {rf.dist_km:.2f} km
                            &nbsp;·&nbsp; Since Shift: {fat:.1f} h
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col_pick:
                    btn_txt = "▶" if selected else "○"
                    if st.button(btn_txt, key=f"pick_{i}", help=f"Select {p.name}"):
                        st.session_state.sel_idx = i
                        st.rerun()
            st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label" style="margin-top:12px;">Route Map</div>',
                        unsafe_allow_html=True)
            sel_idx  = st.session_state.sel_idx
            sel_res  = results[sel_idx]
            sel_risk = computed[sel_idx]
            m = render_map(G, sel_res, sel_risk)
            st_folium(m, width=None, height=400, use_container_width=True, returned_objects=[])

    # ── RIGHT: driver profile + factors ──────────────────────────────────────
    with right:
        st.markdown('<div class="panel-label">Driver Profile & Factors</div>', unsafe_allow_html=True)
        if not st.session_state.results:
            st.markdown("""
            <div class="empty-state" style="padding:40px 10px;">
                <div style="font-size:11px;color:#9ca3af;">
                    Select a driver + route<br>to view derived factors.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            r = st.session_state.results[st.session_state.sel_idx]
            _render_right_panel(
                result_dict = r,
                DG          = DG,
                driver_ctx  = r["driver_ctx"],
                vehicle_ctx = r["vehicle_ctx"],
                env_ctx     = r["env_ctx"],
                rf          = r["route_features"],
                purpose_v   = r["purpose"],
                dep_time_v  = r["dep_time"],
            )


# ─────────────────────────────────────────────────────────────────────────────
# ███  BATCH ALLOCATION MODE
# ─────────────────────────────────────────────────────────────────────────────
else:

    # ── LEFT: task builder ───────────────────────────────────────────────────
    with left:
        st.markdown('<div class="panel-label">Batch Task Builder</div>', unsafe_allow_html=True)

        # ── Excel upload ──────────────────────────────────────────────────────
        with st.expander("📂  Upload task list from Excel", expanded=False):
            # Template download
            with open("task_template.xlsx", "rb") as f:
                template_bytes = f.read()
            st.download_button(
                "⬇️  Download template",
                data        = template_bytes,
                file_name   = "task_template.xlsx",
                mime        = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width = True,
            )
            st.caption("Fill the Tasks sheet and upload below. See OD Reference and Vehicle Reference sheets for valid values.")

            uploaded = st.file_uploader("Upload filled template (.xlsx)", type=["xlsx"],
                                         label_visibility="collapsed")
            if uploaded:
                try:
                    import pandas as pd
                    df = pd.read_excel(uploaded, sheet_name="Tasks", dtype=str)
                    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

                    # Required columns
                    required = {"task_id","node","origin","destination",
                                "departure_time","purpose","vehicle_number"}
                    missing  = required - set(df.columns)
                    if missing:
                        st.error(f"Missing columns: {', '.join(missing)}")
                    else:
                        # Drop blank/note rows
                        df = df.dropna(subset=["task_id","origin","destination",
                                                "departure_time","purpose","vehicle_number"])
                        df = df[~df["task_id"].str.startswith("←")]

                        errors  = []
                        parsed  = []
                        valid_veh  = {v.vehicle_number for v in vehicle_db.all()}
                        valid_purp = {"Admin","Training","Operation","Emergency"}
                        # Build valid O-D lookup (case-insensitive)
                        valid_ods  = {(e.origin.upper(), e.destination.upper())
                                      for e in route_db.all_valid()}

                        for _, row in df.iterrows():
                            tid   = str(row["task_id"]).strip()
                            orig  = str(row["origin"]).strip()
                            dest  = str(row["destination"]).strip()
                            dep   = str(row["departure_time"]).strip()
                            purp  = str(row["purpose"]).strip()
                            veh   = str(row["vehicle_number"]).strip()

                            # Validate departure time
                            try:
                                h, m = dep.split(":")
                                dep_time_obj = time(int(h), int(m))
                                dep_str = dep_time_obj.strftime("%H:%M")
                            except Exception:
                                errors.append(f"Row {tid}: invalid departure_time '{dep}' (use HH:MM)")
                                continue

                            # Validate purpose
                            if purp not in valid_purp:
                                errors.append(f"Row {tid}: invalid purpose '{purp}'")
                                continue

                            # Validate vehicle
                            if veh not in valid_veh:
                                errors.append(f"Row {tid}: vehicle '{veh}' not found")
                                continue

                            # Validate O-D
                            if (orig.upper(), dest.upper()) not in valid_ods:
                                errors.append(f"Row {tid}: O-D '{orig} → {dest}' not in route.json")
                                continue

                            parsed.append({
                                "task_id":        tid,
                                "origin":         orig,
                                "destination":    dest,
                                "departure_time": dep_str,
                                "purpose":        purp,
                                "vehicle_number": veh,
                            })

                        if errors:
                            for e in errors:
                                st.warning(e, icon="⚠️")
                        if parsed:
                            if st.button(f"➕  Add {len(parsed)} task(s) from file",
                                         use_container_width=True, type="primary"):
                                # Re-number task IDs to avoid duplicates
                                existing = len(st.session_state.batch_tasks)
                                for i, t in enumerate(parsed):
                                    t["task_id"] = f"T{existing + i + 1:03d}"
                                st.session_state.batch_tasks.extend(parsed)
                                st.rerun()
                        elif not errors:
                            st.info("No valid rows found in uploaded file.")

                except Exception as ex:
                    st.error(f"Failed to read file: {ex}")

        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

        # ── Manual add (existing) ─────────────────────────────────────────────
        st.markdown('<div class="section-label">Add Task Manually</div>', unsafe_allow_html=True)
        nodes_list = route_db.nodes()
        sel_node   = st.selectbox("Transport Node", nodes_list, key="ba_node")

        # O-D pairs for this node
        node_ods    = [(e.origin, e.destination)
                       for e in route_db.all_valid() if e.node == sel_node]
        od_labels   = [f"{o}  →  {d}" for o, d in node_ods]
        sel_od_idx  = st.selectbox("O-D Pair", range(len(od_labels)),
                                    format_func=lambda i: od_labels[i], key="ba_od")
        sel_origin, sel_dest = node_ods[sel_od_idx]

        c1, c2 = st.columns(2)
        with c1:  ba_dep  = st.time_input("Departure", value=time(8, 0), key="ba_dep")
        with c2:  ba_purp = st.selectbox("Purpose", ["Admin","Training","Operation","Emergency"], key="ba_purp")

        veh_opts   = {v.vehicle_number: f"{v.vehicle_number} ({v.vehicle_type})"
                      for v in vehicle_db.all()}
        ba_veh     = st.selectbox("Vehicle", list(veh_opts.keys()),
                                   format_func=lambda k: veh_opts[k], key="ba_veh")

        add_btn    = st.button("➕  Add Task", use_container_width=True)
        if add_btn:
            tid = f"T{len(st.session_state.batch_tasks)+1:03d}"
            st.session_state.batch_tasks.append({
                "task_id":        tid,
                "origin":         sel_origin,
                "destination":    sel_dest,
                "departure_time": ba_dep.strftime("%H:%M"),
                "purpose":        ba_purp,
                "vehicle_number": ba_veh,
            })
            st.rerun()

        # Task list with remove buttons
        st.markdown('<div class="section-label" style="margin-top:10px;">Task Queue</div>',
                    unsafe_allow_html=True)
        if not st.session_state.batch_tasks:
            st.caption("No tasks added yet.")
        else:
            for idx, t in enumerate(st.session_state.batch_tasks):
                col_t, col_rm = st.columns([10, 1])
                with col_t:
                    vtype = vehicle_db.get(t["vehicle_number"]).vehicle_type
                    st.markdown(
                        f'<div style="font-size:12px;padding:5px 0;line-height:1.6;">'
                        f'<b>{t["task_id"]}</b> · {t["departure_time"]} · {t["purpose"]}<br>'
                        f'<span style="color:#6b7280;">{t["origin"][:22]}… → {t["destination"][:22]}…</span><br>'
                        f'<span style="color:#6b7280;">{t["vehicle_number"]} ({vtype})</span>'
                        f'</div>', unsafe_allow_html=True)
                with col_rm:
                    if st.button("✕", key=f"rm_{idx}", help="Remove"):
                        st.session_state.batch_tasks.pop(idx)
                        st.rerun()

        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
        col_run, col_clr = st.columns(2)
        with col_run:
            run_btn = st.button("🚀  Allocate", type="primary",
                                use_container_width=True,
                                disabled=len(st.session_state.batch_tasks) == 0)
        with col_clr:
            if st.button("🗑  Clear All", use_container_width=True):
                st.session_state.batch_tasks  = []
                st.session_state.alloc_result = None
                st.rerun()

    # ── Allocation solver ────────────────────────────────────────────────────
    if run_btn and st.session_state.batch_tasks:
        progress_bar = st.progress(0, text="Initialising…")

        def _cb(msg, frac):
            progress_bar.progress(min(frac, 1.0), text=msg)

        try:
            tasks_for_alloc = []
            with st.spinner("Geocoding task locations…"):
                for t in st.session_state.batch_tasks:
                    dep = time(*[int(x) for x in t["departure_time"].split(":")])
                    # Geocode O-D to OSMnx node IDs
                    try:
                        sc = ox.geocoder.geocode(t["origin"] + ", Singapore")
                        ec = ox.geocoder.geocode(t["destination"] + ", Singapore")
                    except Exception:
                        sc = ox.geocoder.geocode("Singapore")
                        ec = ox.geocoder.geocode("Singapore")
                    orig_node = ox.distance.nearest_nodes(G, sc[1], sc[0])
                    dest_node = ox.distance.nearest_nodes(G, ec[1], ec[0])
                    route_seqs = find_candidate_routes(DG, orig_node, dest_node)
                    tasks_for_alloc.append(AllocTask(
                        task_id         = t["task_id"],
                        origin          = t["origin"],
                        destination     = t["destination"],
                        departure_time  = dep,
                        purpose         = t["purpose"],
                        vehicle_number  = t["vehicle_number"],
                        route_sequences = route_seqs,
                    ))

            allocator = Allocator(
                driver_db, vehicle_db, trip_db, env_sim, DG,
                compute_P_fn          = compute_P,
                compute_T_fn          = compute_T,
                compute_A_fn          = compute_A,
                compute_route_risk_fn = compute_route_risk,
            )
            result = allocator.solve(tasks_for_alloc, progress_cb=_cb)
            st.session_state.alloc_result  = result
            st.session_state.alloc_sel_idx = 0
            progress_bar.empty()
        except Exception as err:
            progress_bar.empty()
            st.error(f"Allocation failed: {err}")
            raise

    # ── MIDDLE: allocation results ───────────────────────────────────────────
    with middle:
        st.markdown('<div class="panel-label">Allocation Results</div>', unsafe_allow_html=True)
        result: AllocationResult = st.session_state.alloc_result

        if result is None:
            st.markdown("""
            <div class="empty-state">
                <div style="font-size:44px;margin-bottom:10px;">📋</div>
                <div style="font-size:13px;font-weight:600;color:#6b7280;margin-bottom:4px;">
                    No allocation yet
                </div>
                <div style="font-size:12px;">Add tasks and press <b>Allocate</b>.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # ── Fleet summary strip ───────────────────────────────────────────
            n_tasks    = len(result.assignments)
            n_unassign = len(result.unassigned)
            max_risk   = result.objective
            max_lvl    = risk_category(max_risk)
            avg_risk   = sum(a.risk["prob"] for a in result.assignments) / max(n_tasks, 1)
            lvl_col    = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}[max_lvl]

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:#111827;">{n_tasks}</div>
                <div style="font-size:11px;color:#6b7280;">Assigned</div>
              </div>
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:{'#dc2626' if n_unassign else '#111827'};">{n_unassign}</div>
                <div style="font-size:11px;color:#6b7280;">Unassigned</div>
              </div>
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:{lvl_col};">{max_risk*100:.3f}%</div>
                <div style="font-size:11px;color:#6b7280;">Max risk (objective)</div>
              </div>
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:#111827;">{avg_risk*100:.3f}%</div>
                <div style="font-size:11px;color:#6b7280;">Avg risk per task</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Warnings ──────────────────────────────────────────────────────
            if result.warnings:
                for w in result.warnings:
                    st.warning(w, icon="⚠️")

            # ── Assignment cards ──────────────────────────────────────────────
            st.markdown('<div class="section-label">Assignments</div>', unsafe_allow_html=True)
            for i, asgn in enumerate(result.assignments):
                task  = asgn.task
                d     = asgn.driver_ctx["profile"]
                rf    = asgn.route_features
                risk  = asgn.risk
                prob  = risk["prob"]
                lvl   = risk_category(prob)
                lvl_icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[lvl]
                selected = (st.session_state.alloc_sel_idx == i)
                border   = "2px solid #3b82f6" if selected else "1.5px solid #e2e5ea"
                bg       = "#eff6ff"           if selected else "#ffffff"
                vtype    = vehicle_db.get(task.vehicle_number).vehicle_type
                fat_warn = " ⚠️" if asgn.fatigue_at_dep > 6 else ""

                col_card, col_sel = st.columns([11, 1])
                with col_card:
                    st.markdown(f"""
                    <div style="background:{bg};border:{border};border-radius:8px;
                                padding:10px 14px;margin-bottom:2px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:13px;font-weight:600;color:#111827;">
                                {task.task_id} · {d.name} ({d.driver_id})
                            </span>
                            <span style="font-family:'DM Mono',monospace;font-size:12px;
                                         background:{'#dbeafe' if selected else '#f3f4f6'};
                                         color:{'#1d4ed8' if selected else '#6b7280'};
                                         padding:2px 8px;border-radius:20px;">
                                {lvl_icon} {lvl} · {prob*100:.3f}%
                            </span>
                        </div>
                        <div style="font-family:'DM Mono',monospace;font-size:11px;
                                    color:#6b7280;margin-top:3px;line-height:1.7;">
                            {task.origin[:28]} → {task.destination[:28]}<br>
                            {task.departure_time.strftime("%H:%M")} · {task.purpose}
                            · R{rf.route_index+1} · {rf.dist_km:.1f} km
                            · {task.vehicle_number} ({vtype})
                            · Fatigue {asgn.fatigue_at_dep:.1f}h{fat_warn}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col_sel:
                    btn_txt = "▶" if selected else "○"
                    if st.button(btn_txt, key=f"asel_{i}", help=f"View {task.task_id}"):
                        st.session_state.alloc_sel_idx = i
                        st.rerun()

            # ── Map for selected assignment ───────────────────────────────────
            if result.assignments:
                st.markdown('<div class="section-label" style="margin-top:12px;">Route Map</div>',
                            unsafe_allow_html=True)
                sel_asgn = result.assignments[st.session_state.alloc_sel_idx]
                map_result_dict = {
                    "route_features": sel_asgn.route_features,
                    "driver_ctx":     sel_asgn.driver_ctx,
                    "vehicle_ctx":    sel_asgn.vehicle_ctx,
                    "env_ctx":        sel_asgn.env_ctx,
                    "purpose":        sel_asgn.task.purpose,
                    "dep_time":       sel_asgn.task.departure_time,
                }
                m = render_map(G, map_result_dict, sel_asgn.risk)
                st_folium(m, width=None, height=400, use_container_width=True, returned_objects=[])

            # ── Driver workload table ─────────────────────────────────────────
            st.markdown('<div class="section-label" style="margin-top:12px;">Driver Workload</div>',
                        unsafe_allow_html=True)
            wl_rows = []
            for did, km in result.driver_workload().items():
                d       = driver_db.get(did)
                n_tasks = sum(1 for a in result.assignments
                              if a.driver_ctx["profile"].driver_id == did)
                wl_rows.append({
                    "Driver":   f"{d.name} ({did})",
                    "Cat":      d.category,
                    "Tasks":    n_tasks,
                    "Dist (km)":round(km, 1),
                })
            if wl_rows:
                st.dataframe(pd.DataFrame(wl_rows), use_container_width=True,
                             hide_index=True, height=min(38 + 35*len(wl_rows), 220))

    # ── RIGHT: selected assignment detail ────────────────────────────────────
    with right:
        st.markdown('<div class="panel-label">Assignment Detail</div>', unsafe_allow_html=True)
        result = st.session_state.alloc_result
        if result is None or not result.assignments:
            st.markdown("""
            <div class="empty-state" style="padding:40px 10px;">
                <div style="font-size:11px;color:#9ca3af;">
                    Run allocation to<br>view assignment details.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            sel_asgn    = result.assignments[st.session_state.alloc_sel_idx]
            driver_ctx  = sel_asgn.driver_ctx
            vehicle_ctx = sel_asgn.vehicle_ctx
            env_ctx     = sel_asgn.env_ctx
            rf          = sel_asgn.route_features
            purpose_v   = sel_asgn.task.purpose
            dep_time_v  = sel_asgn.task.departure_time

            # Patch fatigue with chained value before rendering
            driver_ctx_display = dict(driver_ctx)
            driver_ctx_display["fatigue_hours"] = sel_asgn.fatigue_at_dep

            _render_right_panel(
                result_dict = {
                    "route_features": rf,
                    "driver_ctx":     driver_ctx_display,
                    "vehicle_ctx":    vehicle_ctx,
                    "env_ctx":        env_ctx,
                    "purpose":        purpose_v,
                    "dep_time":       dep_time_v,
                },
                DG          = DG,
                driver_ctx  = driver_ctx_display,
                vehicle_ctx = vehicle_ctx,
                env_ctx     = env_ctx,
                rf          = rf,
                purpose_v   = purpose_v,
                dep_time_v  = dep_time_v,
            )

    # ── EXPLANATION PANEL (full width, below three columns) ──────────────────
    result = st.session_state.alloc_result
    if result and result.assignments:
        st.markdown("---")
        st.markdown('<div class="panel-label">Allocation Explanation</div>',
                    unsafe_allow_html=True)
        st.caption("Plain-English summary of why each driver and route was selected, key risk factors, and fleet-level observations.")

        # ── Fleet insights (full width) ───────────────────────────────────────
        insights = _fleet_insight(result)
        if insights:
            cols = st.columns(min(len(insights), 2))
            for i, insight in enumerate(insights):
                with cols[i % 2]:
                    st.markdown(
                        f'<div style="background:var(--background-color,#f8fafc);'
                        f'border:0.5px solid #e2e5ea;border-radius:8px;'
                        f'padding:10px 14px;margin-bottom:8px;font-size:12px;'
                        f'line-height:1.6;color:#374151;">{insight}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:12px;">Per-Task Justification</div>',
                    unsafe_allow_html=True)

        # ── Per-task justification cards ──────────────────────────────────────
        sev_color = {"high": "#fef2f2", "medium": "#fffbeb", "low": "#f0fdf4"}
        sev_border= {"high": "#fca5a5", "medium": "#fcd34d", "low": "#86efac"}
        sev_text  = {"high": "#991b1b", "medium": "#92400e", "low": "#166534"}

        n_cols = min(len(result.assignments), 3)
        cols   = st.columns(n_cols)

        for i, asgn in enumerate(result.assignments):
            with cols[i % n_cols]:
                task    = asgn.task
                d       = asgn.driver_ctx["profile"]
                prob    = asgn.risk["prob"]
                lvl     = risk_category(prob)
                lvl_col = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}[lvl]
                lvl_bg  = {"Low": "#f0fdf4", "Medium": "#fffbeb", "High": "#fef2f2"}[lvl]

                driver_reason = _driver_selection_reason(asgn, result.assignments,
                                                          getattr(result, "score_matrix", None))
                route_reason  = _route_selection_reason(asgn)
                flags         = _risk_flags(asgn)

                # Risk flag pills HTML
                flag_html = ""
                for sev, msg in flags:
                    flag_html += (
                        f'<div style="background:{sev_color[sev]};'
                        f'border:0.5px solid {sev_border[sev]};'
                        f'border-radius:6px;padding:4px 8px;margin-bottom:4px;'
                        f'font-size:11px;color:{sev_text[sev]};line-height:1.5;">'
                        f'{msg}</div>')

                if not flag_html:
                    flag_html = ('<div style="font-size:11px;color:#6b7280;">'
                                 '✅ No significant risk flags.</div>')

                st.markdown(f"""
                <div style="border:1px solid {lvl_col}33;border-radius:10px;
                            padding:14px;margin-bottom:10px;background:{lvl_bg};">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:8px;">
                        <span style="font-size:13px;font-weight:600;color:#111827;">
                            {task.task_id} · {d.name}
                        </span>
                        <span style="font-size:12px;font-weight:500;
                                     color:{lvl_col};">{lvl} · {prob*100:.3f}%</span>
                    </div>
                    <div style="font-size:11px;color:#6b7280;margin-bottom:8px;">
                        {task.origin[:30]} → {task.destination[:30]}<br>
                        {task.departure_time.strftime("%H:%M")} · {task.purpose}
                        · {task.vehicle_number}
                    </div>
                    <div style="font-size:12px;color:#374151;margin-bottom:6px;">
                        <b>Driver:</b> {driver_reason}
                    </div>
                    <div style="font-size:12px;color:#374151;margin-bottom:8px;">
                        <b>Route:</b> {route_reason}
                    </div>
                    <div style="font-size:11px;font-weight:500;color:#6b7280;
                                margin-bottom:4px;">Risk factors:</div>
                    {flag_html}
                </div>""", unsafe_allow_html=True)