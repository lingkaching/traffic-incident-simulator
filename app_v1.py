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
from data_layer import (
    DriverDB, VehicleDB, TripLogDB, EnvironmentSim,
    RouteFeatureExtractor, build_task_context,
    VEHICLE_ELIGIBILITY, ROAD_CLASS_LABELS,
)

st.set_page_config(layout="wide", page_title="Fleet Risk Command", page_icon="🛡️")

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
ROUTE_EXP_THRESHOLD = 10   # total trips — High familiarity
ROUTE_EXP_BONUS    = 0.85  # multiplier if High familiarity
ROUTE_EXP_MED_MULT = 0.95  # Medium familiarity
ROUTE_EXP_LOW_MULT = 1.20  # Low familiarity
VEH_EXP_PENALTY    = 1.15  # never driven this type before
PURPOSE_EXP_THRESHOLD = 20 # task_exp trips for general purpose experience
PURPOSE_EXP_BONUS  = 0.9
PURPOSE_EXP_PENALTY= 1.2
MAINT_POOR_MULT    = 1.3
TECH_PENALTY       = 0.05
NIGHT_VIS_MULT     = 1.4
NARROW_ROAD_MULT   = 1.3
SHARP_TURN_MULT    = 1.4
HOTSPOT_MULT_VAL   = 2.5

# ══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE FUNCTIONS
# All inputs are typed data-layer objects — no raw dicts from old design
# ══════════════════════════════════════════════════════════════════════════════

def risk_category(prob: float) -> str:
    if prob < 0.0002:  return "Low"
    if prob < 0.0005:  return "Medium"
    return "High"

def risk_pill_html(level: str) -> str:
    cls = {"Low": "pill-low", "Medium": "pill-med", "High": "pill-high"}[level]
    return f'<span class="risk-pill {cls}">{level}</span>'

def frow(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="factor-row">'
            f'<span class="factor-label">{label}</span>'
            f'<span class="factor-value {cls}">{value}</span>'
            f'</div>')

# ── Index P : Driver ──────────────────────────────────────────────────────────
def compute_P(driver_ctx: dict) -> float:
    """P = mileage × crash × fatigue (from shift_start → departure_time)."""
    profile = driver_ctx["profile"]
    fatigue = driver_ctx["fatigue_hours"]
    return (MILEAGE_MULT[profile.mileage_band]
            * (CRASH_MULT if profile.crash_history else 1.0)
            * (1 + fatigue * FATIGUE_RATE))

# ── Index T : Task ────────────────────────────────────────────────────────────
def compute_T(driver_ctx: dict, purpose: str, route_idx: int) -> float:
    """
    T = purpose × route_familiarity(route_idx) × general_exp × vehicle_exp

    route_familiarity is per-route-index: how many times the driver has
    specifically taken route 0, 1, or 2 between this O-D pair.
    This is what differentiates T across the 3 candidate routes for the same driver.
    """
    route_exp  = driver_ctx["route_experience"]
    has_driven = driver_ctx["has_driven_type"]

    # General task experience: total trips ever as proxy for overall driving maturity
    total_trips = sum(driver_ctx["vehicle_type_exp"].values())
    exp_mult    = PURPOSE_EXP_BONUS if total_trips > PURPOSE_EXP_THRESHOLD else PURPOSE_EXP_PENALTY

    # Per-route familiarity — uses route_idx to pick the right bucket
    route_fam  = TripLogDB.per_route_familiarity(route_exp, route_idx)
    fam_mult   = {"High": ROUTE_EXP_BONUS,
                  "Medium": ROUTE_EXP_MED_MULT,
                  "Low":    ROUTE_EXP_LOW_MULT}[route_fam]

    # Vehicle type experience
    veh_mult = 1.0 if has_driven else VEH_EXP_PENALTY

    return PURPOSE_MULT[purpose] * exp_mult * fam_mult * veh_mult

# ── Index A : Vehicle ─────────────────────────────────────────────────────────
def compute_A(vehicle_ctx: dict, weather: "WeatherSnapshot") -> float:
    """
    A = vehicle_type × condition × safety_tech_discount × braking_degradation
    braking_degradation comes from weather (wet/flooded roads).
    """
    profile   = vehicle_ctx["profile"]
    condition = vehicle_ctx["condition"]
    tech_disc = len(profile.safe_tech) * TECH_PENALTY
    return (VTYPE_MULT.get(profile.vehicle_type, 1.0)
            * (MAINT_POOR_MULT if condition == "Poor" else (1.1 if condition == "Fair" else 1.0))
            * (1 - tech_disc)
            * weather.braking_mult)

# ── Index E : Environment ─────────────────────────────────────────────────────
def compute_E(env_ctx: dict, route_features, dep_time: "time") -> float:
    """
    E = visibility_mult × avg_traffic_density_mult

    visibility_mult   = f(weather condition, day/night) via EnvironmentSim.visibility_mult
    avg_traffic       = per-segment densities already include weather degradation
                        (baked in by EnvironmentSim.query_traffic at data-layer time)

    Note: weather braking effect is captured in A, not here.
    """
    weather_snap = env_ctx["weather"]
    vis_m        = EnvironmentSim.visibility_mult(weather_snap, dep_time)

    segs         = route_features.segment_traffic
    avg_density  = (sum(s.density for s in segs) / len(segs)) if segs else 3.0
    traffic_m    = avg_density / 3.0

    return vis_m * traffic_m

# ── Per-segment roadway multiplier ────────────────────────────────────────────
def roadway_mult_for_edge(edge: dict) -> float:
    mult  = edge.get("hotspot_mult", 1.0)
    mult *= ROAD_MULT.get(edge.get("highway", "residential"), 1.5)
    if edge.get("is_narrow"): mult *= NARROW_ROAD_MULT
    if edge.get("is_sharp"):  mult *= SHARP_TURN_MULT
    return mult

# ── Full route risk ────────────────────────────────────────────────────────────
def compute_route_risk(route_features, P: float, T: float, A: float, E: float, DG) -> dict:
    """
    Walk every edge of the route, compute per-segment lambda, return risk dict.
    """
    nodes         = route_features.node_sequence
    total_lambda  = 0.0
    segment_data  = []          # (u, v, risk_per_meter)
    hotspot_coords= []

    for u, v in zip(nodes[:-1], nodes[1:]):
        edge   = DG.get_edge_data(u, v) or {}
        length = edge.get("length", 0.0)
        R      = roadway_mult_for_edge(edge)

        if edge.get("hotspot_mult", 1.0) > 1.0:
            hotspot_coords.append((DG.nodes[u]["y"], DG.nodes[u]["x"]))

        lam = BASE_LAMBDA * length * P * T * A * E * R
        total_lambda += lam
        segment_data.append((u, v, lam / max(length, 1)))

    return {
        "prob":           1 - math.exp(-total_lambda),
        "dist_km":        route_features.dist_km,
        "segments":       segment_data,       # (u, v, risk_per_meter)
        "hotspot_coords": hotspot_coords,
        "P": P, "T": T, "A": A, "E": E,
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
    trip_db    = TripLogDB(driver_db, vehicle_db)
    env_sim    = EnvironmentSim(seed=42)
    print("\n=== Route experience D001: Orchard Road → Changi Airport ===")
    for driver_id in ['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015']:
        print(driver_id)
        print(trip_db.route_experience(driver_id, "Orchard Road, Singapore", "Changi Airport, Singapore"))
    return driver_db, vehicle_db, trip_db, env_sim

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS = {
    "results":    None,
    "sel_idx":    0,
    "task_params": {},
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
            E    = compute_E(env_ctx, rf, dep_time)
            risk = compute_route_risk(rf, P, T, A, E, DG)
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
    E = compute_E(env_ctx, rf, dep_time)
    return compute_route_risk(rf, P, T, A, E, DG)

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
        <b>Risk Level</b><br>
        <span style="color:#16a34a;">●</span> Low &nbsp;
        <span style="color:#d97706;">●</span> Medium &nbsp;
        <span style="color:#dc2626;">●</span> High<br>
        <span>⚠️</span> Collision Hotspot
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    for u, v, rpm in risk["segments"]:
        color = "#16a34a" if rpm < 2e-6 else "#d97706" if rpm < 5e-6 else "#dc2626"
        pts   = [[G_raw.nodes[u]["y"], G_raw.nodes[u]["x"]],
                 [G_raw.nodes[v]["y"], G_raw.nodes[v]["x"]]]
        folium.PolyLine(pts, color=color, weight=5, opacity=0.85).add_to(m)

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

# ══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading road network and data layer…"):
    G, DG = load_graph()
    driver_db, vehicle_db, trip_db, env_sim = get_data_layer()

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, middle, right = st.columns([1.1, 2.4, 1.3], gap="medium")

# ──────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — Task Inputs
# ──────────────────────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="panel-label">Task Configuration</div>', unsafe_allow_html=True)

    # st.markdown('<div class="section-label">Task</div>', unsafe_allow_html=True)
    start_loc = st.text_input("Origin", "Orchard Road, Singapore")
    # st.caption("ORIGIN")
    end_loc   = st.text_input("Destination", "Changi Airport, Singapore")
    # st.caption("DESTINATION")

    c1, c2 = st.columns(2)
    with c1:
        dep_time = st.time_input("Departure Time", value=time(10, 0))
    with c2:
        purpose  = st.selectbox("Purpose", ["Admin", "Training", "Operation", "Emergency"])

    # st.markdown('<div class="section-label">Vehicle</div>', unsafe_allow_html=True)
    # Show vehicle numbers with type hint
    vehicle_options = {v.vehicle_number: f"{v.vehicle_number} ({v.vehicle_type})"
                       for v in vehicle_db.all()}
    vehicle_number = st.selectbox(
        "Vehicle", options=list(vehicle_options.keys()),
        format_func=lambda k: vehicle_options[k],
    )

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
    st.caption("Drivers pre-filtered by vehicle eligibility.")

    analyze_btn = st.button("🔍  Analyze Routes", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
if analyze_btn:
    try:
        with st.spinner("Geocoding and computing routes…"):
            start_coords = ox.geocoder.geocode(start_loc)
            end_coords   = ox.geocoder.geocode(end_loc)
            orig = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
            dest = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

            route_sequences = find_candidate_routes(DG, orig, dest)
            if not route_sequences:
                st.error("No route found between the two locations.")
                st.stop()

            # Build full task context via data layer
            task_ctx = build_task_context(
                vehicle_number  = vehicle_number,
                origin          = start_loc,
                destination     = end_loc,
                purpose         = purpose,
                departure_time  = dep_time,
                route_sequences = route_sequences,
                DG              = DG,
                driver_db       = driver_db,
                vehicle_db      = vehicle_db,
                trip_log_db     = trip_db,
                env_sim         = env_sim,
            )
            # Attach purpose and dep_time so build_results can use them
            task_ctx["purpose"]  = purpose
            task_ctx["dep_time"] = dep_time

            if not task_ctx["drivers"]:
                st.error(f"No eligible drivers for vehicle type {task_ctx['vehicle']['profile'].vehicle_type}.")
                st.stop()

            all_rows = build_results(task_ctx, DG)
            top3     = top3_by_risk(all_rows)

        st.session_state.results     = top3
        st.session_state.sel_idx     = 0
        st.session_state.task_params = {
            "origin": start_loc, "destination": end_loc,
            "purpose": purpose, "dep_time": dep_time,
            "vehicle_number": vehicle_number,
        }

    except Exception as err:
        st.error(f"Analysis failed: {err}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# MIDDLE PANEL
# ──────────────────────────────────────────────────────────────────────────────
RANK_LABEL = ["#1", "#2", "#3"]
RANK_EMOJI = ["🥇",  "🥈",   "🥉"]
DRIVER_AVATARS = ["👨‍✈️", "👩‍✈️", "🧑‍✈️", "👨‍🔧", "👩‍🔧"]

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
        results = st.session_state.results
        computed = [compute_risk_for_result(r, DG) for r in results]

        # ── Selection list ────────────────────────────────────────────────────
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
                        <span style="font-size:14px;font-weight:700;color:#111827;">
                            {RANK_EMOJI[i]}&nbsp;{p.name}
                        </span>
                        <span style="font-family:'DM Mono',monospace;font-size:10px;
                                     background:{'#dbeafe' if selected else '#f3f4f6'};
                                     color:{'#1d4ed8' if selected else '#6b7280'};
                                     padding:2px 8px;border-radius:20px;">
                            {lvl_icon} {lvl}
                        </span>
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:11px;
                                color:#6b7280;margin-top:4px;line-height:1.7;">
                        {p.driver_id} · Cat {p.category}
                        &nbsp;·&nbsp; Route {rf.route_index + 1}
                        &nbsp;·&nbsp; {rf.dist_km:.2f} km
                        &nbsp;·&nbsp; Fatigue {fat:.1f} h
                        &nbsp;·&nbsp; {fam_icon} {fam} route familiarity
                    </div>
                </div>""", unsafe_allow_html=True)
            with col_pick:
                # Small select button — visually part of the card row
                btn_txt = "▶" if selected else "○"
                if st.button(btn_txt, key=f"pick_{i}", help=f"Select {p.name}"):
                    st.session_state.sel_idx = i
                    st.rerun()

        st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

        # # ── Comparison table ─────────────────────────────────────────────────
        # st.markdown('<div class="section-label" style="margin-top:8px;">Comparison Table</div>',
        #             unsafe_allow_html=True)
        # rows_tbl = []
        # for i, (res, risk) in enumerate(zip(results, computed)):
        #     p   = res["driver_ctx"]["profile"]
        #     rf  = res["route_features"]
        #     fam = TripLogDB.per_route_familiarity(res["driver_ctx"]["route_experience"], rf.route_index)
        #     rows_tbl.append({
        #         "Rank":        f"{RANK_EMOJI[i]} {RANK_LABEL[i]}",
        #         "Driver":      p.name,
        #         "Cat":         p.category,
        #         "Route":       f"R{rf.route_index + 1}",
        #         "Dist (km)":   f"{rf.dist_km:.2f}",
        #         # "P(Incident)": f"{risk['prob']*100:.3f}%",
        #         "Risk":        risk_category(risk["prob"]),
        #         "Route Familiarity": fam,
        #         "Veh Exp":     "✓" if res["driver_ctx"]["has_driven_type"] else "✗",
        #         "Fatigue (h)": f"{res['driver_ctx']['fatigue_hours']:.1f}",
        #         "Hotspots":    rf.hotspot_count,
        #     })
        # st.dataframe(pd.DataFrame(rows_tbl).set_index("Rank"), use_container_width=True, height=158)

        # ── Route map ────────────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:12px;">Route Map</div>',
                    unsafe_allow_html=True)
        sel_idx  = st.session_state.sel_idx
        sel_res  = results[sel_idx]
        sel_risk = computed[sel_idx]
        m = render_map(G, sel_res, sel_risk)
        st_folium(m, width=None, height=400, use_container_width=True, returned_objects=[])

# ──────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — Driver Profile + All Derived Factors
# ──────────────────────────────────────────────────────────────────────────────
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
        sel_res     = st.session_state.results[st.session_state.sel_idx]
        driver_ctx  = sel_res["driver_ctx"]
        vehicle_ctx = sel_res["vehicle_ctx"]
        env_ctx     = sel_res["env_ctx"]
        rf          = sel_res["route_features"]
        profile     = driver_ctx["profile"]
        vprofile    = vehicle_ctx["profile"]
        purpose_v   = sel_res["purpose"]
        dep_time_v  = sel_res["dep_time"]
        weather_snap = env_ctx["weather"]         # ← defined here, used throughout

        live_risk  = compute_risk_for_result(sel_res, DG)
        live_prob  = live_risk["prob"]
        live_level = risk_category(live_prob)

        # ── Driver profile box ───────────────────────────────────────────────
        avatar      = DRIVER_AVATARS[int(profile.driver_id[1:]) % len(DRIVER_AVATARS)]
        crash_txt   = "🔴 Yes" if profile.crash_history else "🟢 None"
        fam         = TripLogDB.per_route_familiarity(driver_ctx["route_experience"], rf.route_index)
        fam_trips   = driver_ctx["route_experience"]["by_route"].get(rf.route_index, 0)
        od_trips    = driver_ctx["route_experience"]["total_trips"]
        veh_exp     = driver_ctx["has_driven_type"]
        total_trips = sum(driver_ctx["vehicle_type_exp"].values())
        fat_h       = driver_ctx["fatigue_hours"]

        st.markdown(f"""
        <div class="driver-profile-box">
            <div style="font-size:28px;margin-bottom:4px;">{avatar}</div>
            <div class="profile-name">{profile.name}</div>
            <div class="profile-id">{profile.driver_id} · Category {profile.category}
                 · Shift {profile.shift_start_time.strftime("%H:%M")}</div>
            <div class="hdivider"></div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-val">{profile.mileage_band}</div>
                    <div class="metric-lbl">Mileage</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{total_trips}</div>
                    <div class="metric-lbl">Total Trips</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{fam_trips}</div>
                    <div class="metric-lbl">This Route</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{'✓' if veh_exp else '✗'}</div>
                    <div class="metric-lbl">Veh. Exp.</div>
                </div>
            </div>
            <div style="font-size:11px;color:#6b7280;line-height:1.8;">
                Crash History: {crash_txt}
            </div>
            <div class="hdivider"></div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:11px;color:#6b7280;">Incident Probability</span>
                <span style="font-family:'DM Mono',monospace;font-size:13px;
                             font-weight:600;color:#1d4ed8;">{live_prob*100:.3f}%</span>
            </div>
            <div style="margin-top:4px;text-align:right;">{risk_pill_html(live_level)}</div>
        </div>""", unsafe_allow_html=True)

        # ── DRIVER FACTORS ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Driver Factors (P)</div>', unsafe_allow_html=True)
        mil_m = MILEAGE_MULT[profile.mileage_band]
        mil_c = "good" if mil_m <= 0.95 else ("warn" if mil_m <= 1.1 else "danger")
        fat_m = 1 + fat_h * FATIGUE_RATE
        fat_c = "danger" if fat_h > 8 else ("warn" if fat_h > 4 else "good")
        cr_c  = "danger" if profile.crash_history else "good"
        P_val = live_risk["P"]
        st.markdown(
            frow("Mileage Band",   f"{profile.mileage_band}  →  {mil_m:.2f}×", mil_c) +
            frow("Fatigue",        f"{fat_h:.1f} h (shift {profile.shift_start_time.strftime('%H:%M')})  →  {fat_m:.2f}×", fat_c) +
            frow("Crash History",  "Yes  →  1.50×" if profile.crash_history else "None  →  1.00×", cr_c) +
            frow("Driver Index P", f"{P_val:.4f}"),
            unsafe_allow_html=True,
        )

        # ── TASK FACTORS ─────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Task Factors (T)</div>', unsafe_allow_html=True)
        pur_m = PURPOSE_MULT[purpose_v]
        pur_c = "danger" if purpose_v == "Emergency" else ("warn" if purpose_v == "Operation" else "")
        fam_m = {"High": ROUTE_EXP_BONUS, "Medium": ROUTE_EXP_MED_MULT, "Low": ROUTE_EXP_LOW_MULT}[fam]
        fam_c = "good" if fam == "High" else ("warn" if fam == "Medium" else "danger")
        exp_m = PURPOSE_EXP_BONUS if total_trips > PURPOSE_EXP_THRESHOLD else PURPOSE_EXP_PENALTY
        exp_c = "good" if total_trips > PURPOSE_EXP_THRESHOLD else "warn"
        veh_m = 1.0 if veh_exp else VEH_EXP_PENALTY
        veh_c = "good" if veh_exp else "warn"
        T_val = live_risk["T"]
        st.markdown(
            frow("Task Purpose",      f"{purpose_v}  →  {pur_m:.2f}×",               pur_c) +
            frow("Route Familiarity", f"R{rf.route_index+1}: {fam} ({fam_trips} trips)  →  {fam_m:.2f}×", fam_c) +
            frow("O-D Total Trips",   f"{od_trips} trips across all routes",                              exp_c) +
            frow("Vehicle Exp.",      f"{'Yes' if veh_exp else 'No'}  →  {veh_m:.2f}×", veh_c) +
            frow("Task Index T",      f"{T_val:.4f}"),
            unsafe_allow_html=True,
        )

        # ── VEHICLE FACTORS ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Vehicle Factors (V)</div>', unsafe_allow_html=True)
        vt_m      = VTYPE_MULT.get(vprofile.vehicle_type, 1.0)
        vt_c      = "danger" if vprofile.vehicle_type == "10T" else (
                     "warn"   if vprofile.vehicle_type in ["5T", "Light Truck"] else "good")
        cond      = vehicle_ctx["condition"]
        cond_m    = MAINT_POOR_MULT if cond == "Poor" else (1.1 if cond == "Fair" else 1.0)
        cond_c    = "danger" if cond == "Poor" else ("warn" if cond == "Fair" else "good")
        td        = len(vprofile.safe_tech) * TECH_PENALTY
        last_svc  = max(vprofile.maintenance_records).isoformat() if vprofile.maintenance_records else "N/A"
        braking_m = weather_snap.braking_mult
        braking_c = "danger" if braking_m >= 1.55 else ("warn" if braking_m > 1.0 else "good")
        A_val     = live_risk["A"]
        st.markdown(
            frow("Vehicle",          f"{vprofile.vehicle_number} ({vprofile.vehicle_type})", "") +
            frow("Vehicle Age",      f"{vehicle_ctx['age']} yrs",
                 "danger" if vehicle_ctx["age"] >= 10 else ("warn" if vehicle_ctx["age"] >= 6 else "good")) +
            frow("Last Service",     last_svc, "good" if last_svc != "N/A" else "warn") +
            frow("Condition",        f"{cond}  →  {cond_m:.2f}×",                         cond_c) +
            frow("Vehicle Type",     f"{vprofile.vehicle_type}  →  {vt_m:.2f}×",          vt_c) +
            frow("Braking (weather)",f"{weather_snap.condition}  →  {braking_m:.2f}×",    braking_c) +
            frow("Safety Tech",      f"{', '.join(vprofile.safe_tech) or 'None'}  →  −{td:.0%}",
                 "good" if vprofile.safe_tech else "warn") +
            frow("Vehicle Index A",  f"{A_val:.4f}"),
            unsafe_allow_html=True,
        )

        # ── ENVIRONMENT FACTORS ───────────────────────────────────────────────
        st.markdown('<div class="section-label">Environment Factors (E)</div>', unsafe_allow_html=True)
        eff_vis = EnvironmentSim.effective_visibility_km(weather_snap, dep_time_v)
        vis_m   = EnvironmentSim.visibility_mult(weather_snap, dep_time_v)
        vis_c   = "danger" if vis_m >= 1.8 else ("warn" if vis_m >= 1.3 else "good")
        tod     = "Night" if env_ctx["is_night"] else "Day"
        tod_c   = "warn" if env_ctx["is_night"] else "good"
        segs    = rf.segment_traffic
        avg_dens = (sum(s.density for s in segs) / len(segs)) if segs else 3.0
        tr_c    = "warn" if avg_dens >= 4 else "good"
        E_val   = live_risk["E"]
        st.markdown(
            frow("Time of Day",      tod,                                                           tod_c) +
            frow("Weather",          f"{weather_snap.condition}  ({weather_snap.visibility_km:.1f} km daytime vis)", "") +
            frow("Visibility",       f"{eff_vis:.1f} km effective  →  {vis_m:.2f}×",              vis_c) +
            frow("Avg Traffic",      f"{avg_dens:.1f}/5 (weather-adjusted)  →  {avg_dens/3:.2f}×", tr_c) +
            frow("Environ. Index E", f"{E_val:.4f}"),
            unsafe_allow_html=True,
        )

        # ── ROADWAY FACTORS ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Roadway Factors (R)</div>', unsafe_allow_html=True)
        h_c    = "danger" if rf.hotspot_count > 5 else ("warn" if rf.hotspot_count > 2 else "good")
        s_c    = "warn" if rf.sharp_turn_count > 3 else "good"
        n_c    = "warn" if rf.narrow_road_count > 3 else "good"
        rc_str = "  ".join(f"{k}: {v*100:.0f}%" for k, v in rf.road_class_breakdown.items())
        st.markdown(
            frow("Hotspots",     f"{rf.hotspot_count} segments",     h_c) +
            frow("Sharp Turns",  f"{rf.sharp_turn_count} segments",  s_c) +
            frow("Narrow Roads", f"{rf.narrow_road_count} segments", n_c) +
            frow("Road Classes", rc_str or "N/A") +
            frow("Route Dist.",  f"{rf.dist_km:.2f} km"),
            unsafe_allow_html=True,
        )
