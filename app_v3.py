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
from datetime import date, datetime, time, timedelta
from typing import Optional

# ── data layer ────────────────────────────────────────────────────────────────
from data_layer_v2 import (
    DriverDB, VehicleDB, TripLogDB, EnvironmentSim,
    RouteFeatureExtractor, RouteDB, build_task_context,
    VEHICLE_ELIGIBILITY, ROAD_CLASS_LABELS, estimate_duration_h,
)
# ── geocoder ──────────────────────────────────────────────────────────────────
from geocoder import geocode as onemap_geocode, geocode_input, get_cache, geocode_with_candidates

# ── Location alias map ────────────────────────────────────────────────────────
# Add entries here when OneMap cannot find a location name from your task file.
# Keys are matched case-insensitively to the name in the Excel/CSV.
LOCATION_ALIASES: dict[str, str] ={'PASIR LABAR CAMP': 'PASIR LABA CAMP',
 'LIM CHU KANG TP 11': 'LIM CHU KANG ROAD',
 'LOR ASRAMA': 'MANDAI CAMP II',
 'PLAB': 'Paya Lebar Air Base',
 'SKGH': 'SENGKANG GENERAL HOSPITAL',
 'SENG KANG HOSPITAL': 'SENGKANG GENERAL HOSPITAL',
 'KTPH': 'Khoo Teck Puat Hospital',
 'LIM CHU KANG TP 7': 'LIM CHU KANG ROAD',
 'LIM CHU KANG TP 4': 'LIM CHU KANG ROAD',
 'SKH': 'SENGKANG GENERAL HOSPITAL',
 'OTH': 'Our Tampines Hub',
 'KKH': "KK Women's and Children's Hospital",
 'KRANJI CAMP 3': 'KRANJI CAMP',
 'PYLMC': 'Paya Lebar Air Base',
 'KAKI BUIKIT CAMP': 'KAKI BUKIT CAMP',
 'MUTF': 'MURAI CAMP',
 'SAFTI MI': 'SAFTI Military Institute',
 'LORONG ASRAMA': 'MANDAI CAMP II',
 'TENGAH AIRBASE': 'TENGAH AIR BASE',
 'JURONG CAMP 1': 'JURONG CAMP',
 'LIM CHU KANG LIM CHU KANG TP 7': 'LIM CHU KANG ROAD',
 'SEMBAWANG CQAMP': 'SEMBAWANG CAMP',
 'SUNGEI GEONG CAMP': 'SUNGEI GEDONG CAMP',
 'SAFTI TO CAMP': 'SAFTI Military Institute',
 'CLEMENTI TO SAF FERRY TERMINAL': 'CLEMENTI CAMP',
 'SEMBAWANG WHARVES': 'SEMBAWANG PARK',
 'JURONG CAMP 2': 'JURONG CAMP',
 'LONBR ASRAMA': 'MANDAI CAMP II',
 'CCDC': 'MANDAI CAMP II',
 'CCDC TO NEE': 'MANDAI CAMP II',
 'EBUA TO TENGAH': 'MANDAI CAMP II',
 'GEDONG TO SARIMBUN': 'Sungei Gedong Camp',
 'NEESOON CAMP': 'NEE SOON CAMP',
 'POYAN': 'LIM CHU KANG ROAD',
 'PLC': 'Pasir Laba Camp',
 'NTFGH': 'Ng Teng Fong General Hospital',
 'MAJU MEDICAL CENTRE': 'MAJU CAMP',
 'TUAS TRANPORT NODE': 'Tuas Naval Base',
 'BENOI SHIPYARD': '23 Benoi Rd',
 'PASIR LEBAR CAMP': 'PASIR LABA CAMP',
 'ARMY LOGISTICS BASE': '601 Old Choa Chu Kang Rd',
 'SEMBAWANG AIR BASE': 'SEMBAWANG AIR BASE',
 'CHONG PANG CAMP': 'CHONG PANG CAMP',
 'CHANGI TRANSPORT NODE': 'Changi Air Base',
 'TUAS EXPLOSIVE JETTY': 'Tuas South Ave 3',
 'PULAU SUDONG (LITA OCEAN)': '4 Pioneer',
 'MAIDAI HILL CAMP': 'MANDAI HILL CAMP',
 'LOR ASRAMA (TP5)': 'MANDAI CAMP II',
 'KAHTIB CAMP': 'KHATIB CAMP',
 'NEE SOON CAMP 500M OPEN RANGE': 'NEE SOON CAMP',
 'SAF YACHT CLUB': 'SAF YACHT CLUB',
 'SAF FERY TERMINAL': 'SAF FERRY TERMINAL',
 'PALAU BRANI': '11 Brani Way',
 'PULAU BRANI': '11 Brani Way',
 'HENDOM CAMP': 'HENDON CAMP',
 'NATIONAL SHOOTING CENTRE': 'National Shooting Centre',
 'NATION SHOOTING CENTRE': 'National Shooting Centre'}

def _resolve_location(name: str) -> str:
    """Apply LOCATION_ALIASES before geocoding."""
    return LOCATION_ALIASES.get(name.strip().upper(), name)
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



def top3_by_risk(rows: list[dict], diversify: bool = True) -> list[dict]:
    """
    Pick 3 driver × route combinations from the full risk matrix.

    diversify=True (default): pick the single best (lowest-prob) combo from
    each risk bucket (Low, Medium, High) so the demo shows a genuine spread
    across tiers instead of just the 3 globally-safest combos. If a bucket
    has no members (e.g. no High-risk combo exists for this task), that slot
    is skipped and backfilled with the next-best remaining combo — never a
    fabricated result, just a graceful "Low/Low/Medium"-style fallback.

    diversify=False restores the original behaviour: the 3 globally lowest-
    risk combos, deduplicated only on (driver_id, route_index).
    """
    seen: set[tuple] = set()
    top3: list[dict] = []

    def _key(row: dict) -> tuple:
        return (row["driver_ctx"]["profile"].driver_id,
                row["route_features"].route_index)

    if diversify:
        buckets: dict[str, dict] = {"Low": None, "Medium": None, "High": None}
        for row in rows:      # already sorted lowest prob first -> best-in-bucket
            lvl = risk_category(row["prob"])
            if buckets[lvl] is None:
                buckets[lvl] = row
        for lvl in ("Low", "Medium", "High"):
            row = buckets[lvl]
            if row is not None:
                seen.add(_key(row))
                top3.append(row)

    # Top up to 3 with the next-best remaining combos (covers both the
    # diversify=False path and any empty buckets above).
    for row in rows:
        if len(top3) >= 3:
            break
        key = _key(row)
        if key not in seen:
            seen.add(key)
            top3.append(row)

    return top3

# ══════════════════════════════════════════════════════════════════════════════
# DEMO SCENARIO SEARCH
# ══════════════════════════════════════════════════════════════════════════════
# Fixed origin/destination for the demo button. Everything else (vehicle,
# purpose, departure hour) is searched at click-time against the REAL risk
# matrix rather than hardcoded — a single fixed "magic" scenario is brittle:
# depending on the exact route geometry it can just as easily land all-Low
# or all-High. Presets are ordered mild -> severe; we stop at the first one
# that spans all 3 buckets, otherwise keep the widest spread we found.
_DEMO_ORIGIN      = "Seletar Camp"
_DEMO_DESTINATION = "Nee Soon Camp"
_DEMO_PRESETS = [
    # (vehicle_number, purpose,     dep_hour)
    # An 8-step mild -> severe ladder across vehicle type/condition, purpose,
    # and weather (hour is deterministic weather for this data seed — see
    # notes above). Conditions were checked directly against the seeded
    # maintenance records rather than assumed; comments note the actual
    # computed condition, not a guess.
    ("SG2001E",         "Admin",      13),  # Light Truck, Fair, Clear   — mildest
    ("SG2004H",         "Training",   13),  # Light Truck, Fair, Clear
    ("SG3004L",         "Operation",  16),  # 5T, Fair, Rain
    ("SG4001O",         "Operation",   9),  # 10T, Fair, Heavy Rain
    ("SG3005M",         "Operation",  16),  # 5T, Poor, Rain
    ("SG3003K",         "Emergency",  19),  # 5T, Poor, Heavy Rain
    ("SG4002P",         "Emergency",   9),  # 10T, Poor, Heavy Rain
    ("SG4003Q",         "Emergency",  21),  # 10T, Poor, Heavy Rain + night — most severe
]

def find_demo_scenario(DG, driver_db, vehicle_db, trip_db, env_sim) -> Optional[dict]:
    """
    Try each preset against the real driver x route matrix for the fixed
    demo O-D pair and return the task context + rows for whichever preset
    gives the best risk-bucket spread. Returns None if the O-D pair itself
    can't be geocoded/routed (falls back to manual entry in that case).
    """
    try:
        # Resolve locations the SAME way geocode_input() would (first
        # candidate's full label, e.g. "SELETAR CAMP (799999)") rather than
        # the raw typed string. build_task_context uses this string as the
        # key for trip-log route familiarity — if this doesn't match what
        # "Analyze Routes" later resolves to for the same query text, the
        # familiarity component of the risk score silently differs between
        # the two paths, changing which drivers land in each risk bucket.
        origin_candidates = geocode_with_candidates(_DEMO_ORIGIN)
        dest_candidates   = geocode_with_candidates(_DEMO_DESTINATION)
        if not origin_candidates or not dest_candidates:
            return None
        origin_label = origin_candidates[0]["label"]
        dest_label   = dest_candidates[0]["label"]
        start_lat, start_lon = origin_candidates[0]["lat"], origin_candidates[0]["lon"]
        end_lat,   end_lon   = dest_candidates[0]["lat"],   dest_candidates[0]["lon"]
        orig      = ox.distance.nearest_nodes(G, start_lon, start_lat)
        dest_node = ox.distance.nearest_nodes(G, end_lon, end_lat)
        route_sequences = find_candidate_routes(DG, orig, dest_node)
    except Exception:
        return None
    if not route_sequences:
        return None

    best = None   # (n_buckets, -preset_index, task_ctx, rows, preset)
    for i, (veh, purpose, hour) in enumerate(_DEMO_PRESETS):
        try:
            task_ctx = build_task_context(
                vehicle_number=veh, origin=origin_label,
                destination=dest_label, purpose=purpose,
                departure_time=time(hour, 0), route_sequences=route_sequences,
                DG=DG, driver_db=driver_db, vehicle_db=vehicle_db,
                trip_log_db=trip_db, env_sim=env_sim,
            )
            task_ctx["purpose"]  = purpose
            task_ctx["dep_time"] = time(hour, 0)
            if not task_ctx["drivers"]:
                continue
            rows = build_results(task_ctx, DG)
        except Exception:
            continue
        buckets = {risk_category(r["prob"]) for r in rows}
        score   = (len(buckets), -i)   # prefer more buckets, then milder preset
        if best is None or score > best[0]:
            best = (score, task_ctx, rows, (veh, purpose, hour, origin_label, dest_label))
        if len(buckets) >= 3:
            break   # already found the ideal spread — no need to keep escalating

    if best is None:
        return None
    _, task_ctx, rows, preset = best
    return {"task_ctx": task_ctx, "rows": rows, "preset": preset}


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
def render_map(G_raw, result: dict, risk: dict,
              all_routes: Optional[list] = None) -> folium.Map:
    nodes   = result["route_features"].node_sequence
    sel_idx = result["route_features"].route_index

    # Collect every node across all candidate routes (not just the selected
    # one) so the map frame is fit to show all 3 routes at once, rather than
    # centering tightly on the selected route and clipping the others.
    all_seqs = [rf.node_sequence for rf in (all_routes or [])] or [nodes]
    if nodes not in all_seqs:
        all_seqs.append(nodes)
    all_lats = [G_raw.nodes[n]["y"] for seq in all_seqs for n in seq]
    all_lons = [G_raw.nodes[n]["x"] for seq in all_seqs for n in seq]

    m = folium.Map(tiles="CartoDB positron")
    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]],
                padding=(24, 24))

    other_routes = [rf for rf in (all_routes or []) if rf.route_index != sel_idx]
    route_note   = ("<br><span style='font-size:10px;color:#6b7280;'>"
                     "Colored dashed lines = other candidate routes (not selected)</span>"
                     if other_routes else "")

    legend_html = f"""
    <div style="position:fixed;bottom:24px;left:24px;background:#fff;
                border:1px solid #d1d5db;border-radius:8px;padding:10px 14px;
                font-size:11px;font-family:sans-serif;z-index:9999;
                box-shadow:0 2px 6px rgba(0,0,0,.1);">
        <b>Contribution of Route Risk</b><br>
        <span style="color:#16a34a;">●</span> Low &nbsp;
        <span style="color:#d97706;">●</span> Medium &nbsp;
        <span style="color:#dc2626;">●</span> High<br>
        <span style="font-size:10px;color:#6b7280;">Hover for exact % of total risk</span><br>
        <span>⚠️</span> Collision Hotspot{route_note}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # Draw the other 1-2 candidate routes first, underneath the selected one.
    # Full opacity, distinct saturated colors, dashed — visible at a glance
    # but still clearly secondary to the risk-coloured selected route on top.
    _ALT_ROUTE_COLORS = ["#2563eb", "#7c3aed", "#0d9488"]
    for rf in other_routes:
        pts   = [[G_raw.nodes[n]["y"], G_raw.nodes[n]["x"]] for n in rf.node_sequence]
        color = _ALT_ROUTE_COLORS[rf.route_index % len(_ALT_ROUTE_COLORS)]
        folium.PolyLine(
            pts, color=color, weight=4, opacity=0.75, dash_array="10,8",
            tooltip=f"Route {rf.route_index + 1} (not selected) · {rf.dist_km:.2f} km",
        ).add_to(m)

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
            <span style="margin-top:4px;text-align:right;">{risk_pill_html(live_level)}</span>
        <div </div>
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
            qualifier = (f"Fleet trade-off — a lower-risk driver was available for this task "
                         f"but was kept for another assignment to protect the overall fleet. ")
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
    bt_lvl = risk_category(bt_prob)
    if gap < 0.005:
        bottleneck_note = (
            f"🎯 Bottleneck: {bt_task.task_id} ({bt_task.origin[:22]} → "
            f"{bt_task.destination[:22]}) — {bt_lvl} risk. "
            f"The assigned driver is the best available for this task. "
            f"The risk is driven by the task's own conditions — vehicle, route, or environment."
        )
    else:
        bottleneck_note = (
            f"🎯 Bottleneck: {bt_task.task_id} ({bt_task.origin[:22]} → "
            f"{bt_task.destination[:22]}) — {bt_lvl} risk. "
            f"A lower-risk driver was available for this task but was assigned elsewhere "
            f"to minimise the fleet's worst-case risk. "
            f"This is an intentional fleet-level trade-off."
        )
    insights.append(bottleneck_note)

    # ── Spread ────────────────────────────────────────────────────────────────
    if spread < 0.0005:
        insights.append("✅ Risk is well-balanced across all tasks — no single assignment is disproportionately dangerous.")
    elif spread > 0.005:
        insights.append(f"⚡ Risk levels vary significantly across tasks — consider whether the bottleneck task can be rescheduled or replanned.")

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

        # Demo preset: searches a short list of vehicle/purpose/hour presets
        # against the real risk matrix for a fixed O-D pair, and loads
        # whichever gives the best Low/Medium/High spread — see
        # find_demo_scenario() above. This runs the actual pipeline rather
        # than assuming one fixed combo will always land across all 3 tiers.
        if st.button("🎭  Load Demo Scenario", use_container_width=True,
                     help="Searches for a task known to spread across Low/Medium/High risk"):
            with st.spinner("Searching for a good demo scenario…"):
                demo = find_demo_scenario(DG, driver_db, vehicle_db, trip_db, env_sim)
            if demo is None:
                st.error("Couldn't build a demo scenario (routing/geocoding failed).")
            else:
                veh, purpose_p, hour, origin_label, dest_label = demo["preset"]
                task_ctx = demo["task_ctx"]
                top3     = top3_by_risk(demo["rows"])
                st.session_state["st_origin_query"] = _DEMO_ORIGIN
                st.session_state["st_dest_query"]   = _DEMO_DESTINATION
                st.session_state["st_dep_time"]     = time(hour, 0)
                st.session_state["st_purpose"]      = purpose_p
                st.session_state["st_vehicle"]      = veh
                st.session_state.results     = top3
                st.session_state.all_routes  = task_ctx["routes"]
                st.session_state.sel_idx     = 0
                st.session_state.task_params = {
                    "origin": origin_label, "destination": dest_label,
                    "purpose": purpose_p, "dep_time": time(hour, 0),
                    "vehicle_number": veh,
                }
                n_buckets = len({risk_category(r["prob"]) for r in demo["rows"]})
                if n_buckets < 2:
                    st.warning("Best available scenario still only spans one risk "
                              "tier for this route — showing the top-3 anyway.")
                # No st.rerun() here: the button click itself already triggered
                # this run, and the widgets below (origin/dest/time/purpose/
                # vehicle) are created further down in this same pass — they'll
                # pick up the session_state values we just set. An extra
                # rerun would also swallow the warning above before it renders.

        origin_result = geocode_input("Origin", "Seletar Camp", key="st_origin")
        dest_result   = geocode_input("Destination", "Nee Soon Camp", key="st_dest")
        c1, c2    = st.columns(2)
        if "st_dep_time" not in st.session_state:
            st.session_state["st_dep_time"] = time(10, 0)
        with c1:  dep_time = st.time_input("Departure Time", key="st_dep_time")
        with c2:  purpose  = st.selectbox("Purpose", ["Admin", "Training", "Operation", "Emergency"],
                                          key="st_purpose")
        vehicle_options = {v.vehicle_number: f"{v.vehicle_number} ({v.vehicle_type})"
                           for v in vehicle_db.all()}
        vehicle_number  = st.selectbox("Vehicle", options=list(vehicle_options.keys()),
                                       format_func=lambda k: vehicle_options[k],
                                       key="st_vehicle")
        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
        st.caption("Drivers pre-filtered by vehicle eligibility.")
        analyze_btn = st.button("🔍  Analyze Routes", type="primary", use_container_width=True)

    # ── Single-task analysis ─────────────────────────────────────────────────
    if analyze_btn:
        if not origin_result or not dest_result:
            st.error("Please enter valid origin and destination — no OneMap result found.")
        else:
            start_loc, start_lat, start_lon = origin_result
            end_loc,   end_lat,   end_lon   = dest_result
            try:
                with st.spinner("Computing routes…"):
                    orig            = ox.distance.nearest_nodes(G, start_lon, start_lat)
                    dest_node       = ox.distance.nearest_nodes(G, end_lon,   end_lat)
                    route_sequences = find_candidate_routes(DG, orig, dest_node)
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
                st.session_state.results     = top3
                st.session_state.all_routes  = task_ctx["routes"]
                st.session_state.sel_idx     = 0
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
            m = render_map(G, sel_res, sel_risk,
                          all_routes=st.session_state.get("all_routes"))
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
                    # Flexible column normalisation: lowercase, strip, collapse non-alnum to _
                    import re as _re
                    df.columns = [_re.sub(r"[^a-z0-9]+", "_",
                                          c.strip().lower()).strip("_")
                                  for c in df.columns]

                    # Alias map: accept various header phrasings → canonical name
                    _aliases = {
                        "task_id":        ["task_id","taskid","id","task"],
                        "origin":         ["origin","from","start","origin_location"],
                        "destination":    ["destination","to","end","dest","destination_location"],
                        "departure_time": ["departure_time","departure","departure_datetime",
                                           "departure_yyyy_mm_dd_hh_mm",
                                           "departure__yyyy_mm_dd_hh_mm_",
                                           "start_time","dep_time","dep"],
                        "end_datetime":   ["end_datetime","end_time","end",
                                           "end__yyyy_mm_dd_hh_mm_",
                                           "end_yyyy_mm_dd_hh_mm","end_date_time"],
                        "purpose":        ["purpose","task_purpose","type"],
                        "vehicle_number": ["vehicle_number","vehicle","vehicle_no",
                                           "veh","veh_number"],
                    }
                    rename_map = {}
                    for canonical, alts in _aliases.items():
                        for col in df.columns:
                            if col in alts and canonical not in df.columns:
                                rename_map[col] = canonical
                    if rename_map:
                        df = df.rename(columns=rename_map)

                    required = {"task_id","origin","destination","departure_time",
                                "purpose","vehicle_number"}
                    missing  = required - set(df.columns)
                    if missing:
                        st.error(f"Missing columns: {', '.join(sorted(missing))}. "
                                 f"Found: {', '.join(sorted(df.columns))}")
                    else:
                        df = df.dropna(subset=["task_id","origin","destination",
                                                "departure_time","purpose","vehicle_number"])
                        df = df[~df["task_id"].astype(str).str.startswith("←")]

                        errors  = []
                        parsed  = []
                        valid_veh  = {v.vehicle_number for v in vehicle_db.all()}
                        valid_purp = {"Admin","Training","Operation","Emergency"}
                        valid_ods  = {(e.origin.upper(), e.destination.upper())
                                      for e in route_db.all_valid()}

                        for _, row in df.iterrows():
                            tid   = str(row["task_id"]).strip()
                            orig  = str(row["origin"]).strip()
                            dest  = str(row["destination"]).strip()
                            dep   = str(row["departure_time"]).strip()
                            purp  = str(row["purpose"]).strip()
                            veh   = str(row["vehicle_number"]).strip()
                            end_raw = str(row.get("end_datetime","")).strip() if "end_datetime" in df.columns else ""

                            # Parse departure — supports "YYYY-MM-DD HH:MM" or "HH:MM"
                            dep_date_str = None
                            try:
                                if len(dep) > 5:  # full datetime
                                    dep_dt_obj = datetime.strptime(dep, "%Y-%m-%d %H:%M")
                                    dep_date_str = dep_dt_obj.date().isoformat()
                                    dep_str = dep_dt_obj.strftime("%H:%M")
                                else:
                                    h, m = dep.split(":")
                                    time(int(h), int(m))
                                    dep_str = dep
                            except Exception:
                                errors.append(f"Row {tid}: invalid departure_time '{dep}' (use YYYY-MM-DD HH:MM or HH:MM)")
                                continue

                            # Parse end datetime (optional)
                            end_dt_str = None
                            if end_raw and end_raw.lower() not in ("nan","none",""):
                                try:
                                    end_dt_str = datetime.strptime(end_raw, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M")
                                except Exception:
                                    pass  # ignore bad end times

                            if purp not in valid_purp:
                                errors.append(f"Row {tid}: invalid purpose '{purp}'")
                                continue
                            if veh not in valid_veh:
                                errors.append(f"Row {tid}: vehicle '{veh}' not found")
                                continue
                            if (orig.upper(), dest.upper()) not in valid_ods:
                                errors.append(f"Row {tid}: O-D '{orig} → {dest}' not in route.json")
                                continue

                            parsed.append({
                                "task_id":        tid,
                                "origin":         orig,
                                "destination":    dest,
                                "departure_date": dep_date_str,
                                "departure_time": dep_str,
                                "end_datetime":   end_dt_str,
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

        # ── Manual add ────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Add Task Manually</div>', unsafe_allow_html=True)
        if st.session_state.get("_task_add_warning"):
            st.warning(st.session_state.pop("_task_add_warning"))
        nodes_list = route_db.nodes()
        sel_node   = st.selectbox("Transport Node", nodes_list, key="ba_node")

        # All O-D pairs for this node in a single searchable selectbox.
        # Streamlit's native selectbox supports keyboard search so no
        # separate text input is needed.
        node_ods  = [(e.origin, e.destination)
                     for e in route_db.all_valid() if e.node == sel_node]
        od_labels = [f"{o}  →  {d}" for o, d in node_ods]
        sel_od_idx = st.selectbox(
            f"O-D Pair ({len(node_ods)} available)",
            range(len(od_labels)),
            format_func=lambda i: od_labels[i],
            key="ba_od",
        )
        sel_origin, sel_dest = node_ods[sel_od_idx]

        c1, c2, c3 = st.columns(3)
        with c1:  ba_dep_date = st.date_input("Departure Date", value=date.today(), key="ba_dep_date")
        with c2:  ba_dep      = st.time_input("Departure Time", value=time(8, 0),   key="ba_dep")
        with c3:  ba_purp     = st.selectbox("Purpose", ["Admin","Training","Operation","Emergency"], key="ba_purp")

        ba_has_end = st.checkbox("Set End Date/Time", value=False, key="ba_has_end")
        if ba_has_end:
            ce1, ce2 = st.columns(2)
            with ce1: ba_end_date = st.date_input("End Date", value=ba_dep_date, key="ba_end_date")
            with ce2: ba_end_time = st.time_input("End Time", value=time(10, 0), key="ba_end_time")
        else:
            ba_end_date = ba_dep_date
            ba_end_time = None

        veh_opts   = {v.vehicle_number: f"{v.vehicle_number} ({v.vehicle_type})"
                      for v in vehicle_db.all()}
        ba_veh     = st.selectbox("Vehicle", list(veh_opts.keys()),
                                   format_func=lambda k: veh_opts[k], key="ba_veh")

        add_btn    = st.button("➕  Add Task", use_container_width=True)
        if add_btn:
            tid = f"T{len(st.session_state.batch_tasks)+1:03d}"
            end_dt_str = None
            if ba_has_end and ba_end_time:
                _dep_dt_check = datetime.combine(ba_dep_date, ba_dep)
                _end_dt_check = datetime.combine(ba_end_date, ba_end_time)
                if _end_dt_check <= _dep_dt_check:
                    # Store in session_state instead of calling st.warning() here —
                    # st.rerun() below fires immediately and would wipe it before
                    # the browser ever renders this frame.
                    st.session_state["_task_add_warning"] = (
                        f"⚠️ End datetime ({_end_dt_check.strftime('%d/%m/%Y %H:%M')}) is not "
                        f"after the departure time ({_dep_dt_check.strftime('%d/%m/%Y %H:%M')}) — "
                        f"ignoring it; duration will be estimated from the route instead.")
                else:
                    end_dt_str = _end_dt_check.strftime("%Y-%m-%d %H:%M")
            st.session_state.batch_tasks.append({
                "task_id":         tid,
                "origin":          sel_origin,
                "destination":     sel_dest,
                "departure_date":  ba_dep_date.strftime("%Y-%m-%d"),
                "departure_time":  ba_dep.strftime("%H:%M"),
                "end_datetime":    end_dt_str,
                "purpose":         ba_purp,
                "vehicle_number":  ba_veh,
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
                    end_str = f" → {t['end_datetime']}" if t.get("end_datetime") else ""
                    dep_str = f"{t.get('departure_date','')} {t['departure_time']}"
                    st.markdown(
                        f'<div style="font-size:12px;padding:5px 0;line-height:1.6;">'
                        f'<b>{t["task_id"]}</b> · {dep_str}{end_str} · {t["purpose"]}<br>'
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
            st.session_state.pop("_override_dist", None)
            with st.spinner("Geocoding task locations via OneMap…"):
                for t in st.session_state.batch_tasks:
                    dep = time(*[int(x) for x in t["departure_time"].split(":")])
                    # Skip task if origin or destination is blank
                    if not str(t.get("origin","")).strip() or not str(t.get("destination","")).strip():
                        st.warning(f"⚠️ {t['task_id']}: origin or destination is empty — task skipped.")
                        continue

                    def _geocode_loc(name, task_id, role):
                        resolved = _resolve_location(name)
                        if resolved != name:
                            st.info(f"ℹ️ {task_id}: '{name}' → '{resolved}' (alias).")
                        try:
                            return onemap_geocode(resolved)
                        except Exception as e:
                            st.warning(f"⚠️ {task_id}: geocoding failed for {role} '{name}' ({e}) — task skipped.")
                            return None

                    orig_result = _geocode_loc(t["origin"],     t["task_id"], "origin")
                    dest_result = _geocode_loc(t["destination"], t["task_id"], "destination")
                    if orig_result is None or dest_result is None:
                        continue
                    orig_lat, orig_lon = orig_result
                    dest_lat, dest_lon = dest_result
                    try:
                        orig_node  = ox.distance.nearest_nodes(G, orig_lon, orig_lat)
                        dest_node  = ox.distance.nearest_nodes(G, dest_lon, dest_lat)
                        route_seqs = find_candidate_routes(DG, orig_node, dest_node)
                    except Exception as e:
                        st.warning(f"⚠️ Route finding failed for {t['task_id']}: {e}.")
                        route_seqs = []
                    dep_date = None
                    if t.get("departure_date"):
                        try:
                            from datetime import date as _date
                            dep_date = _date.fromisoformat(t["departure_date"])
                        except Exception:
                            pass
                    end_dt = None
                    if t.get("end_datetime"):
                        try:
                            _end_dt_candidate = datetime.fromisoformat(t["end_datetime"])
                            _dep_dt_check     = datetime.combine(dep_date or date.today(), dep)
                            if _end_dt_candidate <= _dep_dt_check:
                                st.warning(
                                    f"⚠️ {t['task_id']}: end datetime "
                                    f"({_end_dt_candidate.strftime('%d/%m/%Y %H:%M')}) is not after "
                                    f"the departure time ({_dep_dt_check.strftime('%d/%m/%Y %H:%M')}) "
                                    f"— ignoring it; duration will be estimated from the route instead.")
                            else:
                                end_dt = _end_dt_candidate
                        except Exception:
                            pass
                    tasks_for_alloc.append(AllocTask(
                        task_id         = t["task_id"],
                        origin          = t["origin"],
                        destination     = t["destination"],
                        departure_time  = dep,
                        departure_date  = dep_date,
                        end_datetime    = end_dt,
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
            # ── Helper: risk badge / pills HTML (used across this panel) ──────
            def _badge(lvl):
                c = {"Low":"#16a34a","Medium":"#d97706","High":"#dc2626"}.get(lvl,"#6b7280")
                return (f'<span style="background:{c};color:#fff;border-radius:9999px;'
                        f'padding:2px 12px;font-size:11px;font-weight:700;">{lvl}</span>')

            def _pills(dist):
                out = ""
                for l in ["High","Medium","Low"]:
                    if dist.get(l,0):
                        c = {"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"}[l]
                        out += (f'<span style="background:{c};color:#fff;border-radius:9999px;'
                               f'padding:3px 14px;font-size:12px;font-weight:700;margin-right:4px;">'
                               f'{l} {dist[l]}</span>')
                return out

            def _th(txt, w=""):
                ws = f'width:{w};' if w else ""
                return (f'<th style="padding:6px 10px;background:#f1f5f9;font-weight:600;'
                        f'text-align:left;border-bottom:1px solid #e2e8f0;'
                        f'font-size:11px;color:#374151;{ws}">{txt}</th>')

            def _td(content, extra=""):
                return (f'<td style="padding:6px 10px;border-bottom:1px solid #f1f5f9;'
                        f'font-size:12px;vertical-align:middle;{extra}">{content}</td>')

            # ── Fleet summary strip ───────────────────────────────────────────
            n_tasks    = len(result.assignments)
            n_unassign = len(result.unassigned)
            # Recomputed live (not result.objective) so it reflects overrides
            max_risk   = max((a.risk["prob"] for a in result.assignments), default=0.0)
            max_lvl    = risk_category(max_risk)
            avg_risk   = sum(a.risk["prob"] for a in result.assignments) / max(n_tasks, 1)
            avg_lvl    = risk_category(avg_risk)
            lvl_col    = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}[max_lvl]
            avg_lvl_col= {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}[avg_lvl]

            from collections import Counter as _Ct1
            _driver_counts     = _Ct1(a.driver_ctx["profile"].driver_id for a in result.assignments)
            n_multi_task_drivers = sum(1 for c in _driver_counts.values() if c > 1)

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:8px;margin-bottom:8px;">
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:#111827;">{n_tasks}</div>
                <div style="font-size:11px;color:#6b7280;">Assigned</div>
              </div>
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:{'#dc2626' if n_unassign else '#111827'};">{n_unassign}</div>
                <div style="font-size:11px;color:#6b7280;">Unassigned</div>
              </div>
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:#111827;">{n_multi_task_drivers}</div>
                <div style="font-size:11px;color:#6b7280;">Multi-task Drivers</div>
              </div>
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:{lvl_col};">{max_lvl}</div>
                <div style="font-size:11px;color:#6b7280;">Highest Risk Task</div>
              </div>
              <div style="background:#f8fafc;border:0.5px solid #e2e5ea;border-radius:8px;padding:10px 12px;">
                <div style="font-size:20px;font-weight:700;color:{avg_lvl_col};">{avg_lvl}</div>
                <div style="font-size:11px;color:#6b7280;">Avg Risk per Task</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Risk distribution (horizontal bar chart) ──────────────────────
            from collections import Counter as _Ct2
            _rd = _Ct2(risk_category(a.risk["prob"]) for a in result.assignments)
            _rc = {"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"}
            _rd_total = n_tasks or 1
            _bars = ""
            for l in ["High","Medium","Low"]:
                _cnt = _rd.get(l, 0)
                _pct = (_cnt / _rd_total * 100) if _rd_total else 0
                _bars += f"""
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <div style="width:64px;font-size:12px;color:#374151;font-weight:600;">{l}</div>
                    <div style="flex:1;background:#f1f5f9;border-radius:4px;height:16px;
                                position:relative;overflow:hidden;">
                        <div style="width:{_pct:.1f}%;background:{_rc[l]};height:100%;
                                    border-radius:4px;"></div>
                    </div>
                    <div style="width:32px;text-align:right;font-size:12px;color:#374151;
                                font-weight:700;margin-left:8px;">{_cnt}</div>
                </div>"""
            st.markdown('<div class="section-label" style="margin-top:16px;">📊 Risk Distribution</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div style="margin-bottom:12px;margin-top:6px;">{_bars}</div>',
                        unsafe_allow_html=True)


            # ── Before / After override risk distribution ─────────────────────
            if "_override_dist" in st.session_state:
                _od = st.session_state["_override_dist"]
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                    f'border-left:4px solid #6b7280;border-radius:6px;'
                    f'padding:10px 14px;margin-bottom:12px;">'
                    f'<div style="font-size:12px;margin-bottom:5px;">'
                    f'<b>Before:&nbsp;</b>{_pills(_od["before"])}</div>'
                    f'<div style="font-size:12px;">'
                    f'<b>After:&nbsp;&nbsp;</b>{_pills(_od["after"])}</div></div>',
                    unsafe_allow_html=True)

            # ── Warnings ──────────────────────────────────────────────────────
            if result.warnings:
                for w in result.warnings:
                    st.warning(w, icon="⚠️")


            # ── Assignment cards ──────────────────────────────────────────────
            st.markdown('<div class="section-label">📋 Assignments</div>', unsafe_allow_html=True)
            for i, asgn in enumerate(
                        sorted(result.assignments, key=lambda a: a.task.task_id)):
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

                _dep_dt = datetime.combine(task.departure_date or date.today(), task.departure_time)
                if task.end_datetime:
                    _end_dt = (task.end_datetime if isinstance(task.end_datetime, datetime)
                               else datetime.combine(task.departure_date or date.today(), task.end_datetime))
                else:
                    _end_dt = _dep_dt + timedelta(hours=estimate_duration_h(rf.dist_km))
                _dep_s = _dep_dt.strftime("%d/%m/%Y %H:%M")
                _end_s = (_end_dt.strftime("%d/%m/%Y %H:%M") if _end_dt.date() != _dep_dt.date()
                          else _end_dt.strftime("%H:%M"))

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
                                {lvl_icon} {lvl}
                            </span>
                        </div>
                        <div style="font-family:'DM Mono',monospace;font-size:11px;
                                    color:#6b7280;margin-top:3px;line-height:1.7;">
                            {task.origin[:28]} → {task.destination[:28]}<br>
                            {_dep_s} → {_end_s} · {task.purpose}
                            · R{rf.route_index+1} · {rf.dist_km:.1f} km
                            · {task.vehicle_number} ({vtype})
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col_sel:
                    btn_txt = "▶" if selected else "○"
                    if st.button(btn_txt, key=f"asel_{i}", help=f"View {task.task_id}"):
                        st.session_state.alloc_sel_idx = i
                        st.rerun()

            # ── Repeated Drivers ──────────────────────────────────────────────
            from collections import Counter
            driver_task_count = Counter(
                asgn.driver_ctx["profile"].driver_id
                for asgn in result.assignments
            )
            repeated = {did: cnt for did, cnt in driver_task_count.items() if cnt > 1}

            # ── Build score matrix for alternatives (best driver per task) ────
            # For each task, find lowest-risk other driver who is available on a diff day
            _score_matrix = getattr(result, "score_matrix", {})

            def _best_alt(asgn):
                """
                Return (driver_profile, risk_label, prob, route_index) for the best
                schedulable (driver, route) combination for this task across all routes.
                Overriding to this driver will also switch to this route (see apply_btn
                logic below), so the suggestion and the applied result stay consistent.
                Returns (None, None, None, None) if no schedulable alternative found.
                """
                task  = asgn.task
                cur_d = asgn.driver_ctx["profile"].driver_id

                candidates = sorted(
                    [(did, ridx, v["prob"])
                     for (tid, did, ridx), v in _score_matrix.items()
                     if tid == task.task_id and did != cur_d],
                    key=lambda x: x[2]  # lowest risk first
                )
                for did, ridx, prob in candidates:
                    if task.departure_date:
                        other_same_day = any(
                            a.task.departure_date == task.departure_date
                            and a.driver_ctx["profile"].driver_id == did
                            and a.task.task_id != task.task_id
                            for a in result.assignments
                        )
                        if other_same_day:
                            continue
                    dp = driver_db.get(did)
                    if dp is None:
                        continue
                    return dp, risk_category(prob), prob, ridx
                return None, None, None, None

            # ── #3: Repeated Drivers table ────────────────────────────────────
            if repeated:
                st.markdown('<div class="section-label" style="margin-top:16px;">🔄 Repeated Drivers</div>',
                            unsafe_allow_html=True)
                n_t_total = len(result.assignments) + len(result.unassigned)
                n_d_total = len(list(driver_db.all())) if hasattr(driver_db, "all") else len({a.driver_ctx["profile"].driver_id for a in result.assignments})
                n_rep_drivers = len(repeated)
                if n_d_total >= n_t_total:
                    st.success(f"One-driver-one-task is possible ({n_d_total} drivers, {n_t_total} tasks). "
                               f"{n_rep_drivers} driver(s) currently hold multiple tasks — can be resolved via override.", icon="ℹ️")
                else:
                    short = n_t_total - n_d_total
                    st.info(f"{n_d_total} drivers for {n_t_total} tasks — "
                            f"at least {short} driver(s) must take multiple tasks (unavoidable). "
                            f"{n_rep_drivers} driver(s) currently repeated.", icon="ℹ️")

                for did, cnt in repeated.items():
                    driver_asgns = [a for a in result.assignments
                                    if a.driver_ctx["profile"].driver_id == did]
                    dname   = driver_asgns[0].driver_ctx["profile"].name
                    task_ids = ", ".join(a.task.task_id for a in driver_asgns)
                    rows_html = ""
                    for a in driver_asgns:
                        t = a.task
                        dep_s = (t.departure_date.strftime("%d/%m/%Y ") if t.departure_date else "") + t.departure_time.strftime("%H:%M")
                        if t.end_datetime:
                            _end_dt_rd = (t.end_datetime if isinstance(t.end_datetime, datetime)
                                          else datetime.combine(t.departure_date or date.today(), t.end_datetime))
                        else:
                            _dep_dt_rd = datetime.combine(t.departure_date or date.today(), t.departure_time)
                            _end_dt_rd = _dep_dt_rd + timedelta(
                                hours=estimate_duration_h(a.route_features.dist_km))
                        end_s = _end_dt_rd.strftime("%d/%m/%Y %H:%M")
                        lvl   = risk_category(a.risk["prob"])
                        ri_r  = getattr(a.route_features, "route_index", 0)
                        route = f"{t.origin[:14]}→{t.destination[:14]}"
                        cur_cell = f'{dname} (R{ri_r+1})<br>{_badge(lvl)}'
                        alt_d2, alt_lbl2, _, alt_ri2 = _best_alt(a)
                        if alt_d2:
                            alt_cell = f"{alt_d2.name} (R{alt_ri2+1})<br>{_badge(alt_lbl2)}"
                        else:
                            alt_cell = '<span style="color:#9ca3af;font-style:italic;">No option</span>'
                        rows_html += (
                            f"<tr>"
                            + _td(f'<b>{t.task_id}</b>')
                            + _td(dep_s, "color:#374151;")
                            + _td(end_s, "color:#6b7280;")
                            + _td(route)
                            + _td(cur_cell)
                            + _td(alt_cell, "color:#1d4ed8;font-weight:600;")
                            + "</tr>"
                        )
                    st.markdown(f"""
                    <div style="border:1px solid #bfdbfe;border-left:4px solid #3b82f6;
                                border-radius:6px;padding:10px 14px;margin-bottom:10px;
                                background:#eff6ff;">
                      <div style="font-weight:700;color:#1d4ed8;font-size:13px;margin-bottom:8px;">
                          🔄 {dname} ({did}) · {cnt} tasks: {task_ids}
                      </div>
                      <table style="width:100%;border-collapse:collapse;">
                        <thead><tr>
                          {_th("Task","60px")}{_th("Start","130px")}{_th("End","130px")}
                          {_th("Route")}{_th("Current Driver (Risk)","160px")}{_th("Best Alternative","160px")}
                        </tr></thead>
                        <tbody>{rows_html}</tbody>
                      </table>
                      <div style="font-size:11px;color:#6b7280;margin-top:6px;font-style:italic;">
                          ℹ Suggestions are computed independently. Apply one override at a time.
                      </div>
                    </div>""", unsafe_allow_html=True)

            # ── #6: Override Assignments — proper table ───────────────────────
            st.markdown('<div class="section-label" style="margin-top:16px;">✏️ Override Assignments</div>',
                        unsafe_allow_html=True)
            st.caption("The initial allocation is globally optimal. "
                       "Any override is an operational adjustment — "
                       "risk distribution before and after will be shown on Apply.")

            all_driver_opts = {d.driver_id: f"{d.name} ({d.driver_id})"
                               for d in driver_db.all()} if hasattr(driver_db, "all") else {}
            if not all_driver_opts:
                all_driver_opts = {a.driver_ctx["profile"].driver_id:
                                   f"{a.driver_ctx['profile'].name} ({a.driver_ctx['profile'].driver_id})"
                                   for a in result.assignments}

            override_selections = {}
            # Column headers
            _hdr_cols = st.columns([1,2,2,3,3,3,2])
            for _ci, _ht in zip(_hdr_cols, ["Task","Start","End","Route",
                                             "Current Driver (Risk)","Suggestion","Override to"]):
                _ci.markdown(f'<div style="font-size:11px;font-weight:600;color:#374151;'
                             f'background:#f1f5f9;padding:5px 2px;border-bottom:2px solid #e2e8f0;">'
                             f'{_ht}</div>', unsafe_allow_html=True)

            with st.form("override_form"):
                th_s = ("padding:5px 8px;background:#f1f5f9;font-weight:600;font-size:11px;"
                        "color:#374151;text-align:left;border-bottom:2px solid #e2e8f0;"
                        "white-space:nowrap;")
                td_s = ("padding:5px 8px;border-bottom:1px solid #f1f5f9;"
                        "font-size:11px;vertical-align:middle;")
                for asgn in sorted(result.assignments, key=lambda a: a.task.task_id):
                    task    = asgn.task
                    d       = asgn.driver_ctx["profile"]
                    lvl     = risk_category(asgn.risk["prob"])
                    dep_s   = ((task.departure_date.strftime("%d/%m/%Y ") if task.departure_date else "")
                                + task.departure_time.strftime("%H:%M"))
                    if task.end_datetime:
                        _end_dt_ov = (task.end_datetime if isinstance(task.end_datetime, datetime)
                                      else datetime.combine(task.departure_date or date.today(), task.end_datetime))
                    else:
                        _dep_dt_ov = datetime.combine(task.departure_date or date.today(), task.departure_time)
                        _end_dt_ov = _dep_dt_ov + timedelta(
                            hours=estimate_duration_h(asgn.route_features.dist_km))
                    end_s   = _end_dt_ov.strftime("%d/%m/%Y %H:%M")
                    ri      = getattr(asgn.route_features, "route_index", 0)
                    route_s = f"{task.origin[:16]}→{task.destination[:16]}"
                    is_rep  = driver_task_count.get(d.driver_id, 0) > 1
                    row_bg  = "#eff6ff" if is_rep else "#ffffff"
                    rep_icon = "🔄 " if is_rep else ""
                    cur_html = f"<b>{d.name}</b> (R{ri+1})<br>{_badge(lvl)}"
                    alt_d3, alt_lbl3, _, alt_ri3 = _best_alt(asgn) if is_rep else (None, None, None, None)
                    alt_html = (f"{alt_d3.name} (R{alt_ri3+1})<br>{_badge(alt_lbl3)}"
                                if alt_d3 else '<span style="color:#9ca3af;font-size:10px;">—</span>')

                    # 7 columns: Task | Start | End | Route | Current | Suggestion | Override(dropdown)
                    # Use [1,2,2,3,3,3,2] ratio columns
                    c1,c2,c3,c4,c5,c6,c7 = st.columns([1,2,2,3,3,3,2])
                    with c1:
                        bg = "background:#eff6ff;" if is_rep else ""
                        st.markdown(f'<div style="font-size:11px;font-weight:600;padding:8px 2px;{bg}">{rep_icon}{task.task_id}</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div style="font-size:11px;padding:8px 2px;color:#374151;">{dep_s}</div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div style="font-size:11px;padding:8px 2px;color:#6b7280;">{end_s}</div>', unsafe_allow_html=True)
                    with c4:
                        st.markdown(f'<div style="font-size:11px;padding:8px 2px;">{route_s}</div>', unsafe_allow_html=True)
                    with c5:
                        st.markdown(f'<div style="font-size:11px;padding:8px 2px;">{cur_html}</div>', unsafe_allow_html=True)
                    with c6:
                        st.markdown(f'<div style="font-size:11px;padding:8px 2px;">{alt_html}</div>', unsafe_allow_html=True)
                    with c7:
                        sel = st.selectbox(
                            f"ov_{task.task_id}",
                            options=list(all_driver_opts.keys()),
                            index=list(all_driver_opts.keys()).index(d.driver_id)
                                  if d.driver_id in all_driver_opts else 0,
                            format_func=lambda k: all_driver_opts[k],
                            key=f"ov_{task.task_id}",
                            label_visibility="collapsed",
                        )
                        override_selections[task.task_id] = sel

                # Column headers (rendered above via st.markdown to stay fixed)
                apply_btn = st.form_submit_button("✔  Apply Overrides", type="primary")

            if apply_btn:
                overrides = {
                    tid: new_did
                    for tid, new_did in override_selections.items()
                    if new_did != next(
                        a.driver_ctx["profile"].driver_id
                        for a in result.assignments if a.task.task_id == tid
                    )
                }
                if not overrides:
                    st.info("No changes detected.")
                else:
                    from collections import Counter as _Ct
                    before_dist = _Ct(risk_category(a.risk["prob"]) for a in result.assignments)
                    conflict_msgs = []
                    for asgn in result.assignments:
                        tid = asgn.task.task_id
                        if tid not in overrides: continue
                        new_did = overrides[tid]
                        new_driver = driver_db.get(new_did)
                        if new_driver is None:
                            conflict_msgs.append(f"{tid}: driver {new_did} not found.")
                            continue
                        conflict = False
                        if asgn.task.departure_date:
                            for other in result.assignments:
                                if (other.driver_ctx["profile"].driver_id == new_did
                                        and other.task.task_id != tid
                                        and other.task.departure_date == asgn.task.departure_date):
                                    conflict_msgs.append(
                                        f"{tid}: {new_driver.name} already has task "
                                        f"{other.task.task_id} on {asgn.task.departure_date} "
                                        f"(same day). Override rejected.")
                                    conflict = True; break
                        if not conflict:
                            vprofile = vehicle_db.get(asgn.task.vehicle_number)
                            fatigue  = DriverDB.compute_fatigue(new_driver, asgn.task.departure_time)
                            route_exp = trip_db.route_experience(
                                new_did, asgn.task.origin, asgn.task.destination)
                            veh_type_trips = trip_db.vehicle_type_trips(
                                new_did, vprofile.vehicle_type, vehicle_db)
                            total_trips = sum(
                                trip_db.vehicle_type_experience(new_did, vehicle_db).values())
                            new_driver_ctx = {
                                "profile":          new_driver,
                                "fatigue_hours":    round(fatigue, 2),
                                "route_experience": route_exp,
                                "veh_type_trips":   veh_type_trips,
                                "total_trips":      total_trips,
                            }
                            P = compute_P(new_driver_ctx)
                            A = compute_A(asgn.vehicle_ctx, asgn.env_ctx["weather"])

                            # Pick the best route for THIS driver — not necessarily
                            # the route the previous driver used — same as how the
                            # solver jointly optimises driver+route per task.
                            candidate_routes = (asgn.task.route_features
                                                or [asgn.route_features])
                            best_rf, best_risk = None, None
                            for rf_c in candidate_routes:
                                T_c    = compute_T(new_driver_ctx, asgn.task.purpose, rf_c.route_index)
                                risk_c = compute_route_risk(
                                    rf_c, P, T_c, A, asgn.env_ctx,
                                    asgn.task.departure_time, DG)
                                if best_risk is None or risk_c["prob"] < best_risk["prob"]:
                                    best_rf, best_risk = rf_c, risk_c

                            asgn.driver_ctx     = new_driver_ctx
                            asgn.route_features = best_rf
                            asgn.risk           = best_risk
                            asgn.fatigue_at_dep = new_driver_ctx["fatigue_hours"]
                    for msg in conflict_msgs:
                        st.warning(f"⚠ {msg}")
                    after_dist = _Ct(risk_category(a.risk["prob"]) for a in result.assignments)
                    st.session_state["_override_dist"] = {
                        "before": dict(before_dist), "after": dict(after_dist)}
                    st.rerun()



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
            _sorted_assignments = sorted(result.assignments, key=lambda a: a.task.task_id)
            sel_asgn    = _sorted_assignments[st.session_state.alloc_sel_idx]
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

            # ── Route Map ─────────────────────────────────────────────────────
            st.markdown('<div class="section-label" style="margin-top:12px;">Route Map</div>',
                        unsafe_allow_html=True)
            map_result_dict = {
                "route_features": rf,
                "driver_ctx":     driver_ctx_display,
                "vehicle_ctx":    vehicle_ctx,
                "env_ctx":        env_ctx,
                "purpose":        purpose_v,
                "dep_time":       dep_time_v,
            }
            m = render_map(G, map_result_dict, sel_asgn.risk)
            st_folium(m, width=None, height=350, use_container_width=True, returned_objects=[])