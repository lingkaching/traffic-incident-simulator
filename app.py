import streamlit as st
import osmnx as ox
import networkx as nx
import math
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import pickle
import random
from datetime import time

st.set_page_config(layout="wide", page_title="Driving Risk Simulator V9")
if "routes" not in st.session_state:
    st.session_state.routes = None

# ==============================
# 1. CONFIG & MULTIPLIERS
# ==============================
GRAPH_PATH = "sg_drive.pkl"
BASE_LAMBDA = 1e-6  #of incident per mile
 
MULTIPLIERS = {
    "mileage": {"<20k": 1.4, "20k-100k": 1.1, "100k-300k": 0.95, ">300k": 0.9},
    "task_purpose": {"Admin": 1.0, "Training": 1.05, "Operation": 1.2, "Emergency": 1.6},
    "vehicle_type": {"Car": 1.0, "Light Truck": 1.2, "5T": 1.5, "10T": 2.0},
    "weather": {"Clear": 1.0, "Rain": 1.4, "Heavy Rain": 1.8},
    "road_class": {
        "motorway": 0.7, "trunk": 0.9, "primary": 1.2, 
        "secondary": 1.5, "tertiary": 1.8, "residential": 2.2
    }
}

# Logic Constants (Formerly hardcoded in functions)
FATIGUE_RATE = 0.08
CRASH_HISTORY_MULT = 1.5
EXP_THRESHOLD = 20
EXP_PENALTY = 1.2
EXP_BONUS = 0.9
MAINT_POOR_MULT = 1.3
TECH_PENALTY = 0.05
NIGHT_VIS_MULT = 1.4
NARROW_ROAD_MULT = 1.3
SHARP_TURN_MULT = 1.4
HOTSPOT_MULT = 2.5


@st.cache_resource
def load_and_prep_data():
    if not os.path.exists(GRAPH_PATH):
        G_raw = ox.graph_from_place("Singapore", network_type="drive")
        with open(GRAPH_PATH, "wb") as f: pickle.dump(G_raw, f)
    else:
        with open(GRAPH_PATH, "rb") as f: G_raw = pickle.load(f)

    G_raw = ox.bearing.add_edge_bearings(G_raw)
    DG = nx.DiGraph()
    for node, data in G_raw.nodes(data=True):
        DG.add_node(node, x=data['x'], y=data['y'])
    
    # 1. Flatten and Sanitize
    all_candidates = []
    for u, v, k, data in G_raw.edges(keys=True, data=True):
        clean_data = {key: (val[0] if isinstance(val, list) else val) for key, val in data.items()}
        
        # Safe numeric parsing
        w_raw = str(clean_data.get("width", "6.0")).replace('m', '').split(',')[0].strip()
        width = float(w_raw) if w_raw.replace('.', '', 1).isdigit() else 6.0
        lanes = int(str(clean_data.get("lanes", "2")).split(',')[0].strip()) if str(clean_data.get("lanes", "2")).split(',')[0].strip().isdigit() else 2
        
        # narrow road condition 
        clean_data['is_narrow'] = (width < 6.0) or (lanes < 2)
        clean_data['hotspot_mult'] = 1.0 
        clean_data['is_sharp'] = False # Default to False, will be updated below
        
        if G_raw.degree[u] >= 4 or clean_data.get("highway") == "motorway_link" or clean_data.get("bridge") == "yes":
            all_candidates.append((u, v))
            
        DG.add_edge(u, v, **clean_data)

    # 2. Sample Hazards (Hotspots)
    rng = random.Random(42)
    sampled_hotspots = rng.sample(all_candidates, min(len(all_candidates), 1000))
    for u, v in sampled_hotspots: DG[u][v]['hotspot_mult'] = HOTSPOT_MULT

    # 3. Detect and Sample Sharp Turns
    # This logic detects turns > 60 degrees and then downsamples for performance
    for node in DG.nodes():
        for _, v_in, d_in in DG.in_edges(node, data=True):
            for _, v_out, d_out in DG.out_edges(node, data=True):
                b1, b2 = d_in.get('bearing', 0), d_out.get('bearing', 0)
                diff = abs(b1 - b2)
                # sharp turn condition 
                if min(diff, 360 - diff) > 90:
                    # Apply 20% downsampling to prevent "Sharp Turn" overload
                    if rng.random() < 0.2:
                        DG[node][v_out]['is_sharp'] = True
    
    # 4. Final Narrow Turn downsampling
    for u, v in DG.edges():
        if DG[u][v].get('is_narrow') and rng.random() > 0.2: 
            DG[u][v]['is_narrow'] = False

    return G_raw, DG

G, DG = load_and_prep_data()

# ==============================
# 3. RISK ENGINE LOGIC
# ==============================
def risk_category(risk_per_meter):
    if risk_per_meter < 2e-6: return "Low"
    elif risk_per_meter < 5e-6: return "Medium"
    return "High"

def compute_global_indices(inputs):
    p = MULTIPLIERS["mileage"][inputs['mileage']] * (CRASH_HISTORY_MULT if inputs['crash'] else 1.0) * (1 + (inputs['fatigue'] * FATIGUE_RATE))
    a = MULTIPLIERS["vehicle_type"][inputs['v_type']] * (MAINT_POOR_MULT if inputs['maint'] == "Poor" else 1.0) * (1 - len(inputs['tech']) * TECH_PENALTY)
    return p, a

def compute_route_task_index(route, inputs, DG):
    # Future we need to compute task/route Familiarity based on driver's past completed task assignment 
    t = MULTIPLIERS["task_purpose"][inputs['purpose']] * (EXP_BONUS  if inputs['task_exp'] > EXP_THRESHOLD else EXP_PENALTY )
    return t

def compute_route_env_index(route, base_traffic, weather, dep_time, DG):
    # Modular: uses global weather/time + route-specific segment traffic
    is_night = dep_time.hour < 7 or dep_time.hour > 19
    vis_m = NIGHT_VIS_MULT if is_night else 1.0
    weather_m = MULTIPLIERS["weather"].get(weather, 1.0)
    # Future: replace 'base_traffic' with a call to a traffic_dict[(u,v)]
    # return e[(u,v)]
    return vis_m * weather_m * (base_traffic / 3)

def compute_route_roadway_index(route, DG):
    """
    Returns the average index and a dictionary of multipliers keyed by (u, v).
    """
    segment_mults_dict = {}

    for u, v in zip(route[:-1], route[1:]):
        edge = DG.get_edge_data(u, v)
        # Calculate multiplier
        mult = edge.get('hotspot_mult', 1.0)
        mult *= MULTIPLIERS["road_class"].get(edge.get("highway", "residential"), 1.5)
        if edge.get('is_narrow'): mult *= NARROW_ROAD_MULT
        if edge.get('is_sharp'): mult *= SHARP_TURN_MULT
        
        # Store using edge tuple as key
        segment_mults_dict[(u, v)] = mult
        

    return segment_mults_dict

def segment_level_risk(route, p, t, a, e, r, DG):
    hotspot_coords = []
    total_lambda, total_len, segment_data = 0, 0, []
    counts = {"hotspots": 0, "sharp": 0, "narrow": 0}
    for u, v in zip(route[:-1], route[1:]):
        edge = DG.get_edge_data(u, v)
        # Check if this edge is a hotspot
        if edge.get('hotspot_mult', 1.0) > 1.0:
            # Store coordinates of the start node of the hotspot segment
            hotspot_coords.append((DG.nodes[u]['y'], DG.nodes[u]['x']))

        length = edge.get("length", 0)
        total_len += length
        # Flags
        if edge.get('hotspot_mult', 1.0) > 1.0: counts["hotspots"] += 1
        if edge.get('is_sharp'): counts["sharp"] += 1
        if edge.get('is_narrow'): counts["narrow"] += 1
        # Lambda

        lambda_i = BASE_LAMBDA * length * p * t * a * e * r[(u,v)]
        total_lambda += lambda_i
        segment_data.append((u, v, lambda_i / max(length, 1)))
    return {"prob": 1-math.exp(-total_lambda), "dist": total_len/1000, "risks": segment_data, "counts": counts, "hotspot_coords": hotspot_coords}



# ==============================
# 4. SIDEBAR UI
# ==============================
with st.sidebar:
    st.title("🛡 Traffic Risk Simulator V7")
    
    st.header("Officer Input")
    with st.expander("📍 Task", expanded=True):

        start_loc = st.text_input("Start Location", "Orchard Road, Singapore")
        end_loc = st.text_input("End Location", "Changi Airport, Singapore")
        dep_time = st.time_input("Departure Time", value=time(10, 0))
        purpose = st.selectbox("Purpose", ["Admin", "Training", "Operation", "Emergency"])

    
    with st.expander("👤 Driver", expanded=True):
        driver_name = st.text_input("Driver Name", "ANONYMOUS")
        driver_id = st.text_input("Driver ID", "12345678")
        fatigue = st.slider("Hours Since Last Break", 0.0, 12.0, 2.0)

    with st.expander("🚛 Vehicle", expanded=True):
        vehicle_id = st.text_input("Vehicle Number", "ABC1234")


    st.header("Derived Features")

    with st.expander("👤 Driver + 📍 Task (94%)", expanded=True):
        st.text("👤 Driver:")
        d_cat = st.selectbox("Driver Category", ["A", "B", "C", "D"])
        mileage = st.select_slider("Driver Mileage", options=["<20k", "20k-100k", "100k-300k", ">300k"])
        crash = st.toggle("Prior Crash History")
        st.text("Fatigue (Based on Hours Since Last Break)")
        
        st.text("📍 Task:")
        task_exp = st.slider("Route Experience", 0, 100, 10)
        vehicle_exp = st.text("Vehicle Experience")
        st.text("Task Urgency (Based on Task Purpose)")

    with st.expander("🚛 Vehicle (2%)", expanded=True):
        v_type = st.selectbox("Vehicle Type", ["Car", "Light Truck", "5T", "10T"])
        maint = st.selectbox("Vehicle Condition (Based on Vehicle Age and Maintenance History)", ["Good", "Fair", "Poor"])
        tech = st.multiselect("Safety Tech", ["ABS", "Reverse Cam", "Blind Spot Monitor"])

    with st.expander("☁️ Environment (1%)", expanded=True):
        weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Heavy Rain"])
        traffic = st.slider("Traffic Density (1-5)", 1, 5, 3)
        st.text("Visibility (Based on Weather and Depature Time)")

    with st.expander("🛣️ Roadway (2%)", expanded=True):
        st.text("Infrastructure (e.g.,Sharp Turns, Narrow Roads)")
        st.text("Road Class (e.g., Highway, Major, Minior)")
        st.text("Hotsports")


    analyze_btn = st.button("🔍 Analyze Routes", type="primary")



# Index,Primary Driver
# I_Driver,"Fatigue, Mileage, Crash History"
# I_Task,"Emergency Urgency, Familiarity (Route/Vehicle)"
# I_Vechicle​,"Vehicle Mass/Size, Condition (Age + Maintenance), Tech"
# I_Environment​,"Visibility (Time + Weather), Traffic"
# I_Roadway​,"Infrastructure, Hotspots, Road Class"

# 5096 Driver+Task 95%
# 130 Vehicle 2.4%
# 17 Environment  0.3%
# 94 Roadway 1.7%

# ==============================
# 5. EXECUTION & VISUALIZATION
# ==============================
if analyze_btn:
    try:
        # 1. ELIGIBILITY CHECK
        valid_map = {
            "A": ["Car"], 
            "B": ["Car", "Light Truck"], 
            "C": ["Car", "Light Truck", "5T"], 
            "D": ["Car", "Light Truck", "5T", "10T"]
        }
        
        # Check if the selected vehicle is permitted for the selected category
        allowed_vehicles = valid_map.get(d_cat, [])
        if v_type not in allowed_vehicles:
            st.error(f"⚠️ Eligibility Error: Driver Category {d_cat} is not permitted to operate {v_type} vehicles.")
            st.stop() # Stops execution here so routing doesn't happen      

        # Geocoding
        start_coords = ox.geocoder.geocode(start_loc)
        end_coords = ox.geocoder.geocode(end_loc)
        orig = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
        dest = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

        # Global Indices (Static)
        inputs = {
            'mileage': mileage, 'crash': crash, 'fatigue': fatigue, 
            'task_exp': task_exp, 'purpose': purpose, 'v_type': v_type, 
            'maint': maint, 'tech': tech
        }
        p, a = compute_global_indices(inputs)

        # Pathfinding & Route-Specific Risk
        results = []
        temp_DG = DG.copy()
        for i in range(3):
            try:
                path = nx.shortest_path(temp_DG, orig, dest, weight="length")
                
                # Dynamic Environment Index (E) computed per route per segment
                e = compute_route_env_index(path, traffic, weather, dep_time, DG)
                t = compute_route_task_index(path, inputs, DG)
                r = compute_route_roadway_index(path, DG)
                
                # Risk calculation with route-specific E
                risk_data = segment_level_risk(path, p, t, a, e, r, DG)
                results.append({"id": f"Route {i+1}", "route": path, "risk": risk_data, "e": e})
                
                # Penalty for route diversity
                for u, v in zip(path[:-1], path[1:]): temp_DG[u][v]["length"] *= 2.5
            except nx.NetworkXNoPath: break

        # SAVE TO SESSION STATE
        st.session_state.routes = results
        st.session_state.coords = start_coords
        st.session_state.indices = (p, t, a)
        
    except Exception as err:
        st.error(f"Routing failed: {err}")

# 2. RENDER DASHBOARD & MAP (Always runs if data exists)
if st.session_state.routes:
    # Table 1: Risk Attribution
    # st.subheader("📋 Risk Attribution")
    # p, t, a = st.session_state.indices
    # factors = []
    # for r in st.session_state.routes:
    #     factors.append({"Route": r['id'], "Driver": f"{p:.2f}", "Task": f"{t:.2f}", "Vehicle": f"{a:.2f}", "Environment": f"{r['e']:.2f}", 
    #                     "Hotspots": r['risk']['counts']['hotspots'], "Sharp/Narrow": r['risk']['counts']['sharp'] + r['risk']['counts']['narrow']})
    # st.table(pd.DataFrame(factors).set_index("Route"))

    # Table 2: Performance
    st.subheader("📋 Performance Metrics")


    outcomes = []
    for r in st.session_state.routes:
        # Calculate an overall category for the table
        # We'll use the average risk per meter to define the route's overall status
        avg_risk = sum(s[2] for s in r['risk']['risks'])/len(r['risk']['risks'])
        level = risk_category(avg_risk)
        
        # Adding a color emoji for better scannability
        status_emoji = "🟢" if level == "Low" else "🟠" if level == "Medium" else "🔴"

        outcomes.append({"Route": r['id'], "Dist": f"{r['risk']['dist']:.2f}km", "Prob": f"{r['risk']['prob']*100:.2f}%", 
                 "Risk": f"{status_emoji} {level}"})


    st.table(pd.DataFrame(outcomes).set_index("Route"))

    m = folium.Map(location=st.session_state.coords, zoom_start=13)

    legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:12px;
                    background-color:white; opacity: 0.9; padding: 10px; border-radius:5px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
        <b>Risk Levels</b><br>
        <i style="background: green; width: 12px; height: 12px; display: inline-block;"></i> Low Risk<br>
        <i style="background: orange; width: 12px; height: 12px; display: inline-block;"></i> Medium Risk<br>
        <i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> High Risk<br>
        <hr style="margin:5px 0;">
        <span style="color:black;">⚠️ Hotspot</span>
        </div>
        '''
    m.get_root().html.add_child(folium.Element(legend_html))

    for r in st.session_state.routes:
        # 1. Plot Risk-Colored Segments
        for u, v, risk_per_meter in r['risk']['risks']:
            color = "green" if risk_per_meter < 2e-6 else "orange" if risk_per_meter < 5e-6 else "red"
            # Extract coordinates directly from the Graph 'G'
            points = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
            folium.PolyLine(points, color=color, weight=5, opacity=0.8).add_to(m)

        # 2. Add Route Label (Centered)
        mid_idx = len(r['route']) // 2
        mid_node = r['route'][mid_idx]
        unique_hotspots = set(r['risk'].get('hotspot_coords', []))
        for lat, lon in unique_hotspots:
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(html="""
                    <div style="
                        font-size: 20px; 
                        color: #FFD700; 
                        text-shadow: 1px 1px 2px black;
                        z-index: 100; 
                        position: relative;">
                        ⚠️
                    </div>"""),
                tooltip="Collision Hotspot"
            ).add_to(m)

        # 2. Plot Route Label (High z-index, so they sit on top)
        folium.Marker(
            location=[G.nodes[mid_node]["y"], G.nodes[mid_node]["x"]],
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-family: sans-serif; color: white; background-color: #333; 
                    border-radius: 50%; width: 24px; height: 24px; display: flex; 
                    justify-content: center; align-items: center; font-weight: bold; 
                    border: 2px solid white; box-shadow: 0px 0px 5px black;
                    z-index: 500; 
                    position: relative;">
                    {r['id'].split()[-1]}
                </div>"""),
        ).add_to(m)
            
    st_folium(m, width=1200, height=500)

