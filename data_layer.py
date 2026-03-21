"""
data_layer.py
=============
Fleet Risk Simulator — Data Layer
All synthetic data generation and database simulation lives here.
No Streamlit, no risk-engine logic — pure data.

Usage
-----
    from data_layer import DriverDB, VehicleDB, EnvironmentSim, RouteFeatureExtractor
"""

import random
import hashlib
from datetime import time, date, timedelta, datetime
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 0.  ENUMERATIONS / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

VEHICLE_TYPES = ["Car", "Light Truck", "5T", "10T"]

# Cat → vehicles the driver is licensed to operate
VEHICLE_ELIGIBILITY: dict[str, list[str]] = {
    "A": ["Car"],
    "B": ["Car", "Light Truck"],
    "C": ["Car", "Light Truck", "5T"],
    "D": ["Car", "Light Truck", "5T", "10T"],
}

MILEAGE_BANDS = ["<20k", "20k-100k", "100k-300k", ">300k"]

SAFE_TECH_OPTIONS = ["ABS", "Reverse Cam", "Blind Spot Monitor"]

WEATHER_CONDITIONS = ["Clear", "Rain", "Heavy Rain"]

ROAD_CLASS_LABELS = {
    "motorway":   "Highway",
    "trunk":      "Highway",
    "primary":    "Major",
    "secondary":  "Major",
    "tertiary":   "Minor",
    "residential":"Minor",
    "unclassified":"Minor",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA CLASSES  (typed records — the "schema")
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DriverProfile:
    """One row in the driver database."""
    driver_id:        str
    name:             str
    category:         str               # A / B / C / D
    mileage_band:     str               # "<20k" … ">300k"
    crash_history:    bool              # any at-fault incident on record
    shift_start_time: time              # daily shift start (HH:MM)
    # --- computed at query time (not stored) ---
    # fatigue_hours   : float  ← dep_time - shift_start_time
    # route_experience: dict   ← queried from TripLogDB
    # vehicle_experience: set  ← queried from TripLogDB


@dataclass
class VehicleProfile:
    """One row in the vehicle database."""
    vehicle_number: str
    vehicle_type:   str                 # Car / Light Truck / 5T / 10T
    manufacture_year: int               # used to derive age → condition
    maintenance_records: list[date]     # dates of last services
    safe_tech:      list[str]           # subset of SAFE_TECH_OPTIONS
    # --- computed at query time ---
    # vehicle_age    : int   ← current_year - manufacture_year
    # condition      : str   ← "Good" / "Fair" / "Poor"  (derived)


@dataclass
class TripLog:
    """
    Historical trip record — used to derive route_experience
    and vehicle_experience per driver.
    """
    trip_id:        str
    driver_id:      str
    vehicle_number: str
    origin:         str
    destination:    str
    route_index:    int                 # 0 / 1 / 2  (which of the 3 candidate routes)
    trip_date:      date


@dataclass
class WeatherSnapshot:
    """
    Simulated (or future: live-API) weather at departure.

    Downstream effects derived at construction time:
      braking_mult        — multiplier on vehicle stopping distance (used in Index A)
      traffic_weather_mult— how much weather degrades traffic flow (applied in query_traffic)
      visibility_km       — base daytime visibility given weather (reduced further at night)
    """
    condition:            str     # Clear / Rain / Heavy Rain
    visibility_km:        float   # daytime visibility, km (0–10)
    braking_mult:         float   # >1.0 means worse braking (wet road)
    traffic_weather_mult: float   # >1.0 means denser/slower traffic due to weather


@dataclass
class TrafficSegment:
    """Per-edge traffic density (u, v) → density 1-5."""
    u: int
    v: int
    density: int                        # 1 (free) … 5 (heavy)


@dataclass
class RouteFeatures:
    """
    Roadway features extracted from the graph for one candidate route.
    Produced by RouteFeatureExtractor.
    """
    route_index:    int
    node_sequence:  list[int]
    dist_km:        float
    hotspot_count:  int
    sharp_turn_count: int
    narrow_road_count: int
    road_class_breakdown: dict[str, float]   # label → fraction of route length
    segment_traffic: list[TrafficSegment]    # one per edge


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DRIVER DATABASE
# ──────────────────────────────────────────────────────────────────────────────

class DriverDB:
    """
    Generates and stores a synthetic pool of drivers.
    In production, replace _generate() with a real DB query.
    """

    # Seed data — extend as needed
    _SEED = [
        # (name,          id,     cat, mileage_band,  crash, shift_start)
        ("SGT Rahman",   "D001", "D", ">300k",       False, time(6,  0)),
        ("CPL Lim",      "D002", "C", "100k-300k",   False, time(7,  0)),
        ("PTE Krishnan", "D003", "B", "20k-100k",    False, time(8,  0)),
        ("SGT Tan",      "D004", "C", "100k-300k",   True,  time(6, 30)),
        ("CPL Wong",     "D005", "B", "20k-100k",    False, time(9,  0)),
        ("LTA Ng",       "D006", "D", ">300k",       False, time(5, 30)),
        ("PTE Ali",      "D007", "A", "<20k",        False, time(7, 30)),
        ("CPL Chen",     "D008", "C", "100k-300k",   False, time(7,  0)),
        ("SGT Yeo",      "D009", "B", "20k-100k",    True,  time(6,  0)),
        ("CPL Muthu",    "D010", "C", ">300k",       False, time(7, 30)),
        ("PTE Ismail",   "D011", "A", "<20k",        False, time(8, 30)),
        ("LTA Chua",     "D012", "D", "100k-300k",   False, time(5,  0)),
        ("SGT Hasan",    "D013", "C", "20k-100k",    False, time(8,  0)),
        ("CPL Siva",     "D014", "B", "100k-300k",   False, time(7,  0)),
        ("PTE Ong",      "D015", "A", "<20k",        True,  time(9,  0)),
    ]

    def __init__(self):
        self._drivers: dict[str, DriverProfile] = {}
        for row in self._SEED:
            name, did, cat, mileage, crash, shift = row
            self._drivers[did] = DriverProfile(
                driver_id        = did,
                name             = name,
                category         = cat,
                mileage_band     = mileage,
                crash_history    = crash,
                shift_start_time = shift,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def all(self) -> list[DriverProfile]:
        return list(self._drivers.values())

    def get(self, driver_id: str) -> Optional[DriverProfile]:
        return self._drivers.get(driver_id)

    def eligible_for(self, vehicle_type: str) -> list[DriverProfile]:
        """Return drivers whose category permits operating vehicle_type."""
        return [d for d in self._drivers.values()
                if vehicle_type in VEHICLE_ELIGIBILITY.get(d.category, [])]

    @staticmethod
    def compute_fatigue(driver: DriverProfile, departure_time: time) -> float:
        """
        Hours since last break = departure_time − shift_start_time.
        Assumes same calendar day; clamps to [0, 16].
        """
        dep_dt   = datetime.combine(date.today(), departure_time)
        shift_dt = datetime.combine(date.today(), driver.shift_start_time)
        delta    = (dep_dt - shift_dt).total_seconds() / 3600
        return max(0.0, min(16.0, delta))


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VEHICLE DATABASE
# ──────────────────────────────────────────────────────────────────────────────

class VehicleDB:
    """
    Generates and stores a synthetic pool of vehicles.
    """

    _SEED = [
        # (plate,     type,          year, last_services (ISO),          safe_tech)
        ("SG1001A", "Car",          2022, ["2024-11-01","2024-05-01"],  ["ABS","Reverse Cam","Blind Spot Monitor"]),
        ("SG1002B", "Light Truck",  2019, ["2024-10-15"],               ["ABS","Reverse Cam"]),
        ("SG1003C", "5T",           2016, ["2024-09-20","2024-03-10"],  ["ABS"]),
        ("SG1004D", "10T",          2014, ["2023-08-01"],               ["ABS"]),
        ("SG1005E", "Car",          2023, ["2025-01-10","2024-07-05"],  ["ABS","Reverse Cam","Blind Spot Monitor"]),
        ("SG1006F", "Light Truck",  2020, ["2024-12-01"],               ["ABS","Blind Spot Monitor"]),
        ("SG1007G", "5T",           2018, ["2024-06-15","2024-01-20"],  ["ABS","Reverse Cam"]),
        ("SG1008H", "Car",          2021, ["2025-02-01"],               ["ABS","Reverse Cam"]),
        ("SG1009I", "Light Truck",  2017, ["2023-11-10"],               ["ABS"]),
        ("SG1010J", "10T",          2015, ["2024-04-22","2023-10-01"],  ["ABS","Blind Spot Monitor"]),
    ]

    # Condition thresholds (age in years, months since last service)
    _AGE_POOR  = 10   # years
    _AGE_FAIR  = 6
    _SVC_POOR  = 18   # months since last service → Poor
    _SVC_FAIR  = 9

    def __init__(self):
        self._vehicles: dict[str, VehicleProfile] = {}
        for row in self._SEED:
            plate, vtype, year, services, tech = row
            self._vehicles[plate] = VehicleProfile(
                vehicle_number    = plate,
                vehicle_type      = vtype,
                manufacture_year  = year,
                maintenance_records = [date.fromisoformat(s) for s in services],
                safe_tech         = tech,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def all(self) -> list[VehicleProfile]:
        return list(self._vehicles.values())

    def get(self, vehicle_number: str) -> Optional[VehicleProfile]:
        return self._vehicles.get(vehicle_number)

    @staticmethod
    def compute_age(vehicle: VehicleProfile) -> int:
        return date.today().year - vehicle.manufacture_year

    @staticmethod
    def compute_condition(vehicle: VehicleProfile) -> str:
        """
        Derive condition from age + recency of last maintenance.
        Returns "Good" / "Fair" / "Poor".
        """
        age = date.today().year - vehicle.manufacture_year
        if vehicle.maintenance_records:
            last_svc     = max(vehicle.maintenance_records)
            months_since = (date.today() - last_svc).days / 30.44
        else:
            months_since = 999   # no record → treat as worst case

        if age >= VehicleDB._AGE_POOR or months_since >= VehicleDB._SVC_POOR:
            return "Poor"
        elif age >= VehicleDB._AGE_FAIR or months_since >= VehicleDB._SVC_FAIR:
            return "Fair"
        return "Good"


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRIP LOG  (route & vehicle experience)
# ──────────────────────────────────────────────────────────────────────────────

class TripLogDB:
    """
    Simulates a historical trip ledger.
    Each log entry records which of the 3 candidate route indices (0/1/2)
    the driver used for a given O-D pair, so we can compute per-route
    familiarity for any driver × O-D × route-index combination.
    """

    # Familiarity thresholds — trips on a *specific* route index
    THRESH_HIGH   = 3   # ≥ 5 trips on this exact route → High
    THRESH_MEDIUM = 1   # ≥ 2 trips                     → Medium
                        # < 2                            → Low

    def __init__(self, driver_db: DriverDB, vehicle_db: VehicleDB, seed: int = 99):
        self._logs: list[TripLog] = []
        self._generate(driver_db, vehicle_db, seed)

    def _generate(self, driver_db: DriverDB, vehicle_db: VehicleDB, seed: int):
        rng = random.Random(seed)
        drivers  = driver_db.all()
        vehicles = vehicle_db.all()

        # All known locations — must include all place names that could appear
        # as origin/destination inputs so matching works.
        locations = [
            "Orchard Road, Singapore", "Changi Airport, Singapore", "Marina Bay Sands, Singapore",
            "Jurong East, Singapore",  "Woodlands, Singapore",      "Tampines, Singapore",
            "Bishan, Singapore",       "Toa Payoh, Singapore",       "Ang Mo Kio, Singapore",
            "Clementi, Singapore",     "Buona Vista, Singapore",     "Tuas Link, Singapore",
        ]
        today = date.today()

        # Build all directed O-D pairs (both directions)
        od_pairs = [(o, d) for o in locations for d in locations if o != d]

        for driver in drivers:
            # Scale trip budget generously so hit-rate is meaningful:
            # "<20k" → 30 trips, ">300k" → 600 trips
            n_trips = {"<20k":24, "20k-100k": 100,
                       "100k-300k": 300, ">300k": 600}[driver.mileage_band]
            eligible_vehicles = [v for v in vehicles
                                 if v.vehicle_type in VEHICLE_ELIGIBILITY.get(driver.category, [])]
            if not eligible_vehicles:
                continue

            # Weight O-D pairs so a few "familiar" corridors are driven repeatedly
            # — this gives realistic High/Medium/Low spread
            favourite_ods = rng.sample(od_pairs, min(6, len(od_pairs)))
            weights = []
            for od in od_pairs:
                weights.append(8 if od in favourite_ods else 1)

            for i in range(n_trips):
                origin, dest = rng.choices(od_pairs, weights=weights, k=1)[0]
                trip_date    = today - timedelta(days=rng.randint(1, 730))
                vehicle      = rng.choice(eligible_vehicles)
                # route_index: drivers tend to stick to one preferred route per corridor
                # simulate by biasing toward route 0 or 1 per O-D
                od_seed      = hash((driver.driver_id, origin, dest)) % (2**16)
                od_rng       = random.Random(od_seed + seed)
                pref_route   = od_rng.randint(0, 2)
                route_index  = pref_route if od_rng.random() < 0.6 else rng.randint(0, 2)       
                route_index  = rng.randint(0, 2)

                self._logs.append(TripLog(
                    trip_id        = f"{driver.driver_id}-{i:05d}",
                    driver_id      = driver.driver_id,
                    vehicle_number = vehicle.vehicle_number,
                    origin         = origin,
                    destination    = dest,
                    route_index    = route_index,
                    trip_date      = trip_date,
                ))

    # ── Public API ────────────────────────────────────────────────────────────

    def route_experience(self, driver_id: str, origin: str, destination: str) -> dict:
        """
        Return per-route-index trip counts for a driver on a given O-D pair.

        route_idx 0 / 1 / 2 correspond to the three candidate paths returned
        by the graph pathfinder — the same indices stored in RouteFeatures.

        Returns
        -------
        {
            "by_route"   : {0: int, 1: int, 2: int},
            "total_trips": int,          # sum across all route indices
            "familiarity": str           # O-D level: High/Medium/Low (for display)
        }
        Use per_route_familiarity(result, route_idx) to get the per-route level.
        """
        def _norm(s: str) -> str:
            return s.lower().split(",")[0].strip()

        o, d   = _norm(origin), _norm(destination)
        by_route: dict[int, int] = {0: 0, 1: 0, 2: 0}

        for log in self._logs:
            if (log.driver_id == driver_id
                    and _norm(log.origin)      == o
                    and _norm(log.destination) == d):
                by_route[log.route_index] = by_route.get(log.route_index, 0) + 1

        total = sum(by_route.values())
        # O-D level familiarity for summary display
        od_fam = ("High"   if total >= self.THRESH_HIGH * 2
                  else "Medium" if total >= self.THRESH_MEDIUM * 2
                  else "Low")
        return {"by_route": by_route, "total_trips": total, "familiarity": od_fam}

    @staticmethod
    def per_route_familiarity(route_exp: dict, route_idx: int) -> str:
        """
        Return High/Medium/Low familiarity for a *specific* route index.
        This is what compute_T should use, not the O-D aggregate.
        """
        trips = route_exp["by_route"].get(route_idx, 0)
        if trips >= TripLogDB.THRESH_HIGH:
            return "High"
        if trips >= TripLogDB.THRESH_MEDIUM:
            return "Medium"
        return "Low"

    def vehicle_type_experience(self, driver_id: str, vehicle_db: VehicleDB) -> dict:
        """Return trip counts broken down by vehicle TYPE (not plate)."""
        type_counts: dict[str, int] = {}
        for log in self._logs:
            if log.driver_id != driver_id:
                continue
            v = vehicle_db.get(log.vehicle_number)
            if v:
                vt = v.vehicle_type
                type_counts[vt] = type_counts.get(vt, 0) + 1
        return type_counts


# ──────────────────────────────────────────────────────────────────────────────
# 5.  ENVIRONMENT SIMULATOR  (weather + traffic)
# ──────────────────────────────────────────────────────────────────────────────

class EnvironmentSim:
    """
    Simulates weather and per-segment traffic.
    In production, replace with live API calls (e.g., NEA, Google Traffic).
    """

    # Hour → base traffic weight [1-5]
    _HOUR_TRAFFIC = {
        **{h: 2 for h in range(0,  6)},   # midnight – 5 am : light
        **{h: 4 for h in range(6,  9)},   # 6–8 am AM peak
        **{h: 3 for h in range(9, 17)},   # 9–4 pm daytime
        **{h: 5 for h in range(17, 20)},  # 5–7 pm PM peak
        **{h: 2 for h in range(20, 24)},  # evening : light
    }

    _WEATHER_PROB = {               # rough Singapore probabilities
        "Clear":      0.60,
        "Rain":       0.30,
        "Heavy Rain": 0.10,
    }

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng  = random.Random(seed)

    # Weather condition → downstream effect tables
    # braking_mult: wet/flooded roads increase stopping distance
    _BRAKING_MULT = {"Clear": 1.0, "Rain": 1.25, "Heavy Rain": 1.55}

    # traffic_weather_mult: rain forces drivers to slow down, increasing effective density
    _TRAFFIC_WEATHER_MULT = {"Clear": 1.0, "Rain": 1.2, "Heavy Rain": 1.5}

    # daytime base visibility by weather condition (km)
    _BASE_VIS_KM = {"Clear": 10.0, "Rain": 5.0, "Heavy Rain": 2.0}

    # night reduces visibility further regardless of weather
    _NIGHT_VIS_CAP_KM = 4.0   # maximum visibility at night even in clear weather

    # ── Weather ───────────────────────────────────────────────────────────────

    def query_weather(self, departure_time: time) -> WeatherSnapshot:
        """
        Simulate a weather query for the given departure time.
        Deterministic per (seed, hour) — all drivers see the same weather.
        Populates all downstream effect fields on WeatherSnapshot.
        """
        hour      = departure_time.hour
        hour_seed = int(hashlib.md5(f"weather-{self._seed}-{hour}".encode()).hexdigest(), 16) % (2**32)
        hour_rng  = random.Random(hour_seed)

        probs = dict(self._WEATHER_PROB)
        if 13 <= hour <= 17:          # afternoon convective rain (Singapore pattern)
            probs["Rain"]       += 0.10
            probs["Heavy Rain"] += 0.05
            probs["Clear"]      -= 0.15

        condition = hour_rng.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        vis_km    = self._BASE_VIS_KM[condition] + hour_rng.uniform(-0.5, 0.5)
        vis_km    = round(max(0.5, min(10.0, vis_km)), 1)

        return WeatherSnapshot(
            condition            = condition,
            visibility_km        = vis_km,
            braking_mult         = self._BRAKING_MULT[condition],
            traffic_weather_mult = self._TRAFFIC_WEATHER_MULT[condition],
        )

    # ── Traffic ───────────────────────────────────────────────────────────────
 
    # Road class → traffic capacity modifier.
    # Motorways move traffic faster (lower density for same volume);
    # residential streets have lower capacity (higher density).
    # These are multipliers on top of the hour base, clamped to [1, 5].
    _ROAD_CLASS_TRAFFIC = {
        "motorway":    0.6,   # fast-moving, low density
        "trunk":       0.75,
        "primary":     0.9,
        "secondary":   1.1,
        "tertiary":    1.25,
        "residential": 1.5,   # slow, high density
        "unclassified":1.3,
    }
 
    def query_traffic(self, route_nodes: list[int], departure_time: time,
                      weather: WeatherSnapshot, DG=None) -> list[TrafficSegment]:
        """
        Generate per-segment traffic density for a given route.
 
        Density = f(hour_base, road_class, weather, per-edge noise)
        - hour_base   : time-of-day pattern from _HOUR_TRAFFIC
        - road_class  : motorways flow faster than residential (from DG edge data)
        - weather     : rain degrades flow via traffic_weather_mult
        - noise       : deterministic ±0.5 jitter per (u, v, hour) for realism
 
        This means different routes get genuinely different avg densities because
        they traverse different road classes, not just different edges of the same class.
        DG is optional: if None, road-class adjustment is skipped (fallback).
        """
        hour      = departure_time.hour
        hour_base = self._HOUR_TRAFFIC.get(hour, 3)
        segments  = []
 
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            # Road-class modifier from edge attributes
            if DG is not None:
                edge = DG.get_edge_data(u, v) or {}
                hw   = edge.get("highway", "unclassified")
                # highway can be a list in OSM data
                if isinstance(hw, list):
                    hw = hw[0]
                road_mult = self._ROAD_CLASS_TRAFFIC.get(hw, 1.3)
            else:
                road_mult = 1.0
 
            # Deterministic per-edge jitter: stable across reruns, varies by edge
            edge_seed = int(hashlib.md5(f"{u}-{v}-{hour}".encode()).hexdigest(), 16) % (2**32)
            edge_rng  = random.Random(edge_seed)
            jitter    = edge_rng.uniform(-3, 3)   # ±0.5 continuous noise
 
            # Combine: hour × road_class × weather × noise, clamp [1, 5]
            raw     = (hour_base * road_mult + jitter) * weather.traffic_weather_mult
            density = max(1, min(5, round(raw, 1)))
            segments.append(TrafficSegment(u=u, v=v, density=density))
 
        return segments

    # ── Visibility ────────────────────────────────────────────────────────────

    @staticmethod
    def is_night(departure_time: time) -> bool:
        return departure_time.hour < 7 or departure_time.hour > 19

    @staticmethod
    def effective_visibility_km(snapshot: WeatherSnapshot, departure_time: time) -> float:
        """
        Effective visibility = weather visibility, further capped at night.
        Night caps visibility at _NIGHT_VIS_CAP_KM regardless of weather.
        """
        if EnvironmentSim.is_night(departure_time):
            return round(min(snapshot.visibility_km, EnvironmentSim._NIGHT_VIS_CAP_KM), 1)
        return snapshot.visibility_km

    @staticmethod
    def visibility_mult(snapshot: WeatherSnapshot, departure_time: time) -> float:
        """
        Visibility risk multiplier for Index E.
        Lower effective visibility → higher multiplier.
        Anchored so that 10 km (clear day) = 1.0, 0.5 km (heavy rain + night) ≈ 2.0.
        """
        eff_vis = EnvironmentSim.effective_visibility_km(snapshot, departure_time)
        # Linear interpolation: vis 10→1.0, vis 0.5→2.0
        return round(1.0 + (10.0 - eff_vis) / 9.5 * 1.0, 3)

    @staticmethod
    def visibility_label(snapshot: WeatherSnapshot, departure_time: time) -> str:
        eff = EnvironmentSim.effective_visibility_km(snapshot, departure_time)
        tod = "Night" if EnvironmentSim.is_night(departure_time) else "Day"
        return f"{tod} + {snapshot.condition} → {eff:.1f} km"


# ──────────────────────────────────────────────────────────────────────────────
# 6.  ROUTE FEATURE EXTRACTOR  (roadway data from the graph)
# ──────────────────────────────────────────────────────────────────────────────

class RouteFeatureExtractor:
    """
    Given a NetworkX DiGraph (pre-processed with hotspot / sharp / narrow flags)
    and a list of node sequences, extract structured RouteFeatures.
    """

    def __init__(self, DG, traffic_sim: EnvironmentSim):
        self._DG  = DG
        self._sim = traffic_sim

    def extract(self, route_index: int, node_sequence: list[int],
                departure_time: time, weather: WeatherSnapshot) -> RouteFeatures:
        DG = self._DG
        total_len     = 0.0
        hotspot_count = 0
        sharp_count   = 0
        narrow_count  = 0
        road_class_lengths: dict[str, float] = {}

        for u, v in zip(node_sequence[:-1], node_sequence[1:]):
            edge   = DG.get_edge_data(u, v) or {}
            length = edge.get("length", 0.0)
            total_len += length

            if edge.get("hotspot_mult", 1.0) > 1.0: hotspot_count += 1
            if edge.get("is_sharp"):                 sharp_count   += 1
            if edge.get("is_narrow"):                narrow_count  += 1

            hw    = edge.get("highway", "unclassified")
            label = ROAD_CLASS_LABELS.get(hw, "Minor")
            road_class_lengths[label] = road_class_lengths.get(label, 0.0) + length

        dist_km    = total_len / 1000.0
        breakdown  = {k: round(v / max(total_len, 1), 3)
                      for k, v in road_class_lengths.items()}
        traffic    = self._sim.query_traffic(node_sequence, departure_time, weather)

        return RouteFeatures(
            route_index          = route_index,
            node_sequence        = node_sequence,
            dist_km              = round(dist_km, 3),
            hotspot_count        = hotspot_count,
            sharp_turn_count     = sharp_count,
            narrow_road_count    = narrow_count,
            road_class_breakdown = breakdown,
            segment_traffic      = traffic,
        )

    def avg_traffic_density(self, features: RouteFeatures) -> float:
        """Average density across all segments of the route."""
        segs = features.segment_traffic
        if not segs:
            return 3.0
        return sum(s.density for s in segs) / len(segs)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ASSEMBLED QUERY  — single call to get everything for one task
# ──────────────────────────────────────────────────────────────────────────────

def build_task_context(
    vehicle_number: str,
    origin:         str,
    destination:    str,
    purpose:        str,
    departure_time: time,
    route_sequences: list[list[int]],   # up to 3 node lists from pathfinder
    DG,
    driver_db:   DriverDB,
    vehicle_db:  VehicleDB,
    trip_log_db: TripLogDB,
    env_sim:     EnvironmentSim,
) -> dict:
    """
    Master query: given task inputs, return a fully resolved context dict
    containing vehicle info, environment snapshot, route features, and
    per-driver derived factors for every eligible driver.

    This is the single function the UI / risk engine calls.

    Returns
    -------
    {
        "vehicle"     : { profile, age, condition },
        "environment" : { weather_snapshot, is_night, visibility_label },
        "routes"      : [ RouteFeatures, ... ],          # one per candidate
        "drivers"     : [                                # eligible only
            {
                "profile"           : DriverProfile,
                "fatigue_hours"     : float,
                "route_experience"  : { total_trips, by_route, familiarity },
                "vehicle_experience": { type_counts },
                "has_driven_type"   : bool,
            }, ...
        ]
    }
    """
    extractor = RouteFeatureExtractor(DG, env_sim)

    # ── Vehicle ───────────────────────────────────────────────────────────────
    vehicle   = vehicle_db.get(vehicle_number)
    if vehicle is None:
        raise ValueError(f"Vehicle '{vehicle_number}' not found in database.")

    vehicle_age       = VehicleDB.compute_age(vehicle)
    vehicle_condition = VehicleDB.compute_condition(vehicle)

    # ── Environment ───────────────────────────────────────────────────────────
    weather_snapshot = env_sim.query_weather(departure_time)
    night_flag       = EnvironmentSim.is_night(departure_time)
    vis_label        = EnvironmentSim.visibility_label(weather_snapshot, departure_time)

    # ── Routes ────────────────────────────────────────────────────────────────
    # Weather must be resolved first — traffic density depends on it
    route_features = [
        extractor.extract(i, seq, departure_time, weather_snapshot)
        for i, seq in enumerate(route_sequences)
    ]

    # ── Drivers ───────────────────────────────────────────────────────────────
    eligible_drivers = driver_db.eligible_for(vehicle.vehicle_type)
    driver_contexts  = []

    for driver in eligible_drivers:
        fatigue  = DriverDB.compute_fatigue(driver, departure_time)
        route_exp = trip_log_db.route_experience(driver.driver_id, origin, destination)
        vtype_exp = trip_log_db.vehicle_type_experience(driver.driver_id, vehicle_db)
        has_driven_type = vehicle.vehicle_type in vtype_exp and vtype_exp[vehicle.vehicle_type] > 0

        driver_contexts.append({
            "profile":            driver,
            "fatigue_hours":      round(fatigue, 2),
            "route_experience":   route_exp,
            "vehicle_type_exp":   vtype_exp,
            "has_driven_type":    has_driven_type,
        })

    return {
        "vehicle": {
            "profile":   vehicle,
            "age":       vehicle_age,
            "condition": vehicle_condition,
        },
        "environment": {
            "weather":         weather_snapshot,
            "is_night":        night_flag,
            "visibility_label": vis_label,
        },
        "routes":  route_features,
        "drivers": driver_contexts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8.  QUICK SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pprint import pprint

    driver_db  = DriverDB()
    vehicle_db = VehicleDB()
    trip_db    = TripLogDB(driver_db, vehicle_db)
    env_sim    = EnvironmentSim()

    # --- Driver smoke test ---
    print("=== Eligible drivers for 5T ===")
    for d in driver_db.eligible_for("5T"):
        fat = DriverDB.compute_fatigue(d, time(10, 0))
        print(f"  {d.driver_id} {d.name:<16} Cat={d.category} "
              f"Shift={d.shift_start_time} Fatigue@10:00={fat:.1f}h")

    # --- Vehicle smoke test ---
    print("\n=== Vehicle SG1003C ===")
    v  = vehicle_db.get("SG1003C")
    print(f"  Type={v.vehicle_type}  Age={VehicleDB.compute_age(v)}yr  "
          f"Condition={VehicleDB.compute_condition(v)}  Tech={v.safe_tech}")

    # --- Trip log smoke test ---
    print("\n=== Route experience D001: Orchard Road → Changi Airport ===")
    for driver_id in ['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015']:
        print(driver_id)
        pprint(trip_db.route_experience(driver_id, "Orchard Road, Singapore", "Changi Airport, Singapore"))

    print("\n=== Vehicle type experience D001 ===")
    pprint(trip_db.vehicle_type_experience("D001", vehicle_db))

    # --- Environment smoke test ---
    print("\n=== Weather @ 14:30 ===")
    snap = env_sim.query_weather(time(14, 30))
    print(f"  {snap.condition}  vis={snap.visibility_km} km")
    print(f"  Visibility label: {EnvironmentSim.visibility_label(snap, time(14,30))}")

    print("\nAll smoke tests passed ✓")

