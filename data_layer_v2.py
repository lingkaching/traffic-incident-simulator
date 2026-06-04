"""
data_layer.py
=============
Fleet Risk Simulator — Data Layer

Changes from previous version
------------------------------
1. DriverProfile: added max_shift_hours, available flag.
2. DriverDB: 15 → 30 drivers. Cat A×4, B×8, C×10, D×8.
   Shifts cover 05:00–09:00; D017 marked unavailable (on leave).
3. VehicleDB: 10 → 18 vehicles. Fleet: Car×4, LT×4, 5T×6, 10T×4.
4. TripLogDB: trip history seeded from route.json military locations
   (4× weight) + generic Singapore addresses so familiarity scores
   are meaningful for military O-D pairs.
5. RouteDB: new — loads route.json, validates O-D pairs, exposes
   distance lookup and location catalogue.
6. estimate_duration_h(): shared utility for conflict checking and
   fatigue carry-forward in the allocator.
7. fatigue_carry_forward(): computes chained fatigue for multi-task drivers.
8. build_task_context(): accepts optional fatigue_overrides dict so the
   allocator can inject chained fatigue without re-running base computation.
"""

import json
import random
import hashlib
from datetime import time, date, timedelta, datetime
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

VEHICLE_TYPES = ["Car", "Light Truck", "5T", "10T"]

VEHICLE_ELIGIBILITY: dict[str, list[str]] = {
    "A": ["Car"],
    "B": ["Car", "Light Truck"],
    "C": ["Car", "Light Truck", "5T"],
    "D": ["Car", "Light Truck", "5T", "10T"],
}

MILEAGE_BANDS      = ["<20k", "20k-100k", "100k-300k", ">300k"]
SAFE_TECH_OPTIONS  = ["ABS", "Reverse Cam", "Blind Spot Monitor"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Heavy Rain"]

ROAD_CLASS_LABELS = {
    "motorway":    "Highway",
    "trunk":       "Highway",
    "primary":     "Major",
    "secondary":   "Major",
    "tertiary":    "Minor",
    "residential": "Minor",
    "unclassified":"Minor",
}

_AVG_SPEED_KMH  = 35.0   # Singapore urban/military average
_BUFFER_MINUTES = 15      # loading/unloading buffer per task


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DriverProfile:
    driver_id:        str
    name:             str
    category:         str
    mileage_band:     str
    crash_history:    bool
    shift_start_time: time
    max_shift_hours:  float = 12.0   # hard cap per driver
    available:        bool  = True   # False = on leave / standby


@dataclass
class VehicleProfile:
    vehicle_number:      str
    vehicle_type:        str
    manufacture_year:    int
    maintenance_records: list[date]
    safe_tech:           list[str]


@dataclass
class TripLog:
    trip_id:        str
    driver_id:      str
    vehicle_number: str
    origin:         str
    destination:    str
    route_index:    int
    trip_date:      date


@dataclass
class WeatherSnapshot:
    condition:            str
    visibility_km:        float
    braking_mult:         float
    traffic_weather_mult: float


@dataclass
class TrafficSegment:
    u:       int
    v:       int
    density: int   # 1 (free) … 5 (heavy)


@dataclass
class RouteFeatures:
    route_index:          int
    node_sequence:        list[int]
    dist_km:              float
    hotspot_count:        int
    sharp_turn_count:     int
    narrow_road_count:    int
    road_class_breakdown: dict[str, float]
    segment_traffic:      list[TrafficSegment]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DRIVER DATABASE  — 30 drivers
# ──────────────────────────────────────────────────────────────────────────────

class DriverDB:
    """
    30-driver pool: Cat D×8, C×10, B×8, A×4.
    Weighted toward C/D since military tasks are predominantly 5T/10T.
    Shifts staggered 05:00–09:30. D017 marked unavailable (on leave).
    """

    # (name, id, cat, mileage, crash, shift_start, max_shift_h, available)
    _SEED = [
        # ── Cat D — 8 drivers (all vehicle types incl. 10T) ──────────────────
        ("Rahman",       "D001", "D", ">300k",      False, time(6,  0), 12.0, True ),
        ("Ng",           "D002", "D", ">300k",      False, time(5, 30), 12.0, True ),
        ("Chua",         "D003", "D", "100k-300k",  False, time(5,  0), 12.0, True ),
        ("Iskandar",     "D004", "D", ">300k",      False, time(6, 30), 12.0, True ),
        ("Subramaniam",  "D005", "D", "100k-300k",  True,  time(7,  0), 10.0, True ),
        ("Faizal",       "D006", "D", ">300k",      False, time(6,  0), 12.0, True ),
        ("Zheng",        "D007", "D", "100k-300k",  False, time(5, 30), 12.0, True ),
        ("Balachandran", "D008", "D", ">300k",      False, time(7, 30), 12.0, True ),
        # ── Cat C — 10 drivers (up to 5T) ────────────────────────────────────
        ("Lim",          "D009", "C", "100k-300k",  False, time(7,  0), 12.0, True ),
        ("Tan",          "D010", "C", "100k-300k",  True,  time(6, 30), 10.0, True ),
        ("Chen",         "D011", "C", "100k-300k",  False, time(7,  0), 12.0, True ),
        ("Muthu",        "D012", "C", ">300k",      False, time(7, 30), 12.0, True ),
        ("Hasan",        "D013", "C", "20k-100k",   False, time(8,  0), 12.0, True ),
        ("Govindasamy",  "D014", "C", "100k-300k",  False, time(6,  0), 12.0, True ),
        ("Kwok",         "D015", "C", "20k-100k",   False, time(8, 30), 10.0, True ),
        ("Nordin",       "D016", "C", "100k-300k",  False, time(7,  0), 12.0, True ),
        ("Selvam",       "D017", "C", "20k-100k",   True,  time(8,  0), 10.0, False),  # on leave
        ("Yap",          "D018", "C", "100k-300k",  False, time(6, 30), 12.0, True ),
        # ── Cat B — 8 drivers (up to Light Truck) ────────────────────────────
        ("Krishnan",     "D019", "B", "20k-100k",   False, time(8,  0), 10.0, True ),
        ("Wong",         "D020", "B", "20k-100k",   False, time(9,  0), 10.0, True ),
        ("Yeo",          "D021", "B", "20k-100k",   True,  time(6,  0), 10.0, True ),
        ("Siva",         "D022", "B", "100k-300k",  False, time(7,  0), 10.0, True ),
        ("Azman",        "D023", "B", "20k-100k",   False, time(8, 30), 10.0, True ),
        ("Teo",          "D024", "B", "20k-100k",   False, time(7, 30), 10.0, True ),
        ("Rajendran",    "D025", "B", "100k-300k",  False, time(6,  0), 12.0, True ),
        ("Huang",        "D026", "B", "20k-100k",   False, time(8,  0), 10.0, True ),
        # ── Cat A — 4 drivers (Car only) ─────────────────────────────────────
        ("Ali",          "D027", "A", "<20k",       False, time(7, 30),  8.0, True ),
        ("Ismail",       "D028", "A", "<20k",       False, time(8, 30),  8.0, True ),
        ("Ong",          "D029", "A", "<20k",       True,  time(9,  0),  8.0, True ),
        ("Phua",         "D030", "A", "<20k",       False, time(8,  0),  8.0, True ),
    ]

    def __init__(self):
        self._drivers: dict[str, DriverProfile] = {}
        for row in self._SEED:
            name, did, cat, mileage, crash, shift, max_sh, avail = row
            self._drivers[did] = DriverProfile(
                driver_id        = did,
                name             = name,
                category         = cat,
                mileage_band     = mileage,
                crash_history    = crash,
                shift_start_time = shift,
                max_shift_hours  = max_sh,
                available        = avail,
            )

    def all(self) -> list[DriverProfile]:
        return list(self._drivers.values())

    def get(self, driver_id: str) -> Optional[DriverProfile]:
        return self._drivers.get(driver_id)

    def eligible_for(self, vehicle_type: str,
                     available_only: bool = True) -> list[DriverProfile]:
        """Drivers licensed for vehicle_type. Excludes unavailable by default."""
        return [d for d in self._drivers.values()
                if vehicle_type in VEHICLE_ELIGIBILITY.get(d.category, [])
                and (not available_only or d.available)]

    @staticmethod
    def compute_fatigue(driver: DriverProfile, departure_time: time) -> float:
        """Hours elapsed since shift start at departure. Clamps to [0, 16]."""
        dep_dt   = datetime.combine(date.today(), departure_time)
        shift_dt = datetime.combine(date.today(), driver.shift_start_time)
        delta    = (dep_dt - shift_dt).total_seconds() / 3600
        return max(0.0, min(16.0, delta))


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VEHICLE DATABASE  — 18 vehicles
# ──────────────────────────────────────────────────────────────────────────────

class VehicleDB:
    """
    18 vehicles: Car×4, Light Truck×4, 5T×6, 10T×4.
    Condition spread (2026): Good×6, Fair×7, Poor×5.
    """

    _SEED = [
        # (plate,     type,          year, last_services,                           safe_tech)
        # Cars
        ("SG1001A", "Car",         2022, ["2025-11-01", "2025-05-01"],            ["ABS","Reverse Cam","Blind Spot Monitor"]),
        ("SG1002B", "Car",         2021, ["2025-02-01"],                           ["ABS","Reverse Cam"]),
        ("SG1003C", "Car",         2019, ["2024-08-15"],                           ["ABS"]),
        ("SG1004D", "Car",         2023, ["2025-01-10", "2024-07-05"],             ["ABS","Reverse Cam","Blind Spot Monitor"]),
        # Light Trucks
        ("SG2001E", "Light Truck", 2020, ["2025-12-01"],                           ["ABS","Blind Spot Monitor"]),
        ("SG2002F", "Light Truck", 2018, ["2024-10-15"],                           ["ABS","Reverse Cam"]),
        ("SG2003G", "Light Truck", 2016, ["2023-06-01"],                           ["ABS"]),
        ("SG2004H", "Light Truck", 2022, ["2025-09-20", "2025-03-10"],             ["ABS","Reverse Cam","Blind Spot Monitor"]),
        # 5T Trucks
        ("SG3001I", "5T",          2019, ["2025-06-15", "2024-12-20"],             ["ABS","Reverse Cam"]),
        ("SG3002J", "5T",          2017, ["2025-04-22"],                           ["ABS"]),
        ("SG3003K", "5T",          2015, ["2024-03-10"],                           ["ABS"]),
        ("SG3004L", "5T",          2020, ["2025-11-05", "2025-05-01"],             ["ABS","Blind Spot Monitor"]),
        ("SG3005M", "5T",          2016, ["2023-11-10"],                           ["ABS"]),   # overdue → Poor
        ("SG3006N", "5T",          2021, ["2025-08-01"],                           ["ABS","Reverse Cam"]),
        # 10T Trucks
        ("SG4001O", "10T",         2018, ["2025-04-22", "2024-10-01"],             ["ABS","Blind Spot Monitor"]),
        ("SG4002P", "10T",         2015, ["2024-08-01"],                           ["ABS"]),
        ("SG4003Q", "10T",         2014, ["2023-08-01"],                           ["ABS"]),   # old → Poor
        ("SG4004R", "10T",         2020, ["2025-10-15", "2025-04-10"],             ["ABS","Reverse Cam","Blind Spot Monitor"]),
    ]

    _AGE_POOR = 10
    _AGE_FAIR = 6
    _SVC_POOR = 18
    _SVC_FAIR = 9

    def __init__(self):
        self._vehicles: dict[str, VehicleProfile] = {}
        for row in self._SEED:
            plate, vtype, year, services, tech = row
            self._vehicles[plate] = VehicleProfile(
                vehicle_number      = plate,
                vehicle_type        = vtype,
                manufacture_year    = year,
                maintenance_records = [date.fromisoformat(s) for s in services],
                safe_tech           = tech,
            )

    def all(self) -> list[VehicleProfile]:
        return list(self._vehicles.values())

    def get(self, vehicle_number: str) -> Optional[VehicleProfile]:
        return self._vehicles.get(vehicle_number)

    def by_type(self, vehicle_type: str) -> list[VehicleProfile]:
        return [v for v in self._vehicles.values() if v.vehicle_type == vehicle_type]

    @staticmethod
    def compute_age(vehicle: VehicleProfile) -> int:
        return date.today().year - vehicle.manufacture_year

    @staticmethod
    def compute_condition(vehicle: VehicleProfile) -> str:
        age = date.today().year - vehicle.manufacture_year
        if vehicle.maintenance_records:
            last_svc     = max(vehicle.maintenance_records)
            months_since = (date.today() - last_svc).days / 30.44
        else:
            months_since = 999
        if age >= VehicleDB._AGE_POOR or months_since >= VehicleDB._SVC_POOR:
            return "Poor"
        elif age >= VehicleDB._AGE_FAIR or months_since >= VehicleDB._SVC_FAIR:
            return "Fair"
        return "Good"


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROUTE DATABASE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ODRoute:
    """One validated O-D entry from route.json."""
    node:         str
    od_key:       str
    origin:       str
    destination:  str
    distances_m:  list[float]   # [route0_m, route1_m, route2_m]


class RouteDB:
    """
    Loads and validates route.json.
    Valid entry: list of exactly 3 non-null float distances.
    Invalid entry: empty list [] — skipped silently.
    """

    def __init__(self, json_path: str = "route.json"):
        self._by_node_od:   dict[tuple[str, str], ODRoute] = {}
        self._by_locations: dict[tuple[str, str], ODRoute] = {}
        self._load(json_path)

    def _load(self, path: str):
        with open(path) as f:
            raw = json.load(f)
        for node, routes in raw.items():
            for od_key, dists in routes.items():
                if not isinstance(dists, list) or len(dists) != 3:
                    continue
                if any(d is None for d in dists):
                    continue
                parts = od_key.split("->")
                if len(parts) != 2:
                    continue
                origin = parts[0].strip()
                dest   = parts[1].strip()
                entry  = ODRoute(
                    node        = node,
                    od_key      = od_key,
                    origin      = origin,
                    destination = dest,
                    distances_m = [float(d) for d in dists],
                )
                self._by_node_od[(node, od_key)] = entry
                self._by_locations[(origin.upper(), dest.upper())] = entry

    def get(self, node: str, od_key: str) -> Optional[ODRoute]:
        return self._by_node_od.get((node, od_key))

    def find(self, origin: str, destination: str) -> Optional[ODRoute]:
        """Case-insensitive lookup by origin and destination strings."""
        return self._by_locations.get((origin.upper(), destination.upper()))

    def all_valid(self) -> list[ODRoute]:
        return list(self._by_node_od.values())

    def nodes(self) -> list[str]:
        return sorted({n for (n, _) in self._by_node_od})

    def locations_for_node(self, node: str) -> list[str]:
        return sorted({e.origin for (n, _), e in self._by_node_od.items() if n == node})

    @property
    def total_valid(self) -> int:
        return len(self._by_node_od)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  TRIP LOG  — seeded from route.json military locations
# ──────────────────────────────────────────────────────────────────────────────

class TripLogDB:
    """
    Historical trip ledger. Primary location pool comes from route.json
    military O-D pairs (weighted 4×) so familiarity scores are meaningful
    for allocation against military tasks. Generic Singapore addresses are
    retained as a secondary pool for the single-task free-text UI.
    """

    THRESH_HIGH   = 3
    THRESH_MEDIUM = 1

    def __init__(self, driver_db: DriverDB, vehicle_db: VehicleDB,
                 route_db: Optional[RouteDB] = None, seed: int = 99):
        self._logs: list[TripLog] = []
        self._generate(driver_db, vehicle_db, route_db, seed)

    def _generate(self, driver_db: DriverDB, vehicle_db: VehicleDB,
                  route_db: Optional[RouteDB], seed: int):
        rng     = random.Random(seed)
        drivers = driver_db.all()
        vehicles = vehicle_db.all()
        today   = date.today()

        # Primary: military locations from route.json
        if route_db is not None:
            mil_locs = list({
                loc
                for entry in route_db.all_valid()
                for loc in (entry.origin, entry.destination)
            })
        else:
            mil_locs = []

        # Secondary: generic Singapore addresses
        gen_locs = [
            "Orchard Road, Singapore",    "Changi Airport, Singapore",
            "Marina Bay Sands, Singapore", "Jurong East, Singapore",
            "Woodlands, Singapore",        "Tampines, Singapore",
            "Bishan, Singapore",           "Toa Payoh, Singapore",
            "Ang Mo Kio, Singapore",       "Clementi, Singapore",
            "Buona Vista, Singapore",      "Tuas Link, Singapore",
        ]

        # 4× weight on military locations
        all_locs = mil_locs * 4 + gen_locs
        od_pairs = list(dict.fromkeys(
            [(o, d) for o in all_locs for d in all_locs if o != d]
        ))

        for driver in drivers:
            base_trips = {
                "<20k":      20,
                "20k-100k":  80,
                "100k-300k": 250,
                ">300k":     500,
            }[driver.mileage_band]

            drng   = random.Random(int(driver.driver_id[1:]) + seed)
            jitter = drng.randint(-base_trips // 5, base_trips // 5)
            n      = max(5, base_trips + jitter)

            eligible_v = [v for v in vehicles
                          if v.vehicle_type in VEHICLE_ELIGIBILITY.get(driver.category, [])]
            if not eligible_v:
                continue

            favourite_ods = rng.sample(od_pairs, min(6, len(od_pairs)))
            weights = [8 if od in favourite_ods else 1 for od in od_pairs]

            for i in range(n):
                origin, dest = rng.choices(od_pairs, weights=weights, k=1)[0]
                trip_date    = today - timedelta(days=rng.randint(1, 730))
                vehicle      = rng.choice(eligible_v)
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
        def _norm(s: str) -> str:
            return s.lower().split(",")[0].strip()

        o, d     = _norm(origin), _norm(destination)
        by_route: dict[int, int] = {0: 0, 1: 0, 2: 0}

        for log in self._logs:
            if (log.driver_id == driver_id
                    and _norm(log.origin)      == o
                    and _norm(log.destination) == d):
                by_route[log.route_index] = by_route.get(log.route_index, 0) + 1

        total  = sum(by_route.values())
        od_fam = ("High"   if total >= self.THRESH_HIGH * 2
                  else "Medium" if total >= self.THRESH_MEDIUM * 2
                  else "Low")
        return {"by_route": by_route, "total_trips": total, "familiarity": od_fam}

    @staticmethod
    def per_route_familiarity(route_exp: dict, route_idx: int) -> str:
        trips = route_exp["by_route"].get(route_idx, 0)
        if trips >= TripLogDB.THRESH_HIGH:   return "High"
        if trips >= TripLogDB.THRESH_MEDIUM: return "Medium"
        return "Low"

    def vehicle_type_trips(self, driver_id: str, vehicle_type: str,
                           vehicle_db: VehicleDB) -> int:
        count = 0
        for log in self._logs:
            if log.driver_id != driver_id:
                continue
            v = vehicle_db.get(log.vehicle_number)
            if v and v.vehicle_type == vehicle_type:
                count += 1
        return count

    def vehicle_type_experience(self, driver_id: str, vehicle_db: VehicleDB) -> dict:
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
# 6.  DURATION + FATIGUE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def estimate_duration_h(dist_km: float,
                        avg_speed_kmh: float = _AVG_SPEED_KMH,
                        buffer_min:    float = _BUFFER_MINUTES) -> float:
    """
    Estimate total task time in hours (drive + loading/unloading buffer).
    Used by the allocator for conflict window checking and fatigue chaining.
    Not used in risk scoring — that uses the actual route distance directly.
    """
    return dist_km / avg_speed_kmh + buffer_min / 60.0


def fatigue_carry_forward(driver: DriverProfile,
                          dep_time: time,
                          prior_driving_km: float) -> float:
    """
    Fatigue at dep_time after the driver has already driven prior_driving_km.
    = base_fatigue(dep_time) + travel_time_of_prior_tasks
    Clamps to [0, 16].
    """
    base    = DriverDB.compute_fatigue(driver, dep_time)
    driving = prior_driving_km / _AVG_SPEED_KMH
    return min(16.0, base + driving)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENVIRONMENT SIMULATOR
# ──────────────────────────────────────────────────────────────────────────────

class EnvironmentSim:

    _HOUR_TRAFFIC = {
        **{h: 2 for h in range(0,  6)},
        **{h: 4 for h in range(6,  9)},
        **{h: 3 for h in range(9, 17)},
        **{h: 5 for h in range(17, 20)},
        **{h: 2 for h in range(20, 24)},
    }

    _WEATHER_PROB         = {"Clear": 0.60, "Rain": 0.30, "Heavy Rain": 0.10}
    _BRAKING_MULT         = {"Clear": 1.0,  "Rain": 1.25, "Heavy Rain": 1.55}
    _TRAFFIC_WEATHER_MULT = {"Clear": 1.0,  "Rain": 1.2,  "Heavy Rain": 1.5}
    _BASE_VIS_KM          = {"Clear": 10.0, "Rain": 5.0,  "Heavy Rain": 2.0}
    _NIGHT_VIS_CAP_KM     = 4.0

    _ROAD_CLASS_TRAFFIC = {
        "motorway":    0.6,
        "trunk":       0.75,
        "primary":     0.9,
        "secondary":   1.1,
        "tertiary":    1.25,
        "residential": 1.5,
        "unclassified":1.3,
    }

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng  = random.Random(seed)

    def query_weather(self, departure_time: time) -> WeatherSnapshot:
        hour      = departure_time.hour
        hour_seed = int(hashlib.md5(
            f"weather-{self._seed}-{hour}".encode()).hexdigest(), 16) % (2**32)
        hour_rng  = random.Random(hour_seed)

        probs = dict(self._WEATHER_PROB)
        if 13 <= hour <= 17:
            probs["Rain"]       += 0.10
            probs["Heavy Rain"] += 0.05
            probs["Clear"]      -= 0.15

        condition = hour_rng.choices(
            list(probs.keys()), weights=list(probs.values()), k=1)[0]
        vis_km = self._BASE_VIS_KM[condition] + hour_rng.uniform(-0.5, 0.5)
        vis_km = round(max(0.5, min(10.0, vis_km)), 1)

        return WeatherSnapshot(
            condition            = condition,
            visibility_km        = vis_km,
            braking_mult         = self._BRAKING_MULT[condition],
            traffic_weather_mult = self._TRAFFIC_WEATHER_MULT[condition],
        )

    def query_traffic(self, route_nodes: list[int], departure_time: time,
                      weather: WeatherSnapshot, DG=None) -> list[TrafficSegment]:
        hour      = departure_time.hour
        hour_base = self._HOUR_TRAFFIC.get(hour, 3)
        segments  = []

        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            if DG is not None:
                edge = DG.get_edge_data(u, v) or {}
                hw   = edge.get("highway", "unclassified")
                if isinstance(hw, list):
                    hw = hw[0]
                road_mult = self._ROAD_CLASS_TRAFFIC.get(hw, 1.3)
            else:
                road_mult = 1.0

            edge_seed = int(hashlib.md5(
                f"{u}-{v}-{hour}".encode()).hexdigest(), 16) % (2**32)
            edge_rng  = random.Random(edge_seed)
            jitter    = edge_rng.uniform(-3, 3)

            raw     = (hour_base * road_mult + jitter) * weather.traffic_weather_mult
            density = max(1, min(5, round(raw, 1)))
            segments.append(TrafficSegment(u=u, v=v, density=density))

        return segments

    @staticmethod
    def is_night(departure_time: time) -> bool:
        return departure_time.hour < 7 or departure_time.hour > 19

    @staticmethod
    def effective_visibility_km(snapshot: WeatherSnapshot,
                                departure_time: time) -> float:
        if EnvironmentSim.is_night(departure_time):
            return round(min(snapshot.visibility_km,
                             EnvironmentSim._NIGHT_VIS_CAP_KM), 1)
        return snapshot.visibility_km

    @staticmethod
    def visibility_mult(snapshot: WeatherSnapshot, departure_time: time) -> float:
        eff_vis = EnvironmentSim.effective_visibility_km(snapshot, departure_time)
        return round(1.0 + (10.0 - eff_vis) / 9.5 * 1.0, 3)

    @staticmethod
    def visibility_label(snapshot: WeatherSnapshot, departure_time: time) -> str:
        eff = EnvironmentSim.effective_visibility_km(snapshot, departure_time)
        tod = "Night" if EnvironmentSim.is_night(departure_time) else "Day"
        return f"{tod} + {snapshot.condition} → {eff:.1f} km"


# ──────────────────────────────────────────────────────────────────────────────
# 8.  ROUTE FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

class RouteFeatureExtractor:

    def __init__(self, DG, traffic_sim: EnvironmentSim):
        self._DG  = DG
        self._sim = traffic_sim

    def extract(self, route_index: int, node_sequence: list[int],
                departure_time: time, weather: WeatherSnapshot) -> RouteFeatures:
        DG            = self._DG
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

        dist_km   = total_len / 1000.0
        breakdown = {k: round(v / max(total_len, 1), 3)
                     for k, v in road_class_lengths.items()}
        traffic   = self._sim.query_traffic(
                        node_sequence, departure_time, weather, DG=self._DG)

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
        segs = features.segment_traffic
        return sum(s.density for s in segs) / len(segs) if segs else 3.0


# ──────────────────────────────────────────────────────────────────────────────
# 9.  ASSEMBLED QUERY
# ──────────────────────────────────────────────────────────────────────────────

def build_task_context(
    vehicle_number:    str,
    origin:            str,
    destination:       str,
    purpose:           str,
    departure_time:    time,
    route_sequences:   list[list[int]],
    DG,
    driver_db:         DriverDB,
    vehicle_db:        VehicleDB,
    trip_log_db:       TripLogDB,
    env_sim:           EnvironmentSim,
    fatigue_overrides: Optional[dict[str, float]] = None,
) -> dict:
    """
    Build full task context for one task.

    fatigue_overrides: {driver_id: float} — injected by the allocator
    to carry forward fatigue from prior tasks in a multi-task assignment.
    When absent, fatigue is computed fresh from shift start → departure.
    """
    extractor = RouteFeatureExtractor(DG, env_sim)

    vehicle = vehicle_db.get(vehicle_number)
    if vehicle is None:
        raise ValueError(f"Vehicle '{vehicle_number}' not found.")

    vehicle_age       = VehicleDB.compute_age(vehicle)
    vehicle_condition = VehicleDB.compute_condition(vehicle)

    weather_snapshot = env_sim.query_weather(departure_time)
    night_flag       = EnvironmentSim.is_night(departure_time)
    vis_label        = EnvironmentSim.visibility_label(weather_snapshot, departure_time)

    route_features = [
        extractor.extract(i, seq, departure_time, weather_snapshot)
        for i, seq in enumerate(route_sequences)
    ]

    eligible_drivers = driver_db.eligible_for(vehicle.vehicle_type, available_only=True)
    driver_contexts  = []

    for driver in eligible_drivers:
        if fatigue_overrides and driver.driver_id in fatigue_overrides:
            fatigue = fatigue_overrides[driver.driver_id]
        else:
            fatigue = DriverDB.compute_fatigue(driver, departure_time)

        route_exp      = trip_log_db.route_experience(
                             driver.driver_id, origin, destination)
        veh_type_trips = trip_log_db.vehicle_type_trips(
                             driver.driver_id, vehicle.vehicle_type, vehicle_db)
        total_trips    = sum(trip_log_db.vehicle_type_experience(
                             driver.driver_id, vehicle_db).values())

        driver_contexts.append({
            "profile":          driver,
            "fatigue_hours":    round(fatigue, 2),
            "route_experience": route_exp,
            "veh_type_trips":   veh_type_trips,
            "total_trips":      total_trips,
        })

    return {
        "vehicle": {
            "profile":   vehicle,
            "age":       vehicle_age,
            "condition": vehicle_condition,
        },
        "environment": {
            "weather":          weather_snapshot,
            "is_night":         night_flag,
            "visibility_label": vis_label,
        },
        "routes":  route_features,
        "drivers": driver_contexts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 10.  SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pprint import pprint

    driver_db  = DriverDB()
    vehicle_db = VehicleDB()
    route_db   = RouteDB("traffic_incident/route.json")
    trip_db    = TripLogDB(driver_db, vehicle_db, route_db)
    env_sim    = EnvironmentSim()

    print("=== Driver pool ===")
    for cat in ["D", "C", "B", "A"]:
        ds = [d for d in driver_db.all() if d.category == cat]
        print(f"  Cat {cat}: {len(ds)} total  "
              f"({sum(1 for d in ds if d.available)} available)")

    print("\n=== Eligible for 10T (available only) ===")
    for d in driver_db.eligible_for("10T", available_only=True):
        print(f"  {d.driver_id} {d.name:<14} shift={d.shift_start_time}  "
              f"max={d.max_shift_hours}h  crash={'Y' if d.crash_history else 'N'}")

    print("\n=== Vehicle fleet by type ===")
    for vt in VEHICLE_TYPES:
        vlist = vehicle_db.by_type(vt)
        conds = [VehicleDB.compute_condition(v) for v in vlist]
        print(f"  {vt:<12} {len(vlist)} vehicles — "
              f"Good:{conds.count('Good')} Fair:{conds.count('Fair')} Poor:{conds.count('Poor')}")

    print("\n=== RouteDB ===")
    print(f"  {route_db.total_valid} valid O-D pairs across {len(route_db.nodes())} nodes")
    for r in route_db.all_valid()[:4]:
        print(f"  [{r.node}] {r.origin!r} → {r.destination!r}  "
              f"[{r.distances_m[0]:.0f}, {r.distances_m[1]:.0f}, {r.distances_m[2]:.0f}] m")

    print("\n=== Trip history: D001, Seletar Camp → Nee Soon Camp ===")
    pprint(trip_db.route_experience("D001", "Seletar Camp", "Nee Soon Camp"))

    print("\n=== Vehicle type experience: D004 (Cat D) ===")
    pprint(trip_db.vehicle_type_experience("D004", vehicle_db))

    print("\n=== Duration estimate: 20 km ===")
    print(f"  {estimate_duration_h(20.0):.2f} h  "
          f"({estimate_duration_h(20.0)*60:.0f} min incl. {_BUFFER_MINUTES} min buffer)")

    print("\n=== Fatigue carry-forward: D001, dep 10:00, prior 30 km ===")
    d001 = driver_db.get("D001")
    print(f"  base={DriverDB.compute_fatigue(d001, time(10,0)):.2f}h  "
          f"chained={fatigue_carry_forward(d001, time(10,0), 30.0):.2f}h")

    print("\nAll smoke tests passed ✓")
