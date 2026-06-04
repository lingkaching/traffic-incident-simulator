"""
allocator.py
============
Fleet Resource Allocation Engine

Given a batch of tasks (each with origin, destination, departure time,
purpose, vehicle) and a pool of drivers, finds the assignment of one
driver + route per task that minimises overall fleet risk subject to
driver availability and conflict constraints.

Design
------
The allocator is completely independent of Streamlit — no UI imports.
It calls the same risk engine functions used by the single-task simulator
(compute_P, compute_T, compute_A, compute_route_risk), which it imports
from risk_engine.py so the two modes share identical scoring logic.

Algorithm: Greedy with 1-step lookahead → Local swap
  1. Score matrix      — compute risk for every valid (task, driver, route)
                         triplet up front
  2. Greedy w/lookahead— sort tasks by fewest valid options first; for each
                         task, pick the option that minimises
                         max(this_lambda, best_remaining_lambda_across_all_tasks)
                         rather than just min(this_lambda) in isolation
  3. Local swap        — for every pair of tasks try swapping their drivers
                         or routes; accept if the objective improves
  4. Return        — final list of Assignment objects + summary stats

Objective: minimise max(prob_i) across all assignments.
Total lambda is used internally for the scoring because it is linear
in the driver/route choice and avoids probability saturation near 1.

Conflict rules
--------------
Hard (block assignment):
  - Driver not licensed for vehicle type            (eligibility)
  - Driver unavailable (on leave)                   (available flag)
  - Time overlap with an already-committed task      (schedule)
  - Shift would exceed driver's max_shift_hours      (fatigue cap)

Soft (penalised via chained fatigue, not blocked):
  - Back-to-back tasks with < BUFFER_MIN gap         (fatigue carries forward)
  - Cumulative driving > 6 h                         (flagged in summary)
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Optional, Callable

from data_layer_v2 import (
    DriverDB, VehicleDB, TripLogDB, EnvironmentSim,
    RouteFeatureExtractor, RouteFeatures,
    DriverProfile, VehicleProfile,
    estimate_duration_h, fatigue_carry_forward,
    VEHICLE_ELIGIBILITY,
)

# ──────────────────────────────────────────────────────────────────────────────
# Risk engine is imported at call time via dependency injection so allocator.py
# does not need to import the Streamlit app.  The caller (app_v1.py or a test
# script) passes compute_P, compute_T, compute_A, compute_route_risk as args
# to Allocator().  Default implementations are provided below for standalone use.
# ──────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT RISK ENGINE  (mirrors app_v1.py exactly — kept in sync manually)
# ══════════════════════════════════════════════════════════════════════════════

BASE_LAMBDA  = 1e-6

MILEAGE_MULT = {"<20k": 1.4, "20k-100k": 1.1, "100k-300k": 0.95, ">300k": 0.9}
PURPOSE_MULT = {"Admin": 1.0, "Training": 1.05, "Operation": 1.2, "Emergency": 1.6}
VTYPE_MULT   = {"Car": 1.0, "Light Truck": 1.2, "5T": 1.5, "10T": 2.0}
ROAD_MULT    = {"motorway": 0.7, "trunk": 0.9, "primary": 1.2,
                "secondary": 1.5, "tertiary": 1.8, "residential": 2.2}

CRASH_MULT             = 1.5
ROUTE_EXP_BONUS        = 0.80
ROUTE_EXP_MED_MULT     = 1.00
ROUTE_EXP_LOW_MULT     = 1.35
VEH_TYPE_EXP_HIGH_THRESH = 100
VEH_TYPE_EXP_MED_THRESH  = 50
VEH_TYPE_EXP_HIGH_MULT   = 0.85
VEH_TYPE_EXP_MED_MULT    = 1.00
VEH_TYPE_EXP_LOW_MULT    = 1.30
MAINT_POOR_MULT        = 1.5
TECH_PENALTY           = 0.05
NARROW_ROAD_MULT       = 1.3
SHARP_TURN_MULT        = 1.4


def _default_compute_P(driver_ctx: dict) -> float:
    profile = driver_ctx["profile"]
    fatigue = driver_ctx["fatigue_hours"]
    if fatigue <= 4:
        fm = 1 + fatigue * 0.05
    elif fatigue <= 10:
        fm = 1 + 4 * 0.05 + (fatigue - 4) * 0.10
    else:
        fm = 1 + 4 * 0.05 + 6 * 0.10 + (fatigue - 10) * 0.18
    return (MILEAGE_MULT[profile.mileage_band]
            * (CRASH_MULT if profile.crash_history else 1.0)
            * fm)


def _default_compute_T(driver_ctx: dict, purpose: str, route_idx: int) -> float:
    route_exp      = driver_ctx["route_experience"]
    veh_type_trips = driver_ctx["veh_type_trips"]

    trips = route_exp["by_route"].get(route_idx, 0)
    if trips >= TripLogDB.THRESH_HIGH:   fam_mult = ROUTE_EXP_BONUS
    elif trips >= TripLogDB.THRESH_MEDIUM: fam_mult = ROUTE_EXP_MED_MULT
    else:                                  fam_mult = ROUTE_EXP_LOW_MULT

    if veh_type_trips >= VEH_TYPE_EXP_HIGH_THRESH:   ve_mult = VEH_TYPE_EXP_HIGH_MULT
    elif veh_type_trips >= VEH_TYPE_EXP_MED_THRESH:  ve_mult = VEH_TYPE_EXP_MED_MULT
    else:                                              ve_mult = VEH_TYPE_EXP_LOW_MULT

    return PURPOSE_MULT[purpose] * fam_mult * ve_mult


def _default_compute_A(vehicle_ctx: dict, weather) -> float:
    profile   = vehicle_ctx["profile"]
    condition = vehicle_ctx["condition"]
    tech_disc = len(profile.safe_tech) * TECH_PENALTY
    cond_mult = (MAINT_POOR_MULT if condition == "Poor" else
                 1.1             if condition == "Fair" else 1.0)
    return (VTYPE_MULT.get(profile.vehicle_type, 1.0)
            * cond_mult * (1 - tech_disc) * weather.braking_mult)


def _roadway_mult_for_edge(edge: dict) -> float:
    hw = edge.get("highway", "residential")
    if isinstance(hw, list):
        hw = hw[0] if hw else "residential"
    mult = edge.get("hotspot_mult", 1.0)
    mult *= ROAD_MULT.get(hw, 1.5)
    if edge.get("is_narrow"): mult *= NARROW_ROAD_MULT
    if edge.get("is_sharp"):  mult *= SHARP_TURN_MULT
    return min(mult, 4.0)


def _default_compute_route_risk(route_features: RouteFeatures,
                                P: float, T: float, A: float,
                                env_ctx: dict, dep_time: time, DG) -> dict:
    weather_snap = env_ctx["weather"]
    vis_m        = EnvironmentSim.visibility_mult(weather_snap, dep_time)
    traffic_lookup = {(s.u, s.v): s.density for s in route_features.segment_traffic}

    nodes = route_features.node_sequence
    total_lambda = total_len = weighted_E = weighted_R = 0.0
    segment_data = []
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
        segment_data.append((u, v, lam))

    denom = max(total_len, 1e-9)
    E = weighted_E / denom
    R = weighted_R / denom
    tl_safe = max(total_lambda, 1e-30)
    segments = [(u, v, lam_i / tl_safe) for u, v, lam_i in segment_data]

    return {
        "prob":          1 - math.exp(-total_lambda),
        "total_lambda":  total_lambda,
        "dist_km":       route_features.dist_km,
        "segments":      segments,
        "hotspot_coords":hotspot_coords,
        "P": P, "T": T, "A": A, "E": E, "R": R,
        "vis_m":         vis_m,
        "avg_traffic_m": weighted_E / (vis_m * denom) if vis_m > 0 else 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN OBJECTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    """
    One unit of work to be allocated.
    route_sequences: up to 3 node-id lists from OSMnx pathfinding.
    route_features:  populated by the allocator after build_task_context.
    """
    task_id:         str
    origin:          str
    destination:     str
    departure_time:  time
    purpose:         str
    vehicle_number:  str
    route_sequences: list[list[int]]       # up to 3 OSMnx paths
    route_features:  list[RouteFeatures] = field(default_factory=list)


@dataclass
class TimeBlock:
    """A committed time window for one driver."""
    start:   time
    end:     time
    task_id: str


@dataclass
class DriverSchedule:
    """
    Tracks a driver's committed time blocks as the solver assigns tasks.
    Accumulated driving km is tracked separately for fatigue carry-forward.
    """
    driver_id:       str
    shift_start:     time
    max_shift_hours: float
    blocks:          list[TimeBlock] = field(default_factory=list)
    driving_km:      float           = 0.0   # sum of distances of committed tasks

    # ── Time arithmetic helpers ───────────────────────────────────────────────

    @staticmethod
    def _to_dt(t: time) -> datetime:
        return datetime.combine(date.today(), t)

    @staticmethod
    def _add_h(t: time, hours: float) -> time:
        dt = DriverSchedule._to_dt(t) + timedelta(hours=hours)
        return dt.time()

    @staticmethod
    def _hours_between(a: time, b: time) -> float:
        return (DriverSchedule._to_dt(b) - DriverSchedule._to_dt(a)).total_seconds() / 3600

    @staticmethod
    def _overlaps(s1: time, e1: time, s2: time, e2: time) -> bool:
        """True if [s1,e1) and [s2,e2) overlap (treating time as same-day)."""
        dt_s1, dt_e1 = DriverSchedule._to_dt(s1), DriverSchedule._to_dt(e1)
        dt_s2, dt_e2 = DriverSchedule._to_dt(s2), DriverSchedule._to_dt(e2)
        return dt_s1 < dt_e2 and dt_s2 < dt_e1

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self, dep_time: time, duration_h: float,
                     buffer_h: float = 0.5) -> tuple[bool, str]:
        """
        Check whether the driver can take a task departing at dep_time
        for duration_h hours.

        Returns (ok: bool, reason: str).  reason is empty string when ok.

        buffer_h: minimum gap required after each committed task (default 30 min).
        """
        end_time = self._add_h(dep_time, duration_h)

        # 1. Must be within shift window
        if self._to_dt(dep_time) < self._to_dt(self.shift_start):
            return False, f"departs before shift start ({self.shift_start})"

        shift_end_h = self._hours_between(self.shift_start, end_time)
        if shift_end_h > self.max_shift_hours:
            return False, (f"would exceed max shift "
                           f"({shift_end_h:.1f}h > {self.max_shift_hours}h)")

        # 2. No overlap with existing committed tasks (+ buffer around each)
        for blk in self.blocks:
            # Expand the existing block by buffer on both sides
            blk_start_buf = self._add_h(blk.start, -buffer_h)
            blk_end_buf   = self._add_h(blk.end,    buffer_h)
            if self._overlaps(dep_time, end_time, blk_start_buf, blk_end_buf):
                return False, (f"overlaps with task {blk.task_id} "
                               f"({blk.start}–{blk.end})")

        return True, ""

    def commit(self, dep_time: time, duration_h: float,
               task_id: str, dist_km: float):
        """Record a confirmed assignment."""
        end_time = self._add_h(dep_time, duration_h)
        self.blocks.append(TimeBlock(start=dep_time, end=end_time, task_id=task_id))
        self.driving_km += dist_km
        # Keep blocks sorted by start time for clarity
        self.blocks.sort(key=lambda b: self._to_dt(b.start))

    def rollback(self, task_id: str):
        """Remove a previously committed task (used during local swap)."""
        self.blocks = [b for b in self.blocks if b.task_id != task_id]
        # driving_km is approximate after rollback — recalculate if needed

    @property
    def total_driving_h(self) -> float:
        return sum(
            self._hours_between(b.start, b.end) for b in self.blocks
        )


@dataclass
class Assignment:
    """One confirmed (task, driver, route) assignment with its risk score."""
    task:           Task
    driver_ctx:     dict          # full driver context dict (profile + experience)
    vehicle_ctx:    dict
    env_ctx:        dict
    route_features: RouteFeatures
    risk:           dict          # output of compute_route_risk
    fatigue_at_dep: float         # chained fatigue value used for this assignment


@dataclass
class AllocationResult:
    """Full output of one allocation run."""
    assignments:    list[Assignment]
    unassigned:     list[Task]
    objective:      float
    total_lambda:   float
    warnings:       list[str]
    score_matrix:   dict = field(default_factory=dict)  # full (task,driver,route)→risk

    # ── Summary helpers ───────────────────────────────────────────────────────

    @property
    def n_tasks(self) -> int:
        return len(self.assignments) + len(self.unassigned)

    @property
    def max_risk_assignment(self) -> Optional[Assignment]:
        return max(self.assignments, key=lambda a: a.risk["prob"]) if self.assignments else None

    def driver_workload(self) -> dict[str, float]:
        """Total driving km per driver across all assignments."""
        wl: dict[str, float] = {}
        for a in self.assignments:
            did = a.driver_ctx["profile"].driver_id
            wl[did] = wl.get(did, 0.0) + a.route_features.dist_km
        return wl

    def to_summary_rows(self) -> list[dict]:
        """Flat list of dicts suitable for a Streamlit dataframe."""
        rows = []
        for i, a in enumerate(self.assignments):
            prof    = a.driver_ctx["profile"]
            risk    = a.risk
            rows.append({
                "rank":        i + 1,
                "task_id":     a.task.task_id,
                "origin":      a.task.origin,
                "destination": a.task.destination,
                "dep_time":    a.task.departure_time.strftime("%H:%M"),
                "purpose":     a.task.purpose,
                "vehicle":     a.task.vehicle_number,
                "driver_id":   prof.driver_id,
                "driver_name": prof.name,
                "route_idx":   a.route_features.route_index + 1,   # 1-indexed for display
                "dist_km":     round(a.route_features.dist_km, 1),
                "fatigue_h":   round(a.fatigue_at_dep, 2),
                "prob_pct":    round(risk["prob"] * 100, 4),
                "risk_level":  _risk_category(risk["prob"]),
                "P":           round(risk["P"], 3),
                "T":           round(risk["T"], 3),
                "A":           round(risk["A"], 3),
                "E":           round(risk["E"], 3),
                "R":           round(risk["R"], 3),
            })
        return rows


def _risk_category(prob: float) -> str:
    if prob < 0.05:  return "Low"
    if prob < 0.10:  return "Medium"
    return "High"


# ══════════════════════════════════════════════════════════════════════════════
# SCORE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_score_matrix(
    tasks:      list[Task],
    task_ctxs:  list[dict],         # one context dict per task (from build_task_context)
    DG,
    compute_P:           Callable = _default_compute_P,
    compute_T:           Callable = _default_compute_T,
    compute_A:           Callable = _default_compute_A,
    compute_route_risk:  Callable = _default_compute_route_risk,
) -> dict[tuple[str, str, int], dict]:
    """
    Pre-compute risk for every valid (task_id, driver_id, route_index) triplet.

    Returns a dict keyed on (task_id, driver_id, route_index) → risk dict.
    Triplets where the driver is ineligible for the vehicle type are omitted.

    This is the most expensive step: O(N_tasks × N_drivers × 3 routes).
    With the RouteCache in place each evaluation takes ~200 ms; for 10 tasks
    × 15 eligible drivers × 3 routes = 450 evaluations ≈ 90 s.
    """
    matrix: dict[tuple[str, str, int], dict] = {}

    for task, task_ctx in zip(tasks, task_ctxs):
        vehicle_ctx = task_ctx["vehicle"]
        env_ctx     = task_ctx["environment"]
        dep_time    = task.departure_time
        weather     = env_ctx["weather"]
        A           = compute_A(vehicle_ctx, weather)

        for driver_ctx in task_ctx["drivers"]:
            driver_id = driver_ctx["profile"].driver_id
            P = compute_P(driver_ctx)
            for rf in task_ctx["routes"]:
                T    = compute_T(driver_ctx, task.purpose, rf.route_index)
                risk = compute_route_risk(rf, P, T, A, env_ctx, dep_time, DG)
                matrix[(task.task_id, driver_id, rf.route_index)] = risk

    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# GREEDY ASSIGNMENT  (with one-step lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def greedy_assign(
    tasks:        list[Task],
    task_ctxs:    list[dict],
    score_matrix: dict[tuple[str, str, int], dict],
    schedules:    dict[str, DriverSchedule],
) -> tuple[list[Optional[tuple[str, int]]], dict[str, DriverSchedule]]:
    """
    Greedy assignment with one-step lookahead.

    Key insight: picking the option with the smallest total_lambda for the
    current task is WRONG when the objective is min max(prob_i) fleet-wide.
    Grabbing the best driver for task i may leave only a bad driver for
    task j, making max(lambda) worse overall.

    Fix: for each candidate option for the current task, tentatively commit
    it, then compute the best available total_lambda for each remaining
    unassigned task given the updated schedules.  Pick the option that
    minimises max(candidate_lambda, best_remaining_lambda_across_all_tasks).

    This is O(options × remaining_tasks × drivers × routes) per task —
    acceptable for typical batch sizes (≤20 tasks, ≤30 drivers, 3 routes).
    """

    def _best_available_lambda(task_idx: int,
                                sched_snapshot: dict[str, DriverSchedule]) -> float:
        """
        Lowest total_lambda reachable for task_idx given the current schedules.
        Returns inf if no option is available (task would be unassigned).
        """
        task = tasks[task_idx]
        ctx  = task_ctxs[task_idx]
        best = float("inf")
        for driver_ctx in ctx["drivers"]:
            did = driver_ctx["profile"].driver_id
            sched = sched_snapshot.get(did)
            if sched is None:
                continue
            for rf in ctx["routes"]:
                key  = (task.task_id, did, rf.route_index)
                risk = score_matrix.get(key)
                if risk is None:
                    continue
                ok, _ = sched.is_available(task.departure_time,
                                            estimate_duration_h(rf.dist_km))
                if ok:
                    best = min(best, risk["total_lambda"])
        return best

    # Sort tasks: fewest valid (driver, route) options first
    # (most-constrained-first — ensures the tightest tasks aren't starved)
    def _n_options(task_idx: int) -> int:
        task = tasks[task_idx]
        ctx  = task_ctxs[task_idx]
        return sum(1 for d in ctx["drivers"] for rf in ctx["routes"]
                   if (task.task_id, d["profile"].driver_id, rf.route_index)
                   in score_matrix)

    order        = sorted(range(len(tasks)), key=_n_options)
    assignments: list[Optional[tuple[str, int]]] = [None] * len(tasks)
    remaining    = set(order)   # task indices not yet assigned

    for task_idx in order:
        remaining.discard(task_idx)
        task = tasks[task_idx]
        ctx  = task_ctxs[task_idx]

        # Collect all available (driver, route) options for this task
        options: list[tuple[str, int, RouteFeatures, float]] = []
        for driver_ctx in ctx["drivers"]:
            did = driver_ctx["profile"].driver_id
            for rf in ctx["routes"]:
                key  = (task.task_id, did, rf.route_index)
                risk = score_matrix.get(key)
                if risk is None:
                    continue
                ok, _ = schedules[did].is_available(task.departure_time,
                                                     estimate_duration_h(rf.dist_km))
                if ok:
                    options.append((did, rf.route_index, rf, risk["total_lambda"]))

        if not options:
            continue   # task cannot be assigned

        # One-step lookahead: for each candidate, tentatively commit it and
        # compute the implied max(total_lambda) across this task + all remaining.
        best_score   = float("inf")
        best_option  = None

        for did, ridx, rf, this_lambda in options:
            # Tentatively commit this option
            dur = estimate_duration_h(rf.dist_km)
            schedules[did].commit(task.departure_time, dur, task.task_id, rf.dist_km)

            # Worst case across remaining unassigned tasks (best each can achieve)
            worst_remaining = max(
                (_best_available_lambda(r_idx, schedules) for r_idx in remaining),
                default=0.0,
            )

            # The fleet objective if we take this option
            lookahead_score = max(this_lambda, worst_remaining)

            # Roll back tentative commit
            schedules[did].rollback(task.task_id)

            if lookahead_score < best_score:
                best_score  = lookahead_score
                best_option = (did, ridx, rf)

        best_did, best_ridx, best_rf = best_option
        assignments[task_idx] = (best_did, best_ridx)
        schedules[best_did].commit(
            dep_time   = task.departure_time,
            duration_h = estimate_duration_h(best_rf.dist_km),
            task_id    = task.task_id,
            dist_km    = best_rf.dist_km,
        )

    return assignments, schedules


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL SWAP IMPROVEMENT
# ══════════════════════════════════════════════════════════════════════════════

def local_swap(
    tasks:        list[Task],
    task_ctxs:    list[dict],
    assignments:  list[Optional[tuple[str, int]]],
    score_matrix: dict[tuple[str, str, int], dict],
    schedules:    dict[str, DriverSchedule],
    max_iters:    int = 50,
) -> tuple[list[Optional[tuple[str, int]]], dict[str, DriverSchedule]]:
    """
    Improve the greedy solution by trying pairwise swaps.

    For each assigned task pair (i, j):
      - Try assigning task_i's (driver, route) to task_j and vice versa
      - Try giving task_i the driver from task_j but keeping task_i's best route
      - Try changing only the route for a single task

    Accept any swap that reduces max(total_lambda) across all assignments.
    Repeat until no improvement or max_iters reached.
    """
    def _objective(asgns: list[Optional[tuple[str, int]]]) -> float:
        """Max total_lambda across all assigned tasks."""
        vals = []
        for i, asgn in enumerate(asgns):
            if asgn is None:
                continue
            did, ridx = asgn
            key = (tasks[i].task_id, did, ridx)
            risk = score_matrix.get(key)
            if risk:
                vals.append(risk["total_lambda"])
        return max(vals) if vals else 0.0

    improved = True
    iters    = 0

    while improved and iters < max_iters:
        improved = False
        iters   += 1
        current_obj = _objective(assignments)

        assigned_indices = [i for i, a in enumerate(assignments) if a is not None]

        for i in assigned_indices:
            task_i = tasks[i]
            did_i, ridx_i = assignments[i]

            # ── Try every route for task_i with its current driver ────────────
            for rf in task_ctxs[i]["routes"]:
                new_ridx = rf.route_index
                if new_ridx == ridx_i:
                    continue
                key = (task_i.task_id, did_i, new_ridx)
                if key not in score_matrix:
                    continue
                # Temporarily swap route
                old_asgn = assignments[i]
                assignments[i] = (did_i, new_ridx)
                if _objective(assignments) < current_obj:
                    current_obj = _objective(assignments)
                    improved = True
                else:
                    assignments[i] = old_asgn

            # ── Try swapping drivers between task_i and task_j ────────────────
            for j in assigned_indices:
                if j <= i:
                    continue
                task_j = tasks[j]
                did_j, ridx_j = assignments[j]

                if did_i == did_j:
                    continue  # same driver — can't swap

                # Check both cross-assignments exist in score matrix
                key_ij = (task_i.task_id, did_j, ridx_i)
                key_ji = (task_j.task_id, did_i, ridx_j)
                if key_ij not in score_matrix or key_ji not in score_matrix:
                    continue

                # Check schedule compatibility after swap
                # i gets did_j, j gets did_i
                sched_i = schedules[did_i]
                sched_j = schedules[did_j]
                dur_i = estimate_duration_h(task_ctxs[i]["routes"][ridx_i].dist_km)
                dur_j = estimate_duration_h(task_ctxs[j]["routes"][ridx_j].dist_km)

                # Roll back both, check, restore
                sched_i.rollback(task_i.task_id)
                sched_j.rollback(task_j.task_id)

                ok_ij, _ = sched_j.is_available(task_i.departure_time, dur_i)
                ok_ji, _ = sched_i.is_available(task_j.departure_time, dur_j)

                if ok_ij and ok_ji:
                    old_i, old_j = assignments[i], assignments[j]
                    assignments[i] = (did_j, ridx_i)
                    assignments[j] = (did_i, ridx_j)

                    if _objective(assignments) < current_obj:
                        # Accept swap — commit new schedule
                        rf_i = next(r for r in task_ctxs[i]["routes"] if r.route_index == ridx_i)
                        rf_j = next(r for r in task_ctxs[j]["routes"] if r.route_index == ridx_j)
                        sched_j.commit(task_i.departure_time, dur_i, task_i.task_id, rf_i.dist_km)
                        sched_i.commit(task_j.departure_time, dur_j, task_j.task_id, rf_j.dist_km)
                        current_obj = _objective(assignments)
                        improved = True
                    else:
                        # Reject swap — restore original assignments and schedules
                        assignments[i] = old_i
                        assignments[j] = old_j
                        # Re-commit original
                        rf_i = next(r for r in task_ctxs[i]["routes"] if r.route_index == ridx_i)
                        rf_j = next(r for r in task_ctxs[j]["routes"] if r.route_index == ridx_j)
                        sched_i.commit(task_i.departure_time, dur_i, task_i.task_id, rf_i.dist_km)
                        sched_j.commit(task_j.departure_time, dur_j, task_j.task_id, rf_j.dist_km)
                else:
                    # Swap not feasible — restore schedules
                    rf_i = next(r for r in task_ctxs[i]["routes"] if r.route_index == ridx_i)
                    rf_j = next(r for r in task_ctxs[j]["routes"] if r.route_index == ridx_j)
                    sched_i.commit(task_i.departure_time, dur_i, task_i.task_id, rf_i.dist_km)
                    sched_j.commit(task_j.departure_time, dur_j, task_j.task_id, rf_j.dist_km)

    return assignments, schedules


# ══════════════════════════════════════════════════════════════════════════════
# FATIGUE CARRY-FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def _build_fatigue_overrides(
    assignments:  list[Optional[tuple[str, int]]],
    tasks:        list[Task],
    task_ctxs:    list[dict],
    driver_db:    DriverDB,
) -> dict[str, dict[str, float]]:
    """
    For each driver assigned to multiple tasks, compute chained fatigue:
      fatigue_at_task_k = base_fatigue + Σ driving_time_of_prior_tasks

    Returns dict: task_id → {driver_id: chained_fatigue_hours}
    Only entries that differ from base fatigue are included.
    """
    # Group assignments by driver, sorted by departure time
    driver_tasks: dict[str, list[tuple[time, int]]] = {}  # did → [(dep_time, task_idx)]
    for task_idx, asgn in enumerate(assignments):
        if asgn is None:
            continue
        did, _ = asgn
        driver_tasks.setdefault(did, []).append(
            (tasks[task_idx].departure_time, task_idx))

    overrides: dict[str, dict[str, float]] = {}  # task_id → {driver_id: fatigue}

    for did, task_list in driver_tasks.items():
        if len(task_list) < 2:
            continue  # single task — no carry-forward needed
        task_list.sort(key=lambda x: datetime.combine(date.today(), x[0]))
        driver = driver_db.get(did)
        cumulative_km = 0.0

        for dep_time, task_idx in task_list:
            if cumulative_km > 0:
                fat = fatigue_carry_forward(driver, dep_time, cumulative_km)
                overrides.setdefault(tasks[task_idx].task_id, {})[did] = fat
            # Add this task's distance to cumulative for subsequent tasks
            ridx       = assignments[task_idx][1]
            rf         = next(r for r in task_ctxs[task_idx]["routes"]
                              if r.route_index == ridx)
            cumulative_km += rf.dist_km

    return overrides


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ALLOCATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Allocator:
    """
    Main entry point for the allocation engine.

    Usage
    -----
    allocator = Allocator(driver_db, vehicle_db, trip_db, env_sim, DG)

    # Build tasks from route JSON + OSMnx paths
    tasks = [Task(...), Task(...), ...]

    result = allocator.solve(tasks)
    print(result.to_summary_rows())
    """

    def __init__(
        self,
        driver_db:  DriverDB,
        vehicle_db: VehicleDB,
        trip_db:    TripLogDB,
        env_sim:    EnvironmentSim,
        DG,
        compute_P_fn:          Callable = _default_compute_P,
        compute_T_fn:          Callable = _default_compute_T,
        compute_A_fn:          Callable = _default_compute_A,
        compute_route_risk_fn: Callable = _default_compute_route_risk,
    ):
        self.driver_db   = driver_db
        self.vehicle_db  = vehicle_db
        self.trip_db     = trip_db
        self.env_sim     = env_sim
        self.DG          = DG
        self._compute_P  = compute_P_fn
        self._compute_T  = compute_T_fn
        self._compute_A  = compute_A_fn
        self._compute_R  = compute_route_risk_fn

    def solve(
        self,
        tasks:      list[Task],
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> AllocationResult:
        """
        Run the full allocation pipeline.

        progress_cb: optional callback(message: str, fraction: float)
        for UI progress bars.
        """
        if not tasks:
            return AllocationResult([], [], 0.0, 0.0, [])

        def _progress(msg: str, frac: float):
            if progress_cb:
                progress_cb(msg, frac)

        # ── 1. Build task contexts ────────────────────────────────────────────
        _progress("Building task contexts…", 0.05)
        from data_layer_v2 import build_task_context, RouteFeatureExtractor
        task_ctxs = []
        for task in tasks:
            ctx = build_task_context(
                vehicle_number  = task.vehicle_number,
                origin          = task.origin,
                destination     = task.destination,
                purpose         = task.purpose,
                departure_time  = task.departure_time,
                route_sequences = task.route_sequences,
                DG              = self.DG,
                driver_db       = self.driver_db,
                vehicle_db      = self.vehicle_db,
                trip_log_db     = self.trip_db,
                env_sim         = self.env_sim,
            )
            ctx["purpose"]  = task.purpose
            ctx["dep_time"] = task.departure_time
            # Store route features back on the task object for later
            task.route_features = ctx["routes"]
            task_ctxs.append(ctx)

        # ── 2. Build score matrix ─────────────────────────────────────────────
        _progress("Scoring all driver × route combinations…", 0.10)
        score_matrix = build_score_matrix(
            tasks, task_ctxs, self.DG,
            self._compute_P, self._compute_T, self._compute_A, self._compute_R,
        )
        _progress("Score matrix complete.", 0.50)

        # ── 3. Initialise per-driver schedules ────────────────────────────────
        schedules: dict[str, DriverSchedule] = {}
        all_drivers = self.driver_db.eligible_for(
            # Collect all vehicle types required by the batch
            # then build schedules for every eligible driver
            "Car", available_only=True)  # placeholder — see below
        # Build schedules for every available driver across all vehicle types
        all_driver_ids = {
            d_ctx["profile"].driver_id
            for ctx in task_ctxs
            for d_ctx in ctx["drivers"]
        }
        for did in all_driver_ids:
            driver = self.driver_db.get(did)
            if driver:
                schedules[did] = DriverSchedule(
                    driver_id       = did,
                    shift_start     = driver.shift_start_time,
                    max_shift_hours = driver.max_shift_hours,
                )

        # ── 4. Greedy assignment ──────────────────────────────────────────────
        _progress("Running greedy assignment…", 0.55)
        raw_assignments, schedules = greedy_assign(
            tasks, task_ctxs, score_matrix, schedules)

        # ── 5. Local swap improvement ─────────────────────────────────────────
        _progress("Optimising with local swaps…", 0.70)
        raw_assignments, schedules = local_swap(
            tasks, task_ctxs, raw_assignments, score_matrix, schedules)

        # ── 6. Fatigue carry-forward recomputation ────────────────────────────
        _progress("Recomputing chained fatigue…", 0.85)
        fatigue_overrides_by_task = _build_fatigue_overrides(
            raw_assignments, tasks, task_ctxs, self.driver_db)

        # For tasks with chained drivers, rebuild their score and replace
        for task_idx, asgn in enumerate(raw_assignments):
            if asgn is None:
                continue
            did, ridx = asgn
            task_id = tasks[task_idx].task_id
            if task_id in fatigue_overrides_by_task:
                override = fatigue_overrides_by_task[task_id]
                if did in override:
                    # Recompute risk with updated fatigue
                    ctx   = task_ctxs[task_idx]
                    d_ctx = next(d for d in ctx["drivers"]
                                 if d["profile"].driver_id == did)
                    d_ctx_updated = dict(d_ctx)
                    d_ctx_updated["fatigue_hours"] = override[did]
                    rf = next(r for r in ctx["routes"] if r.route_index == ridx)
                    A  = self._compute_A(ctx["vehicle"], ctx["environment"]["weather"])
                    P  = self._compute_P(d_ctx_updated)
                    T  = self._compute_T(d_ctx_updated, tasks[task_idx].purpose, ridx)
                    new_risk = self._compute_R(
                        rf, P, T, A, ctx["environment"],
                        tasks[task_idx].departure_time, self.DG)
                    score_matrix[(task_id, did, ridx)] = new_risk

        # ── 7. Assemble final Assignment objects ──────────────────────────────
        _progress("Assembling results…", 0.95)
        assignments:  list[Assignment] = []
        unassigned:   list[Task]       = []

        for task_idx, asgn in enumerate(raw_assignments):
            task = tasks[task_idx]
            if asgn is None:
                unassigned.append(task)
                continue

            did, ridx = asgn
            ctx      = task_ctxs[task_idx]
            d_ctx    = next(d for d in ctx["drivers"]
                            if d["profile"].driver_id == did)
            rf       = next(r for r in ctx["routes"] if r.route_index == ridx)
            risk     = score_matrix[(task.task_id, did, ridx)]

            fat_override = fatigue_overrides_by_task.get(task.task_id, {}).get(did)
            fatigue_at_dep = fat_override if fat_override is not None \
                             else d_ctx["fatigue_hours"]

            assignments.append(Assignment(
                task           = task,
                driver_ctx     = d_ctx,
                vehicle_ctx    = ctx["vehicle"],
                env_ctx        = ctx["environment"],
                route_features = rf,
                risk           = risk,
                fatigue_at_dep = fatigue_at_dep,
            ))

        # Sort by task departure time
        assignments.sort(key=lambda a: datetime.combine(
            date.today(), a.task.departure_time))

        # ── 8. Warnings ───────────────────────────────────────────────────────
        warnings: list[str] = []

        for did, sched in schedules.items():
            if sched.total_driving_h > 6.0:
                driver = self.driver_db.get(did)
                warnings.append(
                    f"{driver.name} ({did}): cumulative driving "
                    f"{sched.total_driving_h:.1f} h exceeds 6 h threshold.")

        for task in unassigned:
            warnings.append(
                f"Task {task.task_id} ({task.origin} → {task.destination}) "
                f"could not be assigned — no available driver without conflict.")

        objective     = max((a.risk["prob"] for a in assignments), default=0.0)
        total_lambda  = sum(a.risk["total_lambda"] for a in assignments)

        _progress("Done.", 1.0)

        return AllocationResult(
            assignments  = assignments,
            unassigned   = unassigned,
            objective    = objective,
            total_lambda = total_lambda,
            warnings     = warnings,
            score_matrix = score_matrix,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pprint import pprint
    from datetime import time

    # ── Minimal test without OSMnx (uses mock route features) ─────────────────
    from data_layer_v2 import (
        DriverDB, VehicleDB, TripLogDB, RouteDB, EnvironmentSim,
        RouteFeatures, TrafficSegment,
    )

    driver_db  = DriverDB()
    vehicle_db = VehicleDB()
    route_db   = RouteDB("traffic_incident/route.json")
    trip_db    = TripLogDB(driver_db, vehicle_db, route_db)
    env_sim    = EnvironmentSim()

    # Build a minimal mock DiGraph for testing
    import networkx as nx
    DG = nx.DiGraph()
    for n in range(100):
        DG.add_node(n, x=103.8 + n * 0.001, y=1.3 + n * 0.001)
    for n in range(99):
        DG.add_edge(n, n+1, length=200.0, highway="primary",
                    hotspot_mult=1.0, is_narrow=False, is_sharp=False)

    # Build 3 mock route sequences (just sequential node chains)
    mock_routes = [list(range(20)),    # route 0
                   list(range(10, 30)), # route 1
                   list(range(5, 25))]  # route 2

    tasks = [
        Task(
            task_id        = "T001",
            origin         = "Seletar Camp",
            destination    = "Nee Soon Camp",
            departure_time = time(8, 0),
            purpose        = "Operation",
            vehicle_number = "SG3001I",   # 5T
            route_sequences = mock_routes,
        ),
        Task(
            task_id        = "T002",
            origin         = "Nee Soon Camp",
            destination    = "Seletar Camp",
            departure_time = time(10, 0),
            purpose        = "Admin",
            vehicle_number = "SG4001O",   # 10T
            route_sequences = mock_routes,
        ),
        Task(
            task_id        = "T003",
            origin         = "Seletar Camp",
            destination    = "Pasir Laba Camp",
            departure_time = time(9, 0),
            purpose        = "Training",
            vehicle_number = "SG3002J",   # 5T
            route_sequences = mock_routes,
        ),
    ]

    print("=== Allocator smoke test ===")
    print(f"  Tasks: {len(tasks)}  Drivers available: {len(driver_db.all())}")

    def _progress(msg, frac):
        print(f"  [{frac*100:.0f}%] {msg}")

    allocator = Allocator(driver_db, vehicle_db, trip_db, env_sim, DG)
    result    = allocator.solve(tasks, progress_cb=_progress)

    print(f"\n=== Result ===")
    print(f"  Assigned:    {len(result.assignments)}/{len(tasks)}")
    print(f"  Unassigned:  {len(result.unassigned)}")
    print(f"  Objective (max prob): {result.objective*100:.4f}%")
    print(f"  Total lambda: {result.total_lambda:.6f}")

    print("\n=== Assignment summary ===")
    for row in result.to_summary_rows():
        print(f"  {row['task_id']}  {row['driver_name']:<14} "
              f"Route {row['route_idx']}  "
              f"{row['dist_km']} km  "
              f"prob={row['prob_pct']:.4f}%  "
              f"[{row['risk_level']}]  "
              f"fatigue={row['fatigue_h']}h")

    if result.warnings:
        print("\n=== Warnings ===")
        for w in result.warnings:
            print(f"  ⚠ {w}")

    if result.unassigned:
        print("\n=== Unassigned ===")
        for t in result.unassigned:
            print(f"  {t.task_id}: {t.origin} → {t.destination}")

    print("\nDriver workload:")
    for did, km in result.driver_workload().items():
        d = driver_db.get(did)
        print(f"  {d.name:<14} {km:.1f} km")

    print("\nSmoke test passed ✓")