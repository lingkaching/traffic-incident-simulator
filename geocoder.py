"""
geocoder.py
===========
OneMap geocoding wrapper for Singapore addresses.

Replaces OSM/Nominatim (ox.geocoder.geocode) throughout the app.

Design
------
- Calls the OneMap public search endpoint (no token required).
- Always uses results[0] — the highest-ranked OneMap result.
  For route.json O-D locations (named military camps), each name
  corresponds to a unique location; multiple postcodes are just
  different blocks of the same camp and results[0] is the correct
  representative coordinate.
- Results are lru_cached in-process so repeated calls for the same
  location (e.g. the same origin across multiple batch tasks) are free.

Search endpoint (public, no auth):
  GET https://www.onemap.gov.sg/api/common/elastic/search
  ?searchVal=<query>&returnGeom=Y&getAddrDetails=Y&pageNum=1
"""

import requests
import streamlit as st
from functools import lru_cache

_SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"
_TIMEOUT    = 10


# ── Core geocode ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def geocode(query: str) -> tuple[float, float]:
    """
    Geocode a Singapore address using OneMap.
    Returns (latitude, longitude). Raises ValueError if nothing found.
    Results are cached in-process via lru_cache.
    """
    clean = query.strip()
    if not clean:
        raise ValueError("Empty geocode query.")

    search_val = clean
    if not _is_postal_code(clean) and "singapore" not in clean.lower():
        search_val = clean + " Singapore"

    try:
        resp = requests.get(
            _SEARCH_URL,
            params={
                "searchVal":      search_val,
                "returnGeom":     "Y",
                "getAddrDetails": "Y",
                "pageNum":        "1",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        raise ValueError(f"OneMap timed out for '{query}'.")
    except Exception as e:
        raise ValueError(f"OneMap geocode failed for '{query}': {e}")

    results = data.get("results", [])
    if not results:
        raise ValueError(
            f"No OneMap results for '{query}'. "
            "Try a 6-digit postal code or the full building name.")

    first = results[0]
    return float(first["LATITUDE"]), float(first["LONGITUDE"])


# ── Candidate list (used by single-task disambiguation widget) ────────────────

def geocode_with_candidates(query: str, max_results: int = 5) -> list[dict]:
    """Return up to max_results candidates with label, lat, lon."""
    clean = query.strip()
    if not clean:
        return []

    search_val = clean
    if not _is_postal_code(clean) and "singapore" not in clean.lower():
        search_val = clean + " Singapore"

    try:
        resp = requests.get(
            _SEARCH_URL,
            params={
                "searchVal":      search_val,
                "returnGeom":     "Y",
                "getAddrDetails": "Y",
                "pageNum":        "1",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    candidates = []
    for r in data.get("results", [])[:max_results]:
        try:
            building = r.get("BUILDING", "") or ""
            address  = r.get("ADDRESS",  "") or ""
            postal   = r.get("POSTAL",   "") or r.get("POSTALCODE", "") or ""
            label    = building if (building and building != "NIL") else address
            if postal and postal != "NIL":
                label += f" ({postal})"
            candidates.append({
                "label": label.strip(),
                "lat":   float(r["LATITUDE"]),
                "lon":   float(r["LONGITUDE"]),
            })
        except (KeyError, ValueError):
            continue

    return candidates


# ── Streamlit widget (single-task free-text input) ────────────────────────────

def geocode_input(label: str, default: str = "", key: str = ""):
    """
    Text input + live candidate selectbox for the single-task UI.
    Returns (display_label, lat, lon) or None if unresolved.
    """
    query = st.text_input(label, value=default, key=f"{key}_query")
    if not query.strip():
        return None

    candidates = geocode_with_candidates(query)
    if not candidates:
        st.caption("⚠️ No results — try a 6-digit postal code or full building name.")
        return None

    if len(candidates) == 1:
        c = candidates[0]
        st.caption(f"✅ {c['label']}")
        return c["label"], c["lat"], c["lon"]

    # Multiple candidates — disambiguation dropdown
    options = [c["label"] for c in candidates]
    chosen  = st.selectbox(
        f"Select result for '{query}'", options,
        key=f"{key}_pick", label_visibility="collapsed",
    )
    c = candidates[options.index(chosen)]
    return c["label"], c["lat"], c["lon"]


# ── Helper ────────────────────────────────────────────────────────────────────

def _is_postal_code(s: str) -> bool:
    """Singapore postal codes are exactly 6 digits."""
    return s.strip().isdigit() and len(s.strip()) == 6


# ── Stub for backward compatibility (warm_geocode_cache called in app_v2.py) ──

def get_cache():
    """No-op stub — LocationCache removed. Kept so app import doesn't break."""
    class _Stub:
        n_ambiguous = 0
        def populate_all(self, *a, **kw): pass
    return _Stub()