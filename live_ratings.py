"""
live_ratings.py
---------------
Fetches recent international match results from the football-data.org
free API and derives dynamic Attack / Defense ratings for each team.

HOW IT WORKS
------------
1. Pull the last N completed matches from the UEFA Nations League and
   FIFA World Cup competitions (both available on the free tier).
2. For every team, calculate:
       attack_score  = goals_scored_per_game  (relative to league average)
       defense_score = goals_conceded_per_game (relative to league average)
3. Scale both values to the 40-95 range the predictor expects.

FREE API KEY
------------
Sign up at https://www.football-data.org/client/register
It's instant and free. Paste your key into the Streamlit sidebar or
set the environment variable FOOTBALL_DATA_API_KEY.

RATE LIMIT
----------
The free tier allows 10 requests/minute. Results are cached in
Streamlit session state for 30 minutes to avoid hitting the limit.
"""

import os
import time
import datetime
import requests
import pandas as pd
import streamlit as st

# football-data.org base URL
BASE_URL = "https://api.football-data.org/v4"

# Competition IDs available on the free tier
COMPETITIONS = {
    "UEFA Nations League": "UNL",
    "FIFA World Cup":      "WC",
    "UEFA Euro":           "EC",
    "Copa America":        "CLI",
}

# Rating scale boundaries
RATING_MIN = 40.0
RATING_MAX = 95.0
CACHE_MINUTES = 30        # how long to keep fetched data in session state
MIN_MATCHES = 3           # minimum matches needed to include a team


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def _headers(api_key: str) -> dict:
    return {"X-Auth-Token": api_key}


def _get(endpoint: str, api_key: str, params: dict = None) -> dict:
    """Make a single GET request; return parsed JSON or None on error."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=_headers(api_key), params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            st.warning("⏳ API rate limit hit — waiting 15 seconds and retrying…")
            time.sleep(15)
            resp = requests.get(url, headers=_headers(api_key), params=params, timeout=10)
            return resp.json() if resp.status_code == 200 else None
        else:
            st.error(f"API error {resp.status_code}: {resp.text[:200]}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None


# --------------------------------------------------------------------------
# Fetch match results
# --------------------------------------------------------------------------
def fetch_recent_matches(api_key: str, competition_code: str, seasons: int = 1) -> list:
    """
    Return a list of finished match dicts for the given competition.
    Each dict has: home_team, away_team, home_goals, away_goals, date.
    """
    data = _get(f"competitions/{competition_code}/matches", api_key,
                params={"status": "FINISHED"})
    if not data or "matches" not in data:
        return []

    matches = []
    for m in data["matches"]:
        score = m.get("score", {})
        ft = score.get("fullTime", {})
        home_g = ft.get("home")
        away_g = ft.get("away")
        if home_g is None or away_g is None:
            continue
        matches.append({
            "home_team":  m["homeTeam"]["name"],
            "away_team":  m["awayTeam"]["name"],
            "home_goals": int(home_g),
            "away_goals": int(away_g),
            "date":       m.get("utcDate", "")[:10],
        })
    return matches


# --------------------------------------------------------------------------
# Derive ratings from match results
# --------------------------------------------------------------------------
def derive_ratings_from_matches(matches: list[dict]) -> pd.DataFrame:
    """
    Given a list of match result dicts, compute per-team:
        - games_played
        - goals_scored_per_game
        - goals_conceded_per_game
        - attack  (scaled 40-95)
        - defense (scaled 40-95; higher = better defense = fewer goals conceded)

    Returns a DataFrame indexed by team name.
    """
    if not matches:
        return pd.DataFrame()

    records = {}

    def _add(team, scored, conceded):
        if team not in records:
            records[team] = {"played": 0, "scored": 0, "conceded": 0}
        records[team]["played"]   += 1
        records[team]["scored"]   += scored
        records[team]["conceded"] += conceded

    for m in matches:
        _add(m["home_team"], m["home_goals"], m["away_goals"])
        _add(m["away_team"], m["away_goals"], m["home_goals"])

    rows = []
    for team, stats in records.items():
        if stats["played"] < MIN_MATCHES:
            continue
        scored_pg   = stats["scored"]   / stats["played"]
        conceded_pg = stats["conceded"] / stats["played"]
        rows.append({
            "team":            team,
            "games_played":    stats["played"],
            "goals_scored_pg": round(scored_pg,   2),
            "goals_conceded_pg": round(conceded_pg, 2),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("team")

    # Scale attack: more goals scored → higher attack rating
    df["attack"] = _scale_series(df["goals_scored_pg"], RATING_MIN, RATING_MAX)

    # Scale defense: FEWER goals conceded → HIGHER defense rating (invert)
    df["defense"] = _scale_series(
        df["goals_conceded_pg"].max() - df["goals_conceded_pg"],
        RATING_MIN, RATING_MAX
    )

    df["attack"]  = df["attack"].round(2)
    df["defense"] = df["defense"].round(2)

    return df.sort_values("attack", ascending=False)


def _scale_series(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Min-max scale a pandas Series to [lo, hi]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([((lo + hi) / 2)] * len(series), index=series.index)
    return lo + (series - mn) / (mx - mn) * (hi - lo)


# --------------------------------------------------------------------------
# Main entry point: fetch + derive, with session-state caching
# --------------------------------------------------------------------------
def get_live_ratings(api_key: str, competition_code: str) -> tuple:
    """
    Returns (ratings_df, raw_matches, status_message).
    Results are cached in session state for CACHE_MINUTES minutes.
    """
    cache_key = f"live_cache_{competition_code}"
    now = datetime.datetime.utcnow()

    # Return cached result if still fresh
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        age_minutes = (now - cached["fetched_at"]).seconds / 60
        if age_minutes < CACHE_MINUTES:
            remaining = int(CACHE_MINUTES - age_minutes)
            status = (
                f"✅ Live data loaded — {cached['match_count']} matches from "
                f"**{cached['competition']}** · "
                f"Cache refreshes in ~{remaining} min"
            )
            return cached["ratings"], cached["matches"], status

    # Fetch fresh data
    matches = fetch_recent_matches(api_key, competition_code)
    if not matches:
        return pd.DataFrame(), [], "⚠️ No match data returned. Check your API key or competition."

    ratings = derive_ratings_from_matches(matches)

    comp_name = next((k for k, v in COMPETITIONS.items() if v == competition_code), competition_code)
    st.session_state[cache_key] = {
        "ratings":     ratings,
        "matches":     matches,
        "match_count": len(matches),
        "competition": comp_name,
        "fetched_at":  now,
    }

    status = f"✅ Fetched **{len(matches)} matches** from **{comp_name}** — ratings updated live!"
    return ratings, matches, status
