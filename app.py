"""
app.py
------
Soccer Match Predictor ⚽

A Streamlit app that predicts soccer match outcomes using a
Poisson-distribution Monte Carlo simulation, in the style of:

    "Based on 10,000 simulations... Netherlands a 43% chance of winning
    in regular time, Morocco a 27% chance, and a tie of 32%..."

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import poisson

from teams_data import DEFAULT_TEAMS
from live_ratings import get_live_ratings, COMPETITIONS

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Soccer Match Predictor ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants for the prediction model
# --------------------------------------------------------------------------
LEAGUE_AVG_GOALS = 1.35   # average goals scored by a national team per match
BASELINE_RATING = 75.0    # rating treated as "league average" attack/defense
N_SIMULATIONS = 10_000     # Monte Carlo simulation count
DRAW_WARNING_THRESHOLD = 0.32  # draw probability that triggers ET/penalties note

# --------------------------------------------------------------------------
# FIFA World Cup 2026 — built-in ratings (based on June 2026 FIFA rankings)
# --------------------------------------------------------------------------
FIFA_2026_RATINGS = {
    "Argentina":   {"attack": 95.0, "defense": 88.0},
    "Spain":       {"attack": 93.0, "defense": 84.0},
    "France":      {"attack": 91.0, "defense": 82.0},
    "England":     {"attack": 89.0, "defense": 83.0},
    "Portugal":    {"attack": 87.0, "defense": 80.0},
    "Brazil":      {"attack": 85.5, "defense": 77.0},
    "Morocco":     {"attack": 84.0, "defense": 81.0},
    "Netherlands": {"attack": 82.5, "defense": 76.0},
    "Belgium":     {"attack": 81.0, "defense": 74.5},
    "Germany":     {"attack": 79.5, "defense": 76.5},
    "Croatia":     {"attack": 78.0, "defense": 77.0},
    "Italy":       {"attack": 76.5, "defense": 79.5},
    "Colombia":    {"attack": 75.0, "defense": 71.0},
    "Mexico":      {"attack": 73.5, "defense": 70.0},
    "Senegal":     {"attack": 72.0, "defense": 72.5},
    "Uruguay":     {"attack": 70.5, "defense": 73.5},
    "USA":         {"attack": 69.0, "defense": 68.5},
    "Japan":       {"attack": 67.5, "defense": 70.5},
    "Switzerland": {"attack": 66.0, "defense": 69.0},
    "Iran":        {"attack": 64.5, "defense": 67.5},
    "Denmark":     {"attack": 63.0, "defense": 68.0},
    "Turkiye":     {"attack": 61.5, "defense": 64.0},
    "Ecuador":     {"attack": 60.0, "defense": 65.0},
    "Austria":     {"attack": 58.5, "defense": 63.0},
    "South Korea": {"attack": 57.0, "defense": 61.5},
    "Nigeria":     {"attack": 55.5, "defense": 60.0},
    "Australia":   {"attack": 54.0, "defense": 62.0},
    "Algeria":     {"attack": 52.5, "defense": 58.5},
    "Egypt":       {"attack": 51.0, "defense": 60.5},
    "Canada":      {"attack": 49.5, "defense": 55.0},
}


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def read_team_csv_robust(uploaded_file) -> pd.DataFrame:
    """
    Read a team-ratings CSV robustly, handling common real-world issues:
    - Excel saving with semicolons instead of commas (locale-dependent)
    - UTF-8 BOM characters from Excel/Windows
    - Extra whitespace around column names or values
    - Tab-delimited files
    """
    raw_bytes = uploaded_file.getvalue()

    # Decode handling potential BOM
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    # Try comma first, then semicolon, then tab
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            normalized = {c.lower() for c in df.columns}
            if {"team", "attack", "defense"}.issubset(normalized):
                # Strip whitespace from string cells
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.strip()
                return df
        except Exception:
            continue

    # Fall back to pandas' own delimiter sniffer
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def init_session_state():
    """Set up persistent state: team roster and last-used ratings."""
    if "teams" not in st.session_state:
        st.session_state.teams = dict(DEFAULT_TEAMS)

    if "last_used" not in st.session_state:
        st.session_state.last_used = {
            "home_team": "Morocco",
            "away_team": "Netherlands",
            "home_attack": DEFAULT_TEAMS["Morocco"]["attack"],
            "home_defense": DEFAULT_TEAMS["Morocco"]["defense"],
            "away_attack": DEFAULT_TEAMS["Netherlands"]["attack"],
            "away_defense": DEFAULT_TEAMS["Netherlands"]["defense"],
        }

    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None


# --------------------------------------------------------------------------
# Core prediction engine
# --------------------------------------------------------------------------
def compute_expected_goals(attack_for: float, defense_against: float) -> float:
    """
    Translate attack/defense ratings into an Expected Goals (xG) value
    using a multiplicative ratio against the league baseline.

    A team with above-baseline attack scores more; a team facing an
    above-baseline (i.e. stronger) opposing defense scores less.
    """
    attack_factor = attack_for / BASELINE_RATING
    defense_factor = BASELINE_RATING / defense_against
    xg = LEAGUE_AVG_GOALS * attack_factor * defense_factor
    return max(xg, 0.05)  # floor so Poisson never gets a non-positive mean


def run_monte_carlo_simulation(
    home_xg: float,
    away_xg: float,
    n_simulations: int = N_SIMULATIONS,
):
    """
    Run a Poisson-based Monte Carlo simulation of the match.

    For each of n_simulations trials, draw a random goal count for the
    home and away team from a Poisson distribution with mean = xG.
    Tally outcomes and the most frequent exact scorelines.
    """
    rng = np.random.default_rng()

    home_goals = rng.poisson(lam=home_xg, size=n_simulations)
    away_goals = rng.poisson(lam=away_xg, size=n_simulations)

    home_wins = np.sum(home_goals > away_goals)
    away_wins = np.sum(home_goals < away_goals)
    draws = np.sum(home_goals == away_goals)

    home_win_pct = home_wins / n_simulations
    away_win_pct = away_wins / n_simulations
    draw_pct = draws / n_simulations

    # Find most common scoreline
    scorelines = list(zip(home_goals.tolist(), away_goals.tolist()))
    scoreline_series = pd.Series(scorelines)
    most_common_score = scoreline_series.value_counts().idxmax()
    most_common_score_pct = scoreline_series.value_counts(normalize=True).max()

    # Build a probability table for a small grid of scorelines (0-5 goals)
    max_goals_grid = 6
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals_grid)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals_grid)]
    score_matrix = np.outer(home_probs, away_probs)

    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_win_pct": home_win_pct,
        "away_win_pct": away_win_pct,
        "draw_pct": draw_pct,
        "most_common_score": most_common_score,
        "most_common_score_pct": most_common_score_pct,
        "score_matrix": score_matrix,
        "max_goals_grid": max_goals_grid,
        "n_simulations": n_simulations,
    }


def build_prediction_narrative(home_team, away_team, home_xg, away_xg, results):
    """
    Construct the human-readable narrative paragraph in the requested
    'X roads worth of data' prediction style.
    """
    home_pct = results["home_win_pct"] * 100
    away_pct = results["away_win_pct"] * 100
    draw_pct = results["draw_pct"] * 100
    n_sims = results["n_simulations"]
    score = results["most_common_score"]

    leader = "tie" if draw_pct >= home_pct and draw_pct >= away_pct else (
        f"{home_team} win" if home_pct > away_pct else f"{away_team} win"
    )

    narrative = (
        f"**This is my prediction for {home_team} versus {away_team}.** "
        f"Based on {n_sims:,} roads worth of data... "
        f"{home_team} a {home_pct:.0f}% chance of winning in regular time, "
        f"{away_team} a {away_pct:.0f}% chance, and a tie of {draw_pct:.0f}% chance... "
        f"after running {n_sims:,} simulations of a Poisson model, "
        f"{home_team} has an expected goal of {home_xg:.2f}, "
        f"{away_team} has an expected goal of {away_xg:.2f}. "
        f"The most likely scoreline is **{score[0]} - {score[1]}**, "
        f"and the model is predicting a **{leader}** in regular time."
    )

    if draw_pct / 100 > DRAW_WARNING_THRESHOLD:
        narrative += (
            " Given the tightness of this matchup, there's a high chance "
            "this game goes to extra time and penalties."
        )

    return narrative


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------
def plot_expected_goals(home_team, away_team, home_xg, away_xg):
    """Plotly bar chart comparing Expected Goals for both teams."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[home_team, away_team],
                y=[home_xg, away_xg],
                marker_color=["#2E7D32", "#FF6F00"],
                text=[f"{home_xg:.2f}", f"{away_xg:.2f}"],
                textposition="outside",
                width=[0.5, 0.5],
            )
        ]
    )
    fig.update_layout(
        title="Expected Goals (xG) — Poisson Model",
        yaxis_title="Expected Goals",
        xaxis_title="Team",
        template="plotly_white",
        height=420,
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


def plot_outcome_probabilities(home_team, away_team, results):
    """Plotly bar chart of Home Win / Draw / Away Win probabilities."""
    labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
    values = [
        results["home_win_pct"] * 100,
        results["draw_pct"] * 100,
        results["away_win_pct"] * 100,
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=["#1565C0", "#9E9E9E", "#C62828"],
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Match Outcome Probabilities (Regular Time)",
        yaxis_title="Probability (%)",
        template="plotly_white",
        height=420,
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


def plot_goal_distribution(home_team, away_team, home_goals, away_goals):
    """Histogram comparing simulated goal distributions for both teams."""
    max_g = int(max(home_goals.max(), away_goals.max())) + 1
    bins = np.arange(0, max_g + 1) - 0.5

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=home_goals, xbins=dict(start=bins[0], end=bins[-1], size=1),
            name=home_team, marker_color="#2E7D32", opacity=0.75,
            histnorm="probability",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=away_goals, xbins=dict(start=bins[0], end=bins[-1], size=1),
            name=away_team, marker_color="#FF6F00", opacity=0.75,
            histnorm="probability",
        )
    )
    fig.update_layout(
        title="Simulated Goal Distribution (10,000 trials)",
        xaxis_title="Goals Scored",
        yaxis_title="Probability",
        barmode="overlay",
        template="plotly_white",
        height=420,
        margin=dict(t=60, b=40),
    )
    return fig


# --------------------------------------------------------------------------
# Sidebar: team management + CSV loading
# --------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("⚙️ Team Management")

    # ---- Add a new team ----
    with st.sidebar.expander("➕ Add a New Team", expanded=False):
        new_name = st.text_input("Team name", key="new_team_name")
        new_attack = st.slider("Attack rating", 40, 95, 70, key="new_team_attack")
        new_defense = st.slider("Defense rating", 40, 95, 70, key="new_team_defense")
        if st.button("Add Team", key="add_team_btn"):
            if new_name.strip() == "":
                st.sidebar.warning("Please enter a team name.")
            else:
                st.session_state.teams[new_name.strip()] = {
                    "attack": float(new_attack),
                    "defense": float(new_defense),
                }
                st.sidebar.success(f"Added {new_name.strip()} to the roster!")

    # ---- Load ratings from CSV ----
    with st.sidebar.expander("📂 Load Teams from CSV", expanded=False):
        st.caption("CSV must have columns: team, attack, defense")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
        if uploaded_file is not None:
            try:
                df = read_team_csv_robust(uploaded_file)
                required_cols = {"team", "attack", "defense"}
                normalized_cols = {c.strip().lower() for c in df.columns}
                if not required_cols.issubset(normalized_cols):
                    st.sidebar.error(
                        "CSV must contain: team, attack, defense columns. "
                        f"Found columns: {list(df.columns)}"
                    )
                else:
                    df.columns = [c.strip().lower() for c in df.columns]
                    loaded = 0
                    for _, row in df.iterrows():
                        team_name = str(row["team"]).strip()
                        if not team_name or team_name.lower() == "nan":
                            continue
                        st.session_state.teams[team_name] = {
                            "attack": float(row["attack"]),
                            "defense": float(row["defense"]),
                        }
                        loaded += 1
                    st.sidebar.success(f"Loaded {loaded} teams from CSV!")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")

    # ---- Current roster table ----
    with st.sidebar.expander("📋 Current Roster", expanded=False):
        roster_df = pd.DataFrame.from_dict(st.session_state.teams, orient="index")
        roster_df.index.name = "team"
        st.dataframe(roster_df, use_container_width=True)

    # ---- FIFA 2026 one-click loader ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🌍 FIFA World Cup 2026 Teams**")
    st.sidebar.caption("Load all 30 World Cup 2026 teams with real-ranking-based ratings instantly — no file needed.")

    if st.sidebar.button("⚡ Load FIFA 2026 Ratings Now", use_container_width=True):
        for team, ratings in FIFA_2026_RATINGS.items():
            st.session_state.teams[team] = ratings
        st.sidebar.success(f"Loaded {len(FIFA_2026_RATINGS)} World Cup 2026 teams!")
        st.rerun()

    csv_bytes = "team,attack,defense\n" + "\n".join(
        f"{t},{r['attack']},{r['defense']}" for t, r in FIFA_2026_RATINGS.items()
    )
    st.sidebar.download_button(
        label="💾 Download FIFA 2026 CSV",
        data=csv_bytes,
        file_name="fifa_june2026_ratings.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model settings**\n\n"
        f"- Simulations per run: `{N_SIMULATIONS:,}`\n"
        f"- League baseline rating: `{BASELINE_RATING}`\n"
        f"- League average goals/team: `{LEAGUE_AVG_GOALS}`"
    )


# --------------------------------------------------------------------------
# Main app body
# --------------------------------------------------------------------------
def render_team_selector(label, default_team, key_prefix):
    """
    Renders a team dropdown + attack/defense sliders for one side
    (home or away). Returns (team_name, attack, defense).
    """
    team_names = sorted(st.session_state.teams.keys())
    default_idx = team_names.index(default_team) if default_team in team_names else 0

    team = st.selectbox(
        f"{label} Team",
        options=team_names,
        index=default_idx,
        key=f"{key_prefix}_team_select",
    )

    base_ratings = st.session_state.teams[team]

    attack = st.slider(
        f"{label} Attack Strength",
        min_value=40,
        max_value=95,
        value=int(round(base_ratings["attack"])),
        key=f"{key_prefix}_attack_slider",
    )
    defense = st.slider(
        f"{label} Defense Strength",
        min_value=40,
        max_value=95,
        value=int(round(base_ratings["defense"])),
        key=f"{key_prefix}_defense_slider",
    )

    return team, float(attack), float(defense)


# --------------------------------------------------------------------------
# Live Ratings Dashboard tab
# --------------------------------------------------------------------------
def render_live_dashboard():
    """Full live data panel rendered inside the Live Ratings tab."""
    st.subheader("📡 Live Ratings Dashboard")
    st.markdown(
        "Enter your free API key from [football-data.org](https://www.football-data.org/client/register) "
        "to fetch real match results and auto-compute team ratings."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        api_key = st.text_input(
            "🔑 API Key",
            type="password",
            placeholder="Paste your football-data.org key here…",
            key="live_api_key",
        )
    with col2:
        competition = st.selectbox(
            "Competition",
            options=list(COMPETITIONS.keys()),
            key="live_competition",
        )

    fetch_clicked = st.button("🔄 Fetch Live Ratings", type="primary", key="fetch_live_btn")

    if fetch_clicked:
        if not api_key.strip():
            st.error("Please enter your API key first.")
            return
        comp_code = COMPETITIONS[competition]
        with st.spinner(f"Fetching {competition} results…"):
            ratings_df, matches, status = get_live_ratings(api_key.strip(), comp_code)
        st.info(status)

        if not ratings_df.empty:
            # Store in session
            st.session_state.live_ratings_df = ratings_df
            st.session_state.live_matches    = matches

            # Push into team roster automatically
            applied = 0
            for team, row in ratings_df.iterrows():
                st.session_state.teams[str(team)] = {
                    "attack":  float(row["attack"]),
                    "defense": float(row["defense"]),
                }
                applied += 1
            st.success(
                f"✅ **{applied} teams** updated in your roster with live ratings! "
                "Switch to the 🎯 Predict tab to use them."
            )

    # Display ratings table if available
    if "live_ratings_df" in st.session_state and not st.session_state.live_ratings_df.empty:
        df = st.session_state.live_ratings_df.copy()
        df.index.name = "Team"
        df = df.rename(columns={
            "games_played":      "Games",
            "goals_scored_pg":   "Goals Scored / Game",
            "goals_conceded_pg": "Goals Conceded / Game",
            "attack":            "Attack Rating",
            "defense":           "Defense Rating",
        })

        st.markdown("### 📊 Derived Team Ratings")
        st.dataframe(df, use_container_width=True)

        # Top 5 attack vs defense chart
        top_df = df.head(10).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_df["Team"], y=top_df["Attack Rating"],
            name="Attack", marker_color="#FF6F00",
        ))
        fig.add_trace(go.Bar(
            x=top_df["Team"], y=top_df["Defense Rating"],
            name="Defense", marker_color="#1565C0",
        ))
        fig.update_layout(
            title="Top 10 Teams — Attack vs Defense Rating (Live Data)",
            barmode="group", template="plotly_white",
            height=450, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Enter your API key and click **Fetch Live Ratings** to load real data.")
        st.markdown(
            "**Getting a free API key takes 30 seconds:**\n"
            "1. Go to [football-data.org/client/register](https://www.football-data.org/client/register)\n"
            "2. Enter your email — key arrives instantly\n"
            "3. Paste it above and click Fetch"
        )


def render_recent_results():
    """Show a table and chart of raw fetched match results."""
    st.subheader("📋 Recent Match Results")

    if "live_matches" not in st.session_state or not st.session_state.live_matches:
        st.info("Fetch live data in the 📡 Live Ratings Dashboard tab first.")
        return

    matches = st.session_state.live_matches
    df = pd.DataFrame(matches)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["result"] = df.apply(
        lambda r: f"{r['home_team']} {r['home_goals']}–{r['away_goals']} {r['away_team']}", axis=1
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=False)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches", len(df))
    c2.metric("Avg Goals / Game", f"{df['total_goals'].mean():.2f}")
    c3.metric("Highest Scoring", df.loc[df['total_goals'].idxmax(), 'result'])
    draws = (df['home_goals'] == df['away_goals']).sum()
    c4.metric("Draw Rate", f"{draws/len(df)*100:.1f}%")

    # Goals per game over time
    df_time = df.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_time["date"], y=df_time["total_goals"],
        mode="lines+markers", name="Goals / Game",
        line=dict(color="#2E7D32", width=2),
    ))
    fig.add_hline(
        y=df["total_goals"].mean(), line_dash="dash",
        line_color="gray", annotation_text="Average"
    )
    fig.update_layout(
        title="Goals Per Game Over Time",
        xaxis_title="Date", yaxis_title="Total Goals",
        template="plotly_white", height=380,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full results table
    st.markdown("### All Results")
    display_df = df[["date", "result", "total_goals"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["Date", "Result", "Total Goals"]
    st.dataframe(display_df, use_container_width=True, height=400)


def main():
    init_session_state()
    render_sidebar()

    # ---- Header ----
    st.title("⚽ Soccer Match Predictor")
    st.markdown(
        "Predict match outcomes using a **Poisson distribution Monte Carlo "
        "simulation** built on attack and defense ratings. "
        "Pick two teams, tune their ratings, and run the simulation."
    )

    # ---- Tab layout ----
    tab_predict, tab_live, tab_recent = st.tabs([
        "🎯 Predict",
        "📡 Live Ratings Dashboard",
        "📋 Recent Results",
    ])

    with tab_live:
        render_live_dashboard()

    with tab_recent:
        render_recent_results()

    with tab_predict:
        st.markdown("---")
        # ---- Team selection columns ----
        col_home, col_away = st.columns(2)

    with col_home:
        st.subheader("🏠 Home Team")
        home_team, home_attack, home_defense = render_team_selector(
            "Home", st.session_state.last_used["home_team"], "home"
        )

    with col_away:
        st.subheader("🚩 Away Team")
        away_team, away_attack, away_defense = render_team_selector(
            "Away", st.session_state.last_used["away_team"], "away"
        )

    if home_team == away_team:
        st.warning("⚠️ Home and Away teams are the same. Please pick two different teams.")

    st.markdown("---")

    # ---- Run simulation button ----
    run_clicked = st.button(
        f"🎲 Run {N_SIMULATIONS:,} Simulations",
        type="primary",
        use_container_width=True,
        disabled=(home_team == away_team),
    )

    if run_clicked:
        # Persist last-used ratings
        st.session_state.last_used = {
            "home_team": home_team,
            "away_team": away_team,
            "home_attack": home_attack,
            "home_defense": home_defense,
            "away_attack": away_attack,
            "away_defense": away_defense,
        }

        with st.spinner(f"Running {N_SIMULATIONS:,} Poisson simulations..."):
            home_xg = compute_expected_goals(home_attack, away_defense)
            away_xg = compute_expected_goals(away_attack, home_defense)
            results = run_monte_carlo_simulation(home_xg, away_xg, N_SIMULATIONS)

        st.session_state.sim_results = {
            "home_team": home_team,
            "away_team": away_team,
            "home_attack": home_attack,
            "home_defense": home_defense,
            "away_attack": away_attack,
            "away_defense": away_defense,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "results": results,
        }

    # ---- Display results ----
    sim = st.session_state.sim_results
    if sim is not None:
        home_team = sim["home_team"]
        away_team = sim["away_team"]
        home_xg = sim["home_xg"]
        away_xg = sim["away_xg"]
        results = sim["results"]

        st.markdown("## 📊 Prediction Results")

        # Team ratings recap
        rating_col1, rating_col2 = st.columns(2)
        with rating_col1:
            st.info(
                f"**{home_team}** — Attack: `{sim['home_attack']:.2f}` | "
                f"Defense: `{sim['home_defense']:.2f}`"
            )
        with rating_col2:
            st.info(
                f"**{away_team}** — Attack: `{sim['away_attack']:.2f}` | "
                f"Defense: `{sim['away_defense']:.2f}`"
            )

        # Metric cards: Win / Draw / Win
        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏆 {home_team} Win", f"{results['home_win_pct']*100:.1f}%")
        m2.metric("🤝 Draw", f"{results['draw_pct']*100:.1f}%")
        m3.metric(f"🏆 {away_team} Win", f"{results['away_win_pct']*100:.1f}%")

        # xG metric cards
        x1, x2 = st.columns(2)
        x1.metric(f"⚽ {home_team} Expected Goals", f"{home_xg:.2f}")
        x2.metric(f"⚽ {away_team} Expected Goals", f"{away_xg:.2f}")

        st.markdown("---")

        # Narrative prediction text
        narrative = build_prediction_narrative(home_team, away_team, home_xg, away_xg, results)
        st.markdown(narrative)

        # Draw warning box
        if results["draw_pct"] > DRAW_WARNING_THRESHOLD:
            st.warning(
                f"⚠️ **High Draw Probability ({results['draw_pct']*100:.1f}%)** — "
                "This matchup looks tight. Model suggests an elevated chance "
                "this game goes to **extra time and penalties**."
            )

        # Most likely scoreline callout
        score = results["most_common_score"]
        st.success(
            f"🎯 **Most Likely Scoreline:** {home_team} {score[0]} – "
            f"{score[1]} {away_team}  "
            f"(occurred in {results['most_common_score_pct']*100:.1f}% of simulations)"
        )

        st.markdown("---")

        # Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(
                plot_expected_goals(home_team, away_team, home_xg, away_xg),
                use_container_width=True,
            )
        with chart_col2:
            st.plotly_chart(
                plot_outcome_probabilities(home_team, away_team, results),
                use_container_width=True,
            )

        st.plotly_chart(
            plot_goal_distribution(
                home_team, away_team, results["home_goals"], results["away_goals"]
            ),
            use_container_width=True,
        )

        # Scoreline probability table (top 5 most likely scorelines)
        with st.expander("🔢 Top 5 Most Likely Scorelines (Poisson Probability Grid)"):
            grid = results["score_matrix"]
            rows = []
            for i in range(results["max_goals_grid"]):
                for j in range(results["max_goals_grid"]):
                    rows.append({"Home Goals": i, "Away Goals": j, "Probability": grid[i, j]})
            grid_df = pd.DataFrame(rows).sort_values("Probability", ascending=False).head(5)
            grid_df["Probability"] = (grid_df["Probability"] * 100).round(2).astype(str) + "%"
            grid_df = grid_df.reset_index(drop=True)
            st.dataframe(grid_df, use_container_width=True)

    else:
        st.info("👆 Configure your teams above and click **Run Simulations** to see predictions.")


if __name__ == "__main__":
    main()
