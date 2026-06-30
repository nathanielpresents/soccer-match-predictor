# ⚽ Soccer Match Predictor

A Streamlit web app that predicts soccer match outcomes using a **Poisson
distribution Monte Carlo simulation**, generating commentary-style
predictions like:

> "This is my prediction for Morocco versus Netherlands. Based on 10,000
> roads worth of data... Netherlands a 43% chance of winning in regular
> time, Morocco a 27% chance and a tie of 32% chance... after running
> 10,000 simulations of a Poisson model, Morocco has an expected goal of
> 1.19, Netherlands 2 goals... model is predicting a tie in regular time...
> high chance this game goes to extra time and penalties."

## Features

- Poisson-based Monte Carlo simulation (10,000+ trials per prediction)
- Expected Goals (xG) calculated from Attack vs. opposing Defense ratings
- Home Win / Draw / Away Win probability breakdown
- Most likely scoreline detection
- Automatic "extra time & penalties" warning when draw probability > 32%
- Interactive Plotly charts: xG comparison, outcome probabilities, goal
  distribution histogram, and a top-5 scoreline probability table
- 16 pre-loaded national teams (Morocco, Netherlands, France, Brazil,
  Argentina, England, Spain, Germany, Portugal, Belgium, Croatia, Italy,
  USA, Japan, Senegal, Nigeria)
- Manually adjustable Attack/Defense sliders (40–95 range)
- Add custom teams via the sidebar
- Load/override team ratings from a CSV file (`team, attack, defense`)
- Session-state persistence of the last-used ratings

## Project Structure

```
soccer-match-predictor/
├── app.py              # Main Streamlit application
├── teams_data.py        # Default national team ratings
├── sample_teams.csv     # Example CSV for the "Load Teams from CSV" feature
├── requirements.txt     # Python dependencies
└── README.md             # This file
```

## Setup & Run

1. **Clone or download** this project folder.

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`) in your
   browser.

## How the Model Works

1. **Expected Goals (xG):** For each team, xG is calculated as:

   ```
   xG = league_avg_goals * (attack / baseline) * (baseline / opponent_defense)
   ```

   where `league_avg_goals = 1.35` and `baseline = 75` (an "average"
   national team rating). A stronger attack increases xG; a stronger
   opposing defense decreases it.

2. **Monte Carlo Simulation:** For each of 10,000 trials, goals for each
   team are drawn from a Poisson distribution with mean equal to that
   team's xG (`numpy.random.default_rng().poisson`). Win/Draw/Loss
   outcomes are tallied across all trials to estimate probabilities.

3. **Most Likely Scoreline:** The most frequent exact scoreline across all
   simulated trials is reported, along with its probability.

4. **Extra Time Warning:** If the simulated draw probability exceeds 32%,
   the app flags a high likelihood of the match going to extra time and
   penalties (common in knockout-stage analysis).

## Loading Custom Teams via CSV

Use the sidebar **"📂 Load Teams from CSV"** expander. Your CSV must have
the following columns (case-insensitive):

```csv
team,attack,defense
Morocco,71.02,77.90
Netherlands,74.82,73.57
```

A ready-to-use example is included as `sample_teams.csv`.

## Adding a New Team

Use the sidebar **"➕ Add a New Team"** expander to manually enter a team
name plus Attack and Defense ratings (40–95), then click **Add Team**. The
new team immediately becomes available in both the Home and Away dropdowns.

## Disclaimer

The pre-loaded ratings are illustrative sample values for demonstration
purposes, not official FIFA, Elo, or any other rating-provider data. This
is a statistical/educational simulation, not betting advice.
