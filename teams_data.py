"""
teams_data.py
-------------
Default national-team ratings used to seed the app.

Ratings are on a 40-95 scale:
    attack  -> goal-scoring threat
    defense -> defensive solidity (HIGHER = better defense)

These are illustrative sample ratings, not official FIFA/Elo numbers.
Users can override everything via the UI or load their own CSV.
"""

DEFAULT_TEAMS = {
    "Morocco":      {"attack": 71.02, "defense": 77.90},
    "Netherlands":  {"attack": 74.82, "defense": 73.57},
    "France":       {"attack": 86.50, "defense": 81.20},
    "Brazil":       {"attack": 88.10, "defense": 76.40},
    "Argentina":    {"attack": 85.30, "defense": 79.80},
    "England":      {"attack": 82.60, "defense": 78.90},
    "Spain":        {"attack": 83.90, "defense": 77.10},
    "Germany":      {"attack": 81.20, "defense": 75.60},
    "Portugal":     {"attack": 80.40, "defense": 76.80},
    "Belgium":      {"attack": 78.70, "defense": 74.30},
    "Croatia":      {"attack": 75.90, "defense": 76.50},
    "Italy":        {"attack": 76.30, "defense": 79.40},
    "USA":          {"attack": 68.50, "defense": 70.20},
    "Japan":        {"attack": 72.40, "defense": 73.10},
    "Senegal":      {"attack": 73.80, "defense": 72.60},
    "Nigeria":      {"attack": 72.90, "defense": 71.40},
}
