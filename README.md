# ⚽ Soccer Player Recommendation Engine

A machine learning-powered scouting tool that finds **statistically similar players** across 12 leagues and surfaces **undervalued talent** from smaller leagues. Built with Python, scikit-learn, and an interactive HTML dashboard.

![Dashboard Screenshot](images/dashboard_screenshot.png)

---

##  What This Project Does

### 1. Find Similar Players
Search any player and instantly get their closest statistical matches across 3,370 players. The engine compares 14 per-90-minute metrics using **cosine similarity**, which focuses on the *shape* of a player's stat profile rather than raw numbers.

### 2. Scout Undervalued Players
A custom **Value Score** (performance rating ÷ market value) identifies players who are performing above their price tag — especially in smaller leagues like the Eredivisie, Primeira Liga, Austrian Bundesliga, and MLS.

### 3. Find Position-Specific Replacements
Need a replacement winger under €10M and younger than 24? The engine filters by position, budget, age, and league tier, then ranks candidates by statistical similarity.

### 4. Visual Player Map
**PCA** reduces 14 statistical dimensions down to 2, creating a scatter plot where players naturally cluster by playing style. Strikers group together, defensive midfielders group together — without the algorithm being told positions.

---

##  Algorithms Used

| Algorithm | Purpose | Library |
|-----------|---------|---------|
| **K-Nearest Neighbors (KNN)** | Find the N most similar players to any target | scikit-learn |
| **Cosine Similarity** | Measure how similar two player profiles are (0–100%) | scikit-learn |
| **PCA** | Reduce 14 dimensions → 2 for visualization (78.5% variance explained) | scikit-learn |
| **StandardScaler** | Normalize stats so tackles and goals are weighted fairly | scikit-learn |
| **Custom Value Score** | Weighted performance rating ÷ market value for scouting | NumPy |

---

## 📊 Features Analyzed (per 90 minutes)

**Attacking:** Goals, Assists, xG, xA, Shots, Key Passes, Dribbles Completed

**Defending:** Tackles, Interceptions, Blocks, Aerial Duels Won

**Passing & Progression:** Pass Completion %, Progressive Passes, Progressive Carries

---

## 🛠️ Tech Stack

- **Python** — core language
- **scikit-learn** — KNN, PCA, StandardScaler, cosine similarity
- **Pandas / NumPy** — data processing and feature engineering
- **Chart.js** — interactive HTML dashboard
- **HTML / CSS / JavaScript** — frontend with real-time filtering and cosine similarity computed in the browser

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas scikit-learn
```

### Generate the player dataset
```bash
python generate_data.py
```
Creates `soccer_players.csv` with 3,370+ players across 12 leagues with realistic FBref-style statistics.

### Run the recommendation engine
```bash
python soccer_recommender.py
```
This will:
- Load and normalize the player data
- Fit KNN and PCA models
- Print example results for similar players, replacements, and undervalued scouts
- Export data for the interactive dashboard

### Example usage in your own code
```python
from soccer_recommender import PlayerRecommender

# ... load data and scale features ...

rec = PlayerRecommender(df_outfield, X_scaled, FEATURE_COLS)

# Find players similar to a target
rec.find_similar("Player Name", n=10)

# Find budget-friendly replacements
rec.find_replacement("Player Name", max_value=15, max_age=24)

# Scout undervalued wingers from smaller leagues
rec.scout_undervalued(position="W", max_value=8, max_age=24, tier_min=2)
```

### Open the interactive dashboard
Open `soccer_recommender.html` in any browser. No server needed — everything runs client-side.

---

##  Project Structure

```
soccer-player-recommender/
├── README.md
├── generate_data.py             # Creates realistic player dataset
├── soccer_recommender.py        # Recommendation engine (scikit-learn)
├── soccer_recommender.html      # Interactive dashboard
├── data/
│   └── soccer_players.csv       # 3,370 players, 12 leagues
└── images/
    └── dashboard_screenshot.png
```

---

##  What I Learned

- **Cosine similarity vs. Euclidean distance.** Cosine similarity compares the *direction* of two vectors, not their magnitude. This means a winger with 0.3 goals/90 and 0.2 assists/90 is considered similar to one with 0.6 goals/90 and 0.4 assists/90 — they have the same profile shape, just at different levels. For player comparison, this makes more sense than raw distance.

- **PCA reveals hidden structure.** When I plotted the 2D PCA projection, positions naturally clustered together without the algorithm knowing positions existed. The first principal component roughly captured "attacking output" while the second captured "creativity and progression."

- **Value Score is simple but powerful.** Dividing a performance rating by market value is a straightforward idea, but it consistently surfaced interesting players from the Austrian Bundesliga, Portuguese league, and MLS who had elite per-90 stats at a fraction of the cost.

- **Normalization decides everything.** Without `StandardScaler`, pass completion (70-95%) would dominate over goals per 90 (0-0.7). After normalizing, every stat contributes equally to similarity calculations.

- **Real scouting is harder.** This engine uses per-90 stats, but real scouts also consider physicality, mentality, tactical fit, injury history, and contract situations. ML is a powerful starting point, not the whole answer.

---

##  Future Improvements

- [ ] Plug in real data from FBref or Kaggle
- [ ] Add radar charts for player comparison
- [ ] Include goalkeeper-specific analysis
- [ ] Add league difficulty adjustment (a goal in the Premier League ≠ a goal in the Austrian Bundesliga)
- [ ] Time-series analysis to track player development over seasons

---

##  Contact

Built by **Manny** — Business Administration (MIS) student at Washington State University.

This is project #2 in my ML portfolio. Project #1: [Stock Market Clustering with K-Means from Scratch](https://github.com/YOUR-USERNAME/ml-stock-clustering)

Feel free to connect on [LinkedIn](https://linkedin.com/in/YOUR-PROFILE) or reach out with questions!
