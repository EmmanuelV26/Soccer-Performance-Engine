"""
Soccer Player Recommendation Engine
=====================================
Uses scikit-learn to:
1. Find similar players (cosine similarity on normalized stats)
2. Find position-specific replacements
3. Scout undervalued players from smaller leagues

Algorithms used:
- StandardScaler for normalization
- NearestNeighbors (KNN) with cosine distance
- PCA for visualization
- Custom "value score" for scouting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════

df = pd.read_csv("/home/claude/soccer_players.csv")
print(f"Loaded {len(df)} players from {df['league'].nunique()} leagues")

# Feature columns for similarity (position-agnostic attacking + defending + passing)
FEATURE_COLS = [
    "goals_p90", "assists_p90", "xG_p90", "xA_p90",
    "shots_p90", "key_passes_p90", "tackles_p90",
    "interceptions_p90", "blocks_p90", "aerial_won_p90",
    "pass_completion", "progressive_passes_p90",
    "progressive_carries_p90", "dribbles_completed_p90"
]

# Drop GKs for outfield analysis (different stat profile entirely)
df_outfield = df[df["position"] != "GK"].copy().reset_index(drop=True)

# Fill any NaN with 0 for features
df_outfield[FEATURE_COLS] = df_outfield[FEATURE_COLS].fillna(0)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_outfield[FEATURE_COLS])

print(f"Outfield players: {len(df_outfield)}")
print(f"Features: {len(FEATURE_COLS)}")


# ══════════════════════════════════════════════════════════════════
# 2. RECOMMENDATION ENGINE CLASS
# ══════════════════════════════════════════════════════════════════

class PlayerRecommender:
    def __init__(self, df, X_scaled, feature_cols):
        self.df = df.copy()
        self.X = X_scaled
        self.features = feature_cols
        
        # Fit KNN model with cosine metric
        self.knn = NearestNeighbors(n_neighbors=20, metric="cosine")
        self.knn.fit(self.X)
        
        # Compute PCA for 2D visualization
        self.pca = PCA(n_components=2)
        self.coords_2d = self.pca.fit_transform(self.X)
        self.df["pca_x"] = self.coords_2d[:, 0]
        self.df["pca_y"] = self.coords_2d[:, 1]
        
        # Compute value score for scouting
        self._compute_value_scores()
        
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.1%}")
    
    def _compute_value_scores(self):
        """
        Value Score = Performance Rating / Market Value
        Higher = more undervalued (performing above their price tag)
        """
        # Performance rating: weighted combo of key stats
        perf = (
            self.df["goals_p90"] * 40 +
            self.df["assists_p90"] * 30 +
            self.df["xG_p90"] * 20 +
            self.df["xA_p90"] * 20 +
            self.df["key_passes_p90"] * 10 +
            self.df["progressive_carries_p90"] * 8 +
            self.df["progressive_passes_p90"] * 5 +
            self.df["tackles_p90"] * 8 +
            self.df["interceptions_p90"] * 8 +
            self.df["dribbles_completed_p90"] * 10
        )
        
        # Normalize to 0-100
        perf_norm = (perf - perf.min()) / (perf.max() - perf.min()) * 100
        self.df["performance_rating"] = perf_norm.round(1)
        
        # Value score (performance per million EUR)
        self.df["value_score"] = (perf_norm / self.df["market_value_EUR_M"].clip(lower=0.5)).round(2)
    
    def find_similar(self, player_name, n=10):
        """Find the N most similar players to a given player."""
        mask = self.df["name"].str.lower().str.contains(player_name.lower())
        if mask.sum() == 0:
            print(f"Player '{player_name}' not found!")
            return None
        
        idx = self.df[mask].index[0]
        player = self.df.iloc[idx]
        
        distances, indices = self.knn.kneighbors(self.X[idx:idx+1], n_neighbors=n+1)
        
        # Skip the first result (it's the player themselves)
        similar_idx = indices[0][1:]
        similar_dist = distances[0][1:]
        
        result = self.df.iloc[similar_idx].copy()
        result["similarity"] = (1 - similar_dist).round(3)  # cosine similarity
        
        print(f"\n{'='*60}")
        print(f"Players similar to: {player['name']}")
        print(f"  {player['position']} | {player['team']} | {player['league']}")
        print(f"  Age: {player['age']} | Value: €{player['market_value_EUR_M']}M")
        print(f"{'='*60}")
        
        cols = ["name", "position", "team", "league", "age", "market_value_EUR_M", "similarity"]
        print(result[cols].to_string(index=False))
        
        return result
    
    def find_replacement(self, player_name, position=None, max_value=None, 
                         max_age=None, exclude_leagues=None, n=10):
        """
        Find replacement players with optional filters:
        - position: filter by position
        - max_value: budget cap in EUR millions
        - max_age: age cap
        - exclude_leagues: list of leagues to exclude (e.g., same league)
        """
        mask = self.df["name"].str.lower().str.contains(player_name.lower())
        if mask.sum() == 0:
            print(f"Player '{player_name}' not found!")
            return None
        
        idx = self.df[mask].index[0]
        player = self.df.iloc[idx]
        
        # Get similarity to all players
        sim_matrix = cosine_similarity(self.X[idx:idx+1], self.X)[0]
        
        candidates = self.df.copy()
        candidates["similarity"] = sim_matrix
        
        # Apply filters
        candidates = candidates[candidates.index != idx]  # exclude self
        
        if position:
            candidates = candidates[candidates["position"] == position]
        else:
            candidates = candidates[candidates["position"] == player["position"]]
        
        if max_value:
            candidates = candidates[candidates["market_value_EUR_M"] <= max_value]
        
        if max_age:
            candidates = candidates[candidates["age"] <= max_age]
        
        if exclude_leagues:
            candidates = candidates[~candidates["league"].isin(exclude_leagues)]
        
        result = candidates.nlargest(n, "similarity")
        
        print(f"\n{'='*60}")
        print(f"Replacements for: {player['name']}")
        print(f"  {player['position']} | {player['team']} | €{player['market_value_EUR_M']}M")
        filters = []
        if max_value: filters.append(f"budget ≤ €{max_value}M")
        if max_age: filters.append(f"age ≤ {max_age}")
        if exclude_leagues: filters.append(f"excluding {', '.join(exclude_leagues)}")
        if filters: print(f"  Filters: {', '.join(filters)}")
        print(f"{'='*60}")
        
        cols = ["name", "position", "team", "league", "age", 
                "market_value_EUR_M", "performance_rating", "similarity"]
        print(result[cols].to_string(index=False))
        
        return result
    
    def scout_undervalued(self, position=None, max_value=10, max_age=26, 
                          tier_min=2, n=15):
        """
        Find undervalued gems: high performance relative to market value,
        from smaller leagues.
        """
        candidates = self.df.copy()
        
        if position:
            candidates = candidates[candidates["position"] == position]
        
        candidates = candidates[candidates["market_value_EUR_M"] <= max_value]
        candidates = candidates[candidates["age"] <= max_age]
        candidates = candidates[candidates["league_tier"] >= tier_min]
        candidates = candidates[candidates["minutes"] >= 800]  # enough playing time
        
        result = candidates.nlargest(n, "value_score")
        
        print(f"\n{'='*60}")
        print(f"UNDERVALUED SCOUT REPORT")
        print(f"  Filters: {'pos=' + position if position else 'all positions'}")
        print(f"  Value ≤ €{max_value}M | Age ≤ {max_age} | Tier ≥ {tier_min}")
        print(f"{'='*60}")
        
        cols = ["name", "position", "team", "league", "age",
                "market_value_EUR_M", "performance_rating", "value_score"]
        print(result[cols].to_string(index=False))
        
        return result


# ══════════════════════════════════════════════════════════════════
# 3. RUN THE ENGINE
# ══════════════════════════════════════════════════════════════════

rec = PlayerRecommender(df_outfield, X_scaled, FEATURE_COLS)

# Demo: Find similar players
similar = rec.find_similar(df_outfield.iloc[3]["name"])

# Demo: Find a replacement on a budget
rec.find_replacement(df_outfield.iloc[3]["name"], max_value=20, max_age=25)

# Demo: Scout undervalued wingers
rec.scout_undervalued(position="W", max_value=8, max_age=24)

# Demo: Scout undervalued strikers
rec.scout_undervalued(position="ST", max_value=10, max_age=25)


# ══════════════════════════════════════════════════════════════════
# 4. EXPORT DATA FOR INTERACTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════

# Prepare JSON for the frontend
export_df = rec.df.copy()
export_cols = [
    "name", "age", "position", "team", "league", "league_tier",
    "minutes", "market_value_EUR_M", "wage_EUR_K_week",
    "goals_p90", "assists_p90", "xG_p90", "xA_p90",
    "shots_p90", "key_passes_p90", "tackles_p90",
    "interceptions_p90", "progressive_passes_p90",
    "progressive_carries_p90", "dribbles_completed_p90",
    "pass_completion", "aerial_won_p90",
    "performance_rating", "value_score",
    "pca_x", "pca_y"
]

export_data = export_df[export_cols].round(3).to_dict(orient="records")

with open("/home/claude/players_export.json", "w") as f:
    json.dump(export_data, f, separators=(",", ":"))

print(f"\nExported {len(export_data)} players for dashboard")
print(f"JSON size: {len(json.dumps(export_data, separators=(',',':'))) / 1024:.0f} KB")
