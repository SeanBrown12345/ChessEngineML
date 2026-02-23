# ♟️ Estimating Chess Player Strength from Move Sequences

## Overview

This project investigates whether a chess player's strength (average Elo rating) can be estimated using only the moves of a single game written in Standard Algebraic Notation (SAN).

Using 20,000 Lichess games from 2017, I built a machine learning pipeline in Python to:

- Parse PGN files
- Extract structured metadata
- Engineer text-based features from move sequences
- Train an ensemble regression model to predict average Elo

The target variable is the mean rating of both players:

mean_elo = (white_elo + black_elo) / 2

---

## Dataset

- Source: Lichess (2017)
- Size: 20,000 games
- Format: PGN
- Filter: Only games with valid integer Elo values

Extracted features include:
- Player ratings
- Opening and ECO code
- Time control
- Termination type
- Full move sequence
- Move count

---

## Feature Engineering

Several approaches were tested:

**Full-Game TF-IDF:**  
TF-IDF vectorization over entire move sequences.

**Opening-Only TF-IDF (Best Performing):**  
Restricting to the first 15 moves significantly improved generalization.

**Handcrafted Features:**  
Blunder heuristics, queen moves, checks, and move count were tested but added noise and increased overfitting.

Key finding: Opening moves contain structured patterns more strongly correlated with rating than later game phases.

---

## Modeling Approach

Preprocessing:
- OneHotEncoder for time control
- TF-IDF vectorization (max_features=2000, ngram_range=(1,2), min_df=3)

Model:
- VotingRegressor combining:
  - RandomForestRegressor
  - GradientBoostingRegressor

Hyperparameters were tuned using RandomizedSearchCV (3-fold cross-validation).

---

## Results

| Metric       | Score  |
|-------------|--------|
| Training R² | 0.7285 |
| Testing R²  | 0.3526 |

The performance gap indicates overfitting, likely due to dataset size and the high dimensionality of move-based features. However, the test score confirms that rating-related signal exists within opening sequences.

---

## Installation

```bash
pip install pandas numpy scikit-learn python-chess scipy joblib
