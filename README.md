<h1>Chess Elo Prediction from PGN Games</h1>

This project explores whether player strength (measured by the average Elo rating of both players) can be predicted using only the algebraic notation of a chess game. Using 20,000 Lichess games from 2017, the project builds a full machine-learning pipeline that parses PGN files, extracts features from move text, and trains a model to estimate the players’ Elo.

The best model achieved:

Training R²: 0.7285

Testing R²: 0.3526

