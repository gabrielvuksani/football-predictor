# v16 "Sentinel" — Accuracy Breakthrough Spec

## Goal

Fix remaining data leakage (early stopping on test set), prune ~90 noise features, add FBref advanced stats, add draw-specific features, add rolling temporal features, implement per-league model clusters, run SHAP importance pruning, and establish honest accuracy on a leak-free pipeline.

## Critical Fix: Early Stopping Leakage

CatBoost and XGBoost currently use `eval_set=(Xte, yte)` for early stopping — the models choose when to stop based on the test set they'll be evaluated on. This inflates accuracy by ~2-3%.

**Fix:** Use the calibration split for early stopping:
```python
cat_base.fit(Xtr, ytr, eval_set=(Xcal, ycal))  # NOT Xte
xgb_base.fit(Xtr, ytr, eval_set=[(Xcal, ycal)])  # NOT Xte
```

## Data Quality: FBref Team Stats via soccerdata

Ingest per-team rolling stats from FBref (free, top 5 leagues):
- Possession %, PPDA, xG/xGA, shots/SOT, pass completion %, pressing success rate

Store in a new `fbref_team_stats` DuckDB table with columns:
`team, competition, season, matchday, possession, ppda, xg, xga, shots, sot, pass_pct, pressing_success, progressive_passes`

Compute rolling 5-game averages at prediction time.

## Feature Pruning: Remove Noise

Remove features that are mostly zeros across the training set (>80% zero rate):
- All news sentiment features (until news coverage improves)
- All lineup features outside PL
- All opta features (unless coverage exceeds 50%)
- Upset interaction features that multiply two near-zero signals
- Redundant form features (keep form_pts, drop derivative signals)

Target: ~40-50 clean, high-signal features instead of ~133 noisy ones.

## Draw-Specific Features (New)

- `draw_elo_gap`: abs(elo_diff) — small gap = draw-prone
- `draw_both_low_xg`: both teams rolling xG < 1.2
- `draw_fixture_history`: historical draw rate for this matchup
- `draw_league_rate`: league-level seasonal draw rate
- `draw_home_form_match`: similar recent PPG = draw indicator

## Rolling Temporal Features (3/5/10 windows)

For key metrics, compute at 3 distinct windows:
- Goals scored/conceded, xG/xGA, PPG, clean sheets

## Per-League Model Clusters

3 clusters by playing style:
- Attacking (BL1, DED, FL1): high-scoring, fewer draws
- Balanced (PL, PD, ELC): moderate
- Defensive (SA, TR1, GR1): low-scoring, more draws

Each cluster gets CatBoost head trained on its data. Small leagues use nearest cluster.

## Success Criteria

1. Honest walk-forward accuracy (no leakage) exceeds 53% (bookmaker baseline)
2. Draw prediction precision exceeds 30%
3. Feature count reduced from ~133 to ~50 with equal or better accuracy
4. FBref data coverage for top 5 leagues
