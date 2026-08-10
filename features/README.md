# In-Game State & Lineup-Aware Features

## What this builds

One row per (match, minute) for all 380 matches of 2015/16 (36,125 rows total), with:

- `score_diff`, `home_goals`, `away_goals`, `minutes_remaining`
- `xg_diff`: running cumulative shot-xG differential (StatsBomb's `statsbomb_xg`), home minus away
- `momentum_5min_diff`, `momentum_10min_diff`: `possession_value/`'s rolling possession-value momentum, home minus away
- `red_cards_diff`: cumulative sendings-off (both "Red Card" and "Second Yellow" count identically -- both leave the team a player down)
- `subs_diff`: cumulative substitution count differential
- `sub_quality_diff`: cumulative sum of (incoming player's quality_per90 - outgoing player's), home minus away -- this is the lineup-aware part: subbing off a false 9 for a bench forward reads differently than a like-for-like change
- `home_formation` / `away_formation`: starting XI formation code
- `final_result`, `final_home_goals`, `final_away_goals`: the label, joined in from football-data.co.uk via `data/team_names.py`

## Player quality proxy (`features/player_quality.py`)

Season-long xT (from `possession_value/`) generated per 90 minutes played, shrunk toward the league-average rate for players with limited minutes (< 180 minutes gets meaningfully pulled toward the mean rather than trusted raw).

**Disclosed limitation:** this only credits ball-progression (passing/ carrying), not defending, aerial ability, or finishing -- it's a genuinely partial quality signal, biased toward creative players. Validated it's at least directionally sound: **Mesut Özil ranks #1** (he set the Premier League single-season assist record, 19, in this exact 2015/16 season), followed by Payet, De Bruyne, Mahrez, Silva, Fàbregas -- accurate, specific, well-known top performers of that season. Predictably weak at the bottom: goalkeepers and out-and-out target men (Benteke, Bony, Pellè), whose value isn't in progressing the ball -- the disclosed bias showing up exactly where expected.

## Disclosed simplification

Formation is the starting-XI formation only; mid-match tactical shifts (StatsBomb does log these) aren't tracked. Judged a second-order effect relative to score/time/momentum for this stage -- a documented possible extension, not built.

## Validation

1. **Spot check** (`3754217`, Chelsea vs Arsenal): this is the real September 2015 match where Arsenal had Gabriel (45') and Cazorla (78') both sent off and lost 2-0. The feature table's `red_cards_diff` drops to -2 at exactly those two minutes, and `score_diff` correctly ends at +2 for the home side.
2. **Whole-dataset check**: compared every match's StatsBomb-tracked final scoreline against football-data.co.uk's independently-sourced result for all 380 matches. **Zero mismatches** -- exact goal counts, not just win/draw/loss, agree on every single match. Also zero null values anywhere in the 36,125-row table.
3. **Formation sanity**: distribution across the season (4-2-3-1: 193/380, 4-4-2: 58, 4-1-4-1: 36, 4-3-3: 33, ...) matches known 2015/16 Premier League tactical trends (4-2-3-1's dominance that era is well documented).

## Output

`data/processed/ingame_features_2015_16.parquet` -- 36,125 rows, ready for the gradient boosting model in `models/`.
