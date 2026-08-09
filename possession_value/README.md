# Step 4: Possession-Value Momentum Model

## What this builds

An Expected Threat (xT) model in the style of Karun Singh's public xT
work: a 16x12 grid over the pitch, where each zone's value is solved by
value iteration:

    xT[z] = shot_prob[z] * avg_shot_value[z] + move_prob[z] * sum_dest(transition[z,dest] * xT[dest])

A completed pass or carry is credited `xT[end_zone] - xT[start_zone]` --
the change in positional threat it created. This is a genuinely richer
signal than raw shot/xG counts: it credits good buildup play (progressive
passes into dangerous areas) even in spells that don't end in a shot at
all.

## Data scope

Fit on **2015/16 only** (380 matches, 655,476 actions: 368,619 passes,
276,949 carries, 9,908 shots) -- per `data/README.md`'s documented
constraint, the 2003/04 StatsBomb set is Arsenal-only and excluded here
to avoid biasing the grid toward one team's tactics.

## Disclosed judgment call: coordinate normalization

StatsBomb's raw pitch coordinates aren't normalized for attacking
direction -- a team's x-coordinate increases toward whichever goal it's
attacking *in that period*, and teams switch ends at half-time. Before
fitting, every team-period's actions are flipped as needed so x always
increases toward the attacking team's target goal, inferred from that
team's own shot locations in that period (falling back to the opponent's
shots, since they attack the opposite end by construction). See
`possession_value/data.py` docstring for the full fallback chain.

## Sanity check (`possession_value/fit_and_check.py`)

The plan calls for checking that threat increases monotonically moving
toward goal. Collapsing the grid to mean xT per x-bin (0 = own goal end,
15 = opponent's goal end):

```
x_bin  0:   2.97       x_bin  8:  10.62
x_bin  1:   3.66       x_bin  9:  12.76
x_bin  2:   4.29       x_bin 10:  15.45
x_bin  3:   5.00       x_bin 11:  18.78
x_bin  4:   5.77       x_bin 12:  23.68
x_bin  5:   6.61       x_bin 13:  34.83
x_bin  6:   7.68       x_bin 14:  55.58
x_bin  7:   8.98       x_bin 15:  73.15
```

**0 of 15 consecutive steps decrease** -- perfectly monotonic, and the
shape (slow rise through midfield, sharp spike in the final two bins,
strongest in the central channel right in front of goal) matches Karun
Singh's originally published xT grid closely, despite being fit on a
completely different, much smaller dataset (one EPL season vs. multiple
leagues/seasons).

## Momentum feature (`possession_value/momentum.py`)

Per match, per minute, per team: threat generated that minute, plus
rolling 5-minute and 10-minute trailing sums (`compute_momentum`), and a
home-minus-away differential (`momentum_differential`) -- the actual
feature Step 5/6 will consume.

**Validation (`possession_value/demo_momentum.py`):** ran this on
Southampton's real 4-0 win at Arsenal (2015/16). The scoring team's
momentum spikes sharply at or immediately before every one of the 4
goals -- e.g. `threat_this_minute=201.75` (x1000) in the exact minute
Fonte scored Southampton's 3rd, `227.50` in the exact minute Long scored
the 4th, and a clear building momentum through minutes 88-91 ahead of his
stoppage-time goal. This is a genuine, goal-independent buildup signal,
not an artifact of the shots themselves (shots are excluded from the
momentum sum -- only completed passes/carries count).

## Outputs

- `data/processed/possession_actions_2015_16.parquet` -- all 655K parsed
  actions, direction-normalized.
- `data/processed/xt_grid_2015_16.npz` -- the fitted xT grid + zone
  statistics (`possession_value.xt_model.ExpectedThreat.load(...)`).

## How this feeds forward

Step 5 will pull `momentum_differential()` into the in-game feature
table alongside score/time/cards/subs. Step 7's big-moment detection can
also use large single-action xT deltas directly (e.g. a progressive pass
that spikes threat by 0.2+ in one action) as a candidate signal, separate
from win-probability swings.
