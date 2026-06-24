# Experiment Methodology

## Noise floor (luck baseline)
`noise-floor` plays a policy against an independent copy of itself. It answers:
"how big must a win-rate edge be before it's real?"
- **Stochastic policies** (random, scripted): win rate centers on 0.50; the CI
  width is the measurement floor. (Both are seeded per game seed, so a logged
  game still replays byte-identically — stochastic across seeds, reproducible
  within one.)
- **Deterministic policies** (greedy): self-play variance comes only from seat
  asymmetry and the seed's RNG draws, so the win rate may sit stably off 0.50.
  Read the **resolution limit** (CI half-width), not the point value: any
  measured edge smaller than it is indistinguishable from luck.

## SPRT (sequential testing)
`sprt` tests H0: winrate = p0 against H1: winrate = p1 using Wald's
log-likelihood ratio, batching games until the LLR crosses an acceptance
boundary (alpha = beta = 0.05) or `--max-games` is hit. It stops as soon as the
evidence decides — far fewer games than a fixed-n test for clear effects.

## Ratings: Elo and openskill
`tournament` reports both. Elo is the classic pairwise update; openskill
(Plackett-Luce) tracks a (mu, sigma) belief and reports a conservative
**ordinal** = mu - 3*sigma. openskill is the primary number (it models
uncertainty); Elo is kept for continuity and comparison.

## Replay & determinism
Every game is a deterministic function of (seed, policies). `play --log` records
the action sequence and a content hash = sha256(canonical_json(actions +
[winner, turns])). `replay --assert-hash` re-runs from the seed and fails if the
recomputed hash differs — catching any accidental change to engine or policy
behavior.
