# Net-probe bench — HANDOFF (arch-sweep prestudy)

Branch: `feat/net-probe` (over `main` @ 75c39d8). Date: 2026-07-18.

## What this is

Before testing NN architectures beyond the 64x64 towers, we built an
instrument that measures how much of the CURRENT nets' capacity is actually
used: per-layer activation spectra (participation ratio, effective rank),
unit health (tanh saturation / relu death), linear CKA (trained vs random
init, layer vs layer), and linear probes (outcome AUC + teacher-action
accuracy) — each probe compared against the SAME probe on the raw
observation, which is the control that makes probe numbers meaningful.

Built and verified on this box (23 tests green, ruff clean):

- `locma/stats/netdiag.py` — numpy-pure metrics (runs in CI without [ml]).
- `locma/stats/activations.py` — activation capture from a loaded
  MaskablePPO (flat MlpPolicy towers; token TokenSetExtractor also exposes
  `slots` = per-slot transformer output and `features` = fused head output),
  plus `reinit_clone` for the same-arch random-init baseline.
- `scripts/net_probe.py` — the driver: model + practicum in, JSON + table
  out (`runs/netprobe/<model>__<data>.json`).
- Tests: `tests/test_netdiag.py` (numpy-only), `tests/test_netdiag_activations.py`
  (importorskip-gated, tiny dummy env).

## Pilot numbers from this box (EXISTING practicums — see caveats)

Probe states are teacher-vs-baseline games (mcts:100 / dmcts teachers), NOT
freshly recorded held-out data — that is what the fast box should fix.

`ab-flat-s0.zip` on `practicum-flat.npz` (35k states, teacher mcts:100):

| layer  | width | PR   | PR/w  | PR/w init | sat units | outcome AUC | action acc |
|--------|------:|-----:|------:|----------:|----------:|------------:|-----------:|
| raw obs|  308  |  9.2 | 0.030 |     —     |     —     |   0.886     |   0.337    |
| pi_a1  |   64  | 14.1 | 0.221 |   0.127   |   0.97    |   0.842     |   0.300    |
| pi_a2  |   64  | 14.4 | 0.226 |   0.108   |   0.05    |   0.845     |   0.293    |
| vf_a1  |   64  |  8.2 | 0.128 |   0.125   |   0.67    |   0.909     |   0.298    |
| vf_a2  |   64  |  2.3 | 0.035 |   0.130   |   0.11    |   0.917     |   0.303    |
| logits |  155  |  5.5 | 0.035 |   0.044   |     —     |   0.845     |   0.293    |

model masked-argmax agreement with teacher: 0.341 (masked-majority 0.306).

`depot:b0k/b0k_s0.zip` on `practicum-dmcts.npz` (28k states, teacher dmcts):

| layer    | width | PR   | PR/w  | PR/w init | sat/duty  | outcome AUC | action acc |
|----------|------:|-----:|------:|----------:|----------:|------------:|-----------:|
| raw obs  |  373  |  7.9 | 0.021 |     —     |     —     |   0.871     |   0.377    |
| slots    | 1280  | 98.2 | 0.077 |   0.013   |     —     |   0.764     |   0.382    |
| features |  256  | 24.1 | 0.094 |   0.072   | duty 0.43 |   0.887     |   0.349    |
| pi_a1    |   64  | 16.2 | 0.253 |   0.229   | sat 0.38  |   0.868     |   0.333    |
| pi_a2    |   64  | 11.0 | 0.172 |   0.207   | sat 0.02  |   0.847     |   0.333    |
| vf_a1    |   64  |  3.1 | 0.049 |   0.213   | sat 0.88  |   0.904     |   0.343    |
| vf_a2    |   64  |  2.2 | 0.034 |   0.191   | sat 0.14  |   0.906     |   0.349    |
| logits   |  155  |  4.3 | 0.028 |   0.083   |     —     |   0.847     |   0.333    |

model agreement 0.388 (masked-majority 0.354). The b0k-on-mcts:100 run
(`practicum-token.npz`) is directionally identical.

### Preliminary reads (to confirm/refute on held-out data)

1. **No capacity pressure anywhere.** Policy towers use ~11–16 effective
   dims of 64; value towers collapse to PR 2–3 (classic trained-collapse:
   init PR/w ~0.2 → 0.03–0.05). Making the towers wider is unlikely to be
   the lever.
2. **First-layer tanh saturation is a conditioning problem, not capacity.**
   flat pi_a1: 97% of units saturated (unnormalized 308-d obs); token vf_a1
   88%. Argues for input normalization / LayerNorm variants in the sweep,
   not width.
3. **No hidden layer decodes the teacher's action better than the raw
   observation** (0.38 slots ≈ 0.377 raw; every downstream layer LOSES
   tactical information). And the full trained net's own argmax barely beats
   a linear probe on raw inputs (0.388 vs 0.377). Consistent with the
   distill finding (~0.37 BC agreement cap, worklog 2026-06-27) and with
   search-at-play-time being the real lever, not the trunk.
4. The critic genuinely computes: value layers beat raw obs on outcome AUC
   (0.906 vs 0.871) despite PR ~2 — a low-dim but real value feature.

## What to run on the fast box

Setup (once):

```bash
git fetch origin feat/net-probe && git checkout feat/net-probe
uv sync --extra ml --extra dev
uv run locma depot pull b0k          # token recipe of record, 3 seeds
```

### Step 1 — record HELD-OUT probe practicums (the real data)

Seeds ≥ 1_000_000 (outside every training stream). netdmcts is the fair
teacher of record (~13 s/game, single-threaded) — shard by opponent and run
shards in parallel (one process each):

```bash
# ~50 games x 2 seats per shard; 4 shards in parallel ≈ 2-3 h wall
for opp in scripted greedy max-guard max-attack; do
  uv run locma record-practicum --teacher netdmcts --opponents $opp \
    --games 50 --seed 1000000 --obs-mode token \
    --out runs/probe-netdmcts-$opp.npz &
done
# cheap mcts:100 companion set (minutes), same seeds, token + flat:
uv run locma record-practicum --teacher mcts:100 --games 150 --seed 1000000 \
  --obs-mode token --out runs/probe-mcts-token.npz
uv run locma record-practicum --teacher mcts:100 --games 150 --seed 1000000 \
  --obs-mode flat --out runs/probe-mcts-flat.npz
```

Merge the netdmcts shards (game_id must be offset per shard or the
game-level probe split leaks):

```python
import json, numpy as np
shards = [f"runs/probe-netdmcts-{o}.npz" for o in
          ("scripted", "greedy", "max-guard", "max-attack")]
merged, offset = {}, 0
for p in shards:
    d = dict(np.load(p))
    d["game_id"] = d["game_id"] + offset
    offset = int(d["game_id"].max()) + 1
    for k, v in d.items():
        merged.setdefault(k, []).append(v)
np.savez_compressed("runs/probe-netdmcts.npz",
                    **{k: np.concatenate(v) for k, v in merged.items()})
man = json.load(open("runs/probe-netdmcts-scripted.manifest.json"))
man["opponents"] = ["scripted", "greedy", "max-guard", "max-attack"]
man["engine_version"] = "merged-shards"
json.dump(man, open("runs/probe-netdmcts.manifest.json", "w"), indent=2)
```

(Or skip merging and probe each shard separately — smaller N, same picture.)

### Step 2 — the probe matrix (~3 min per cell, CPU)

```bash
for s in s0 s1 s2; do   # seed replication: is the picture stable?
  uv run python scripts/net_probe.py --model depot:b0k/b0k_$s.zip \
    --data runs/probe-netdmcts.npz
  uv run python scripts/net_probe.py --model depot:b0k/b0k_$s.zip \
    --data runs/probe-mcts-token.npz
done
# flat arm, if a current flat checkpoint exists on that box (else skip —
# pilot above already covers flat on this box's ab-flat-s0):
uv run python scripts/net_probe.py --model runs/<flat-model>.zip \
  --data runs/probe-mcts-flat.npz
```

Optional 3rd axis if training checkpoints exist (e.g. `b0k-<steps>.zip`
siblings): probe 3–4 checkpoints of one run on the same data — when does the
representation stop moving (CKA between checkpoints, PR trajectory)?

### Step 3 — bring back results

`runs/netprobe/*.json` are small; commit them on this branch (add a
`!runs/netprobe` carve-out or copy under `docs/notes/`), or just paste the
printed tables into the PR/worklog.

## Decision criteria for the arch sweep (write the verdict against these)

- PR/width high (> ~0.5) or heavy saturation WITH high PR → capacity-bound
  → width/depth variants justified.
- PR/width low + probes flat across depth (the pilot picture) → capacity is
  NOT the lever; sweep should target conditioning (obs normalization,
  LayerNorm placement) and the head/objective, or stop and accept that the
  plateau is representational content — search remains the lever.
- Any layer beating raw obs on teacher-action accuracy by ≥ +0.03 → the
  trunk computes usable tactical features the head wastes → head/objective
  work, not trunk scaling.

## Caveats

- Probe states are from teacher-vs-baseline games, off-distribution vs
  b0k's own play. If the held-out picture is ambiguous, add an on-policy
  practicum (`--teacher` = the model under study, via its policy spec) as a
  robustness check.
- `reinit_clone` uses PyTorch default init, not SB3's orthogonal init —
  fine for spectrum-at-init comparisons, don't read small init deltas.
- Ridge probes use fixed `--l2 1.0` on standardized features (N >> D, the
  penalty is nearly inert). Probe deltas < ~0.01 are noise; use the seed
  replication to gauge spread.
