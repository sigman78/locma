# E36 PFSP — Apple Silicon re-bootstrap runbook

Goal: an **independent replication** of the E36 PFSP scale curve on a second box
(M1 Max, CPU), fresh seed, larger `n_envs`. It re-bootstraps the *whole* chain
gen0->gen3 so `n_envs` is **constant within the chain** — the search-gap
comparison is always within-chain, so n_envs is never a confound. We read the
**direction** of the trend (does the search gap dent across generations?), not the
absolute magnitude (which will differ from x86 because n_envs, platform, and seed
all differ — that's fine and expected).

This is a separate artifact from the x86 gen0-4 chain, not a continuation.

## 1. Clone + environment

Use Python 3.11 or 3.12 (safest torch wheels on Apple Silicon; project needs >=3.11).

```
git clone <repo-url> locma && cd locma
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"
```

On macOS the interpreter is `.venv/bin/python` (not `.venv/Scripts/python`).

Do NOT copy the x86 `runs/` directory over — start clean so the driver bootstraps
a fresh pool and writes its own gen0.

## 2. Fetch the two depot artifacts

gen0 warm-starts from `depot:e29slim` and drafts with `depot:ldraft`. The git
clone already carries the depot *index*; pull fetches + hash-verifies the *blobs*
(needs the `gh` CLI authenticated: `gh auth status`).

```
locma depot pull e29slim
locma depot pull ldraft
# sanity: both must resolve to a local path
locma depot resolve depot:e29slim/e29slim_s0.zip
locma depot resolve depot:ldraft/ldraft_s0.zip
```

If `pull` finds nothing (artifacts local-only on x86, never pushed to the remote):
either run `locma depot push e29slim ldraft` from the x86 box first, or just copy
the x86 `depot/` folder (index + store, self-contained) over to the Mac.

## 3. Sanity bench (optional, ~1 min)

```
python scripts/e36_bench.py --n-envs 12 --device cpu
```
Expect ~600+ fps and `"learner_device": "cpu"` in the JSON. (This is throughput
only — n12 is fine here; see the regime note in step 4 for why the *training* run
also uses 12.)

## 4. Run the fresh chain gen0 -> gen3

```
python scripts/e36_pfsp.py \
  --start-gen 0 --generations 4 \
  --steps 1500000 --n-envs 12 --device cpu \
  --seed 20000000 \
  2>&1 | tee runs/e36/m1_gen0-3.log
```

- **No `--resume`** — fresh `SEED_POOL` (e29slim self + boardkeep/scripted/
  max-guard/max-attack anchors), written to `runs/e36/pool.json`.
- `--warm` defaults to `depot:e29slim/e29slim_s0.zip` (gen0 warm-start), same as x86.
- Regime vs x86 gen0-4: identical hyperparameters and 1.5M steps/gen; **differs**
  only in `n_envs` (12 vs 6), platform (M1/CPU vs x86/CUDA), and seed (20M vs 14M).
  n_envs is held constant across gen0-3, so the within-chain trend stays clean.
- Throughput: ~635 fps -> ~0.66 h/gen -> **~2.6 h total** + a few min/gen for the
  driver's `wr_vs_pool` eval.
- Outputs: `runs/e36_gen{0,1,2,3}.zip`, `runs/e36/pool.json`,
  `runs/e36/history.json`, `runs/e36/m1_gen0-3.log`.

Seed 20,000,000 is deliberately clear of x86 training (14M) and the held-out gate
eval seeds (30M/40M).

## 5. Gate eval (the payoff metric) — coordinate back

The `wr_vs_pool` the driver logs each gen is only the cheap internal check. The
real signal is the **held-out search gap** (fair search `rbeam:shared` win-rate vs
the pure net) plus avg-hard3. Two ways to get it:

- **(a) Ship checkpoints back (recommended, fastest).** Copy `runs/e36_gen{0..3}.
  zip` to the x86 box; the existing gate harness runs there. Eval is a statistical
  win-rate measurement, so it is platform-robust — evaluating M1-trained nets on
  x86 is fine.
- **(b) Eval in place on the Mac.** Slower — `rbeam:shared` is a heavy search
  opponent on CPU — and needs the gate harness parameterized for gen0-3.

## What "confirmed" looks like

- **Internal FP health:** each gen beats the prior self head-to-head (gen1>gen0,
  gen2>gen1, ...), roughly monotone.
- **Held-out (the real test):** fair-search win-rate over the pure net **drops**
  across gen0(control) -> gen1 -> gen2 -> gen3. Direction and monotonicity are
  what confirm the effect; the exact values will not match x86's 0.807 -> 0.703 ->
  0.588 and are not expected to.

A clean downward search-gap trend here = the E36 dent is robust across platform,
n_envs, and seed — a stronger result than a same-box seed replicate.
