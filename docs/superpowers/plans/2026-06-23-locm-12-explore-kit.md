# LoC&M 1.2 Explore Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python kit that simulates single-lane Legends of Code & Magic (LOCM) 1.2, lets pluggable policies play draft + battle, and runs reproducible matches/tournaments with statistical hypothesis testing — with a Gymnasium/SB3 layer on top.

**Architecture:** A pure, dependency-light rules engine at the center (`locma.core`); policies, a Gymnasium adapter, a match/tournament harness, a stats module, and a Typer CLI are layers around it. ML deps live behind an optional `[ml]` extra and never enter the core. Everything random flows through one seeded RNG so games are deterministic and replayable.

**Tech Stack:** Python 3.11+, Typer + Rich (CLI), numpy, scipy (stats), pytest (tests); optional `[ml]` extra = Gymnasium + stable-baselines3 + sb3-contrib (MaskablePPO) + torch, optional `trueskill`. Dependency manager: `uv` (pip/venv fallback).

## Global Constraints

- Python 3.11+ only; use `from __future__ import annotations` in modules with forward type refs.
- Single lane only (LOCM 1.2). No multi-lane (1.5) anywhere.
- The `locma.core` package MUST NOT import gymnasium, torch, stable-baselines3, numpy-only-for-ML, or any `[ml]` dependency. (numpy is allowed in `core` only if needed; prefer pure stdlib in core.)
- All randomness flows through a single seeded `random.Random` stored on the game state. No module-level `random.*` calls, no `random.random()` outside the seeded instance.
- 160 cards. Any card-data load MUST assert `len(cards) == 160`.
- Abilities canonical order is the 6-char string `"BCDGLW"` = Breakthrough, Charge, Drain, Guard, Lethal, Ward. Absent ability = `'-'`.
- TDD: every task writes a failing test first, then minimal code. Commit after each task.
- Run tests with `uv run pytest` (or `pytest` if not using uv).

---

## File Structure

```
pyproject.toml                  package metadata, deps, [ml] extra, console script
locma/__init__.py
locma/core/
  __init__.py
  cards.py        Card, CardType, ABILITY_ORDER, ability helpers
  instance.py     CardInstance (runtime mutable wrapper)
  rng.py          (thin) seeded RNG helpers if needed
  state.py        PlayerState, GameState, Phase enum
  actions.py      Action dataclasses (Summon, Attack, Use, Pass), draft = int
  draft.py        draft setup + legal picks + apply pick
  battle.py       battle legal_actions + apply (rules: keywords, mana, rune)
  engine.py       top-level run_game(policy_a, policy_b, seed) orchestration + views
  views.py        DraftView, BattleView sanitized snapshots
locma/data/
  __init__.py
  cardlist.txt    vendored 160-card definitions
  cards_db.py     load_cards(), parse_cardlist()
  fetch.py        fetch_cards(), fetch_art() CLI helpers
  assets/         (best-effort art, gitignored except manifest)
locma/policies/
  __init__.py
  base.py         Policy protocol, CompositePolicy
  random_policy.py
  scripted.py
  greedy.py
  sb3_policy.py   (in [ml] path; imports guarded)
locma/envs/
  __init__.py
  encode.py       observation vector + action index <-> Action mapping + mask
  battle_env.py   LOCMEnv / BattleEnv (gymnasium)
locma/harness/
  __init__.py
  match.py        run_match, MatchResult, per-game record
  tournament.py   run_tournament, leaderboard, win-rate matrix
  logging.py      JSONL result writer
locma/stats/
  __init__.py
  intervals.py    wilson_ci, binomial_test
  sprt.py         SPRT sequential test
  ratings.py      elo, (optional trueskill wrapper)
locma/cli/
  __init__.py
  app.py          Typer app: play, tournament, eval, train, fetch-cards, fetch-art
train.py          reference MaskablePPO training script
tests/
  test_cards_db.py test_cards.py test_instance.py test_state.py
  test_draft.py test_battle_*.py test_engine.py
  test_policies.py test_match.py test_stats.py test_tournament.py
  test_cli.py test_env.py
```

---

### Task 1: Project scaffolding + package metadata

**Files:**
- Create: `pyproject.toml`, `locma/__init__.py`, `locma/core/__init__.py`, `tests/__init__.py`, `.gitignore`

**Interfaces:**
- Produces: an installable `locma` package; `uv run pytest` runs; `[ml]` optional extra defined.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "locma"
version = "0.1.0"
description = "Legends of Code & Magic 1.2 explore kit"
requires-python = ">=3.11"
dependencies = ["typer>=0.12", "rich>=13", "scipy>=1.11", "numpy>=1.26"]

[project.optional-dependencies]
ml = ["gymnasium>=0.29", "stable-baselines3>=2.3", "sb3-contrib>=2.3", "torch>=2.2", "trueskill>=0.4.5"]
dev = ["pytest>=8"]

[project.scripts]
locma = "locma.cli.app:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package files**

`locma/__init__.py`, `locma/core/__init__.py`, `tests/__init__.py` each empty. `.gitignore`:

```
__pycache__/
*.pyc
.venv/
*.zip
locma/data/assets/*
!locma/data/assets/manifest.json
runs/
```

- [ ] **Step 3: Smoke test the package imports**

`tests/test_smoke.py`:

```python
def test_package_imports():
    import locma
    import locma.core
```

- [ ] **Step 4: Install and run**

Run: `uv sync --extra dev && uv run pytest tests/test_smoke.py -v`
Expected: PASS (1 test). If not using uv: `python -m venv .venv && .venv/Scripts/pip install -e .[dev] && .venv/Scripts/pytest tests/test_smoke.py -v`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml locma tests .gitignore
git commit -m "chore: scaffold locma package"
```

---

### Task 2: Card model

**Files:**
- Create: `locma/core/cards.py`
- Test: `tests/test_cards.py`

**Interfaces:**
- Produces:
  - `class CardType(IntEnum)`: `CREATURE=0, GREEN_ITEM=1, RED_ITEM=2, BLUE_ITEM=3`
  - `ABILITY_ORDER = "BCDGLW"`
  - `@dataclass(frozen=True) Card(id:int, name:str, type:CardType, cost:int, attack:int, defense:int, abilities:str, player_hp:int, enemy_hp:int, card_draw:int)` where `abilities` is a length-6 mask of letters/`'-'`.
  - `Card.has(self, ability:str) -> bool`
  - `normalize_abilities(raw:str) -> str` → 6-char canonical mask.

- [ ] **Step 1: Write failing test**

`tests/test_cards.py`:

```python
from locma.core.cards import Card, CardType, ABILITY_ORDER, normalize_abilities

def test_normalize_abilities_from_letters():
    assert normalize_abilities("BG") == "B--G--"
    assert normalize_abilities("------") == "------"
    assert normalize_abilities("BCDGLW") == "BCDGLW"

def test_card_has_ability():
    c = Card(1, "Test", CardType.CREATURE, 2, 3, 2, normalize_abilities("G"), 0, 0, 0)
    assert c.has("G") is True
    assert c.has("L") is False
    assert ABILITY_ORDER == "BCDGLW"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cards.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

ABILITY_ORDER = "BCDGLW"  # Breakthrough, Charge, Drain, Guard, Lethal, Ward

class CardType(IntEnum):
    CREATURE = 0
    GREEN_ITEM = 1
    RED_ITEM = 2
    BLUE_ITEM = 3

def normalize_abilities(raw: str) -> str:
    present = {ch for ch in raw if ch in ABILITY_ORDER}
    return "".join(ch if ch in present else "-" for ch in ABILITY_ORDER)

@dataclass(frozen=True)
class Card:
    id: int
    name: str
    type: CardType
    cost: int
    attack: int
    defense: int
    abilities: str  # length-6 mask over ABILITY_ORDER
    player_hp: int
    enemy_hp: int
    card_draw: int

    def has(self, ability: str) -> bool:
        return self.abilities[ABILITY_ORDER.index(ability)] != "-"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cards.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add locma/core/cards.py tests/test_cards.py
git commit -m "feat(core): add Card model and ability helpers"
```

---

### Task 3: Card database — fetch, vendor, parse

**Files:**
- Create: `locma/data/__init__.py`, `locma/data/cards_db.py`, `locma/data/cardlist.txt` (vendored)
- Test: `tests/test_cards_db.py`

**Interfaces:**
- Consumes: `Card`, `CardType`, `normalize_abilities` from Task 2.
- Produces:
  - `parse_cardlist(text:str) -> list[Card]`
  - `load_cards() -> list[Card]` (reads vendored `cardlist.txt`, asserts 160)
  - `card_by_id(cards) -> dict[int, Card]`

**IMPORTANT discovery step:** The exact column layout of the canonical `cardlist.txt` must be confirmed against the real file before finalizing the parser. The expected schema (from the LOCM/CodinGame spec) is whitespace-separated:
`id name type cost attack defense abilities player_hp enemy_hp card_draw`
where `type` is an integer `0..3` and `abilities` is a 6-char mask. If the fetched file differs (e.g. type as a word, or names with spaces), adapt `parse_cardlist` to the real columns and keep the same `Card` output. The `len==160` assertion is the guard.

- [ ] **Step 1: Fetch and vendor the real cardlist**

Run (try in order, keep the first that yields a 160-line file):

```bash
mkdir -p locma/data
curl -fsSL https://raw.githubusercontent.com/acatai/Legends-of-Code-and-Magic/master/referee1.5-java/src/main/resources/cardlist.txt -o locma/data/cardlist.txt || \
curl -fsSL https://raw.githubusercontent.com/ronaldosvieira/gym-locm/master/gym_locm/engine/resources/cardlist.txt -o locma/data/cardlist.txt
wc -l locma/data/cardlist.txt
head -3 locma/data/cardlist.txt
```

Confirm the file has 160 card lines and inspect the first 3 lines to lock the column mapping used in Step 3. If neither URL works, record the working source URL in the file header comment and in spec §10.

- [ ] **Step 2: Write failing test**

`tests/test_cards_db.py`:

```python
from locma.core.cards import CardType
from locma.data.cards_db import load_cards, parse_cardlist, card_by_id

def test_load_cards_count():
    cards = load_cards()
    assert len(cards) == 160
    assert all(1 <= c.id <= 160 for c in cards)

def test_card_by_id_unique():
    cards = load_cards()
    by_id = card_by_id(cards)
    assert len(by_id) == 160

def test_parse_one_line():
    # Adjust this literal to a real line copied from cardlist.txt during Step 1.
    line = "1 Slime 0 0 1 1 ------ 0 0 0"
    cards = parse_cardlist(line)
    assert len(cards) == 1
    c = cards[0]
    assert c.id == 1
    assert c.type in set(CardType)
    assert len(c.abilities) == 6
```

- [ ] **Step 3: Implement parser**

`locma/data/cards_db.py` (map columns to the layout confirmed in Step 1; below assumes `id name type cost atk def abilities pHP eHP draw`):

```python
from __future__ import annotations
from importlib import resources
from locma.core.cards import Card, CardType, normalize_abilities

def parse_cardlist(text: str) -> list[Card]:
    cards: list[Card] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Fixed layout: id name type cost atk def abilities pHP eHP draw
        cid = int(parts[0]); name = parts[1]
        ctype = CardType(int(parts[2]))
        cost, atk, dfn = int(parts[3]), int(parts[4]), int(parts[5])
        abilities = normalize_abilities(parts[6])
        php, ehp, draw = int(parts[7]), int(parts[8]), int(parts[9])
        cards.append(Card(cid, name, ctype, cost, atk, dfn, abilities, php, ehp, draw))
    return cards

def load_cards() -> list[Card]:
    text = resources.files("locma.data").joinpath("cardlist.txt").read_text(encoding="utf-8")
    cards = parse_cardlist(text)
    assert len(cards) == 160, f"expected 160 cards, got {len(cards)}"
    return cards

def card_by_id(cards: list[Card]) -> dict[int, Card]:
    return {c.id: c for c in cards}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cards_db.py -v` → PASS. (If `test_parse_one_line` fails, the assumed column order is wrong — fix `parse_cardlist` to the real layout from Step 1 and update the test literal.)

- [ ] **Step 5: Commit**

```bash
git add locma/data tests/test_cards_db.py
git commit -m "feat(data): vendor cardlist.txt and add parser/loader"
```

---

### Task 4: CardInstance (runtime wrapper)

**Files:**
- Create: `locma/core/instance.py`
- Test: `tests/test_instance.py`

**Interfaces:**
- Consumes: `Card` from Task 2.
- Produces:
  - `@dataclass CardInstance(card:Card, instance_id:int, attack:int, defense:int, abilities:str, can_attack:bool, has_attacked:bool)`
  - `CardInstance.from_card(card:Card, instance_id:int) -> CardInstance` — copies stats; `can_attack=True` iff card has Charge `'C'`.
  - `CardInstance.has(ability:str) -> bool` (reads its own mutable `abilities`).

- [ ] **Step 1: Write failing test**

`tests/test_instance.py`:

```python
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance

def _creature(abilities=""):
    return Card(1, "C", CardType.CREATURE, 2, 3, 2, normalize_abilities(abilities), 0, 0, 0)

def test_from_card_copies_stats():
    inst = CardInstance.from_card(_creature(), instance_id=7)
    assert (inst.attack, inst.defense, inst.instance_id) == (3, 2, 7)
    assert inst.can_attack is False
    assert inst.has_attacked is False

def test_charge_can_attack_immediately():
    inst = CardInstance.from_card(_creature("C"), instance_id=1)
    assert inst.can_attack is True

def test_instance_has_reads_mutable_abilities():
    inst = CardInstance.from_card(_creature("G"), instance_id=1)
    assert inst.has("G") is True
```

- [ ] **Step 2: Run → FAIL.** `uv run pytest tests/test_instance.py -v`

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
from dataclasses import dataclass
from locma.core.cards import Card, ABILITY_ORDER

@dataclass
class CardInstance:
    card: Card
    instance_id: int
    attack: int
    defense: int
    abilities: str
    can_attack: bool = False
    has_attacked: bool = False

    @classmethod
    def from_card(cls, card: Card, instance_id: int) -> "CardInstance":
        return cls(card=card, instance_id=instance_id,
                   attack=card.attack, defense=card.defense,
                   abilities=card.abilities,
                   can_attack=card.has("C"))

    def has(self, ability: str) -> bool:
        return self.abilities[ABILITY_ORDER.index(ability)] != "-"
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit**

```bash
git add locma/core/instance.py tests/test_instance.py
git commit -m "feat(core): add CardInstance runtime wrapper"
```

---

### Task 5: Game state + actions

**Files:**
- Create: `locma/core/state.py`, `locma/core/actions.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `CardInstance`.
- Produces (`actions.py`):
  - `@dataclass(frozen=True) Summon(card_instance_id:int)`
  - `@dataclass(frozen=True) Attack(attacker_id:int, target_id:int)` where `target_id == -1` means face.
  - `@dataclass(frozen=True) Use(item_instance_id:int, target_id:int)` (`target_id == -1` = face/no target)
  - `@dataclass(frozen=True) Pass()`
  - `Action = Summon | Attack | Use | Pass`
- Produces (`state.py`):
  - `class Phase(Enum): DRAFT, BATTLE, ENDED`
  - `@dataclass PlayerState(health=30, mana=0, max_mana=0, next_rune=25, bonus_draw=0, deck:list, hand:list, board:list)`
  - `@dataclass GameState(rng:random.Random, phase:Phase, turn:int, current:int (0/1), players:tuple[PlayerState,PlayerState], draft_pool:list, draft_round:int, picks:tuple[list,list], winner:int|None)`
  - `GameState.opponent(player_idx) -> int`
  - `GameState.clone() -> GameState` (deep copy of players/board; RNG state preserved).

- [ ] **Step 1: Write failing test**

`tests/test_state.py`:

```python
import random
from locma.core.state import GameState, PlayerState, Phase
from locma.core.actions import Summon, Attack, Use, Pass

def test_player_defaults():
    p = PlayerState()
    assert p.health == 30 and p.mana == 0 and p.next_rune == 25
    assert p.deck == [] and p.hand == [] and p.board == []

def test_actions_face_sentinel():
    a = Attack(attacker_id=5, target_id=-1)
    assert a.target_id == -1
    assert isinstance(Pass(), Pass)

def test_clone_is_independent():
    gs = GameState.new(random.Random(0))
    gs2 = gs.clone()
    gs2.players[0].health = 1
    assert gs.players[0].health == 30
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `actions.py`**

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Summon:
    card_instance_id: int

@dataclass(frozen=True)
class Attack:
    attacker_id: int
    target_id: int  # -1 = face

@dataclass(frozen=True)
class Use:
    item_instance_id: int
    target_id: int  # -1 = face/no target

@dataclass(frozen=True)
class Pass:
    pass

Action = Summon | Attack | Use | Pass
```

- [ ] **Step 4: Implement `state.py`**

```python
from __future__ import annotations
import copy, random
from dataclasses import dataclass, field
from enum import Enum, auto

class Phase(Enum):
    DRAFT = auto()
    BATTLE = auto()
    ENDED = auto()

@dataclass
class PlayerState:
    health: int = 30
    mana: int = 0
    max_mana: int = 0
    next_rune: int = 25
    bonus_draw: int = 0
    deck: list = field(default_factory=list)
    hand: list = field(default_factory=list)
    board: list = field(default_factory=list)

@dataclass
class GameState:
    rng: random.Random
    phase: Phase = Phase.DRAFT
    turn: int = 0
    current: int = 0
    players: tuple = None
    draft_pool: list = field(default_factory=list)
    draft_round: int = 0
    picks: tuple = None
    winner: int | None = None

    @classmethod
    def new(cls, rng: random.Random) -> "GameState":
        gs = cls(rng=rng)
        gs.players = (PlayerState(), PlayerState())
        gs.picks = ([], [])
        return gs

    def opponent(self, player_idx: int) -> int:
        return 1 - player_idx

    def clone(self) -> "GameState":
        new_rng = random.Random()
        new_rng.setstate(self.rng.getstate())
        return GameState(
            rng=new_rng, phase=self.phase, turn=self.turn, current=self.current,
            players=copy.deepcopy(self.players), draft_pool=list(self.draft_pool),
            draft_round=self.draft_round, picks=copy.deepcopy(self.picks),
            winner=self.winner,
        )
```

- [ ] **Step 5: Run → PASS. Commit.**

```bash
git add locma/core/state.py locma/core/actions.py tests/test_state.py
git commit -m "feat(core): add GameState, PlayerState, and Action types"
```

---

### Task 6: Draft phase

**Files:**
- Create: `locma/core/draft.py`
- Test: `tests/test_draft.py`

**Interfaces:**
- Consumes: `load_cards`, `GameState`, `Phase`, `CardInstance`.
- Produces:
  - `start_draft(gs:GameState, cards:list[Card], rounds:int=30) -> None` — builds `gs.draft_pool` as a list of `rounds` triplets (each a list of 3 `Card`), using `gs.rng`. Both players see the same triplets (mirrored).
  - `draft_legal(gs) -> list[int]` → `[0,1,2]` while drafting.
  - `current_triplet(gs) -> list[Card]`
  - `apply_draft_pick(gs, pick:int) -> None` — records the chosen `Card` for the player whose turn it is into `gs.picks[player]`; advances. After both players pick in a round, advance `draft_round`. After all rounds, build each player's `deck` of `CardInstance`s (unique incrementing `instance_id`), shuffle each deck with `gs.rng`, set `gs.phase = Phase.BATTLE`, and deal opening hands (see Task 8 hand-out; draft just leaves decks ready).

- [ ] **Step 1: Write failing test**

`tests/test_draft.py`:

```python
import random
from locma.core.state import GameState, Phase
from locma.core.draft import start_draft, draft_legal, apply_draft_pick, current_triplet
from locma.data.cards_db import load_cards

def test_draft_runs_30_rounds_and_builds_decks():
    gs = GameState.new(random.Random(42))
    start_draft(gs, load_cards(), rounds=30)
    assert len(gs.draft_pool) == 30
    assert all(len(t) == 3 for t in gs.draft_pool)
    for _ in range(60):  # 30 rounds * 2 players
        assert draft_legal(gs) == [0, 1, 2]
        assert len(current_triplet(gs)) == 3
        apply_draft_pick(gs, 0)
    assert gs.phase == Phase.BATTLE
    assert len(gs.picks[0]) == 30 and len(gs.picks[1]) == 30
    assert len(gs.players[0].deck) == 30 and len(gs.players[1].deck) == 30

def test_draft_is_deterministic():
    a = GameState.new(random.Random(7)); start_draft(a, load_cards())
    b = GameState.new(random.Random(7)); start_draft(b, load_cards())
    assert [[c.id for c in t] for t in a.draft_pool] == [[c.id for c in t] for t in b.draft_pool]
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
from locma.core.cards import Card
from locma.core.state import GameState, Phase
from locma.core.instance import CardInstance

def start_draft(gs: GameState, cards: list[Card], rounds: int = 30) -> None:
    pool = []
    for _ in range(rounds):
        triplet = [cards[gs.rng.randint(0, len(cards) - 1)] for _ in range(3)]
        pool.append(triplet)
    gs.draft_pool = pool
    gs.draft_round = 0
    gs.current = 0
    gs.phase = Phase.DRAFT

def draft_legal(gs: GameState) -> list[int]:
    return [0, 1, 2]

def current_triplet(gs: GameState) -> list[Card]:
    return gs.draft_pool[gs.draft_round]

def apply_draft_pick(gs: GameState, pick: int) -> None:
    player = gs.current
    chosen = gs.draft_pool[gs.draft_round][pick]
    gs.picks[player].append(chosen)
    if gs.current == 0:
        gs.current = 1
    else:
        gs.current = 0
        gs.draft_round += 1
        if gs.draft_round >= len(gs.draft_pool):
            _finish_draft(gs)

def _finish_draft(gs: GameState) -> None:
    iid = 0
    for p in (0, 1):
        deck = []
        for card in gs.picks[p]:
            deck.append(CardInstance.from_card(card, iid)); iid += 1
        gs.rng.shuffle(deck)
        gs.players[p].deck = deck
    gs.phase = Phase.BATTLE
    gs.draft_round = 0
    gs.current = 0
```

- [ ] **Step 4: Run → PASS. Commit.**

```bash
git add locma/core/draft.py tests/test_draft.py
git commit -m "feat(core): add draft phase"
```

---

### Task 7: Battle — mana, turn structure, draw, win check (no combat yet)

**Files:**
- Create: `locma/core/battle.py`
- Test: `tests/test_battle_turn.py`

**Interfaces:**
- Consumes: `GameState`, `PlayerState`, `Phase`, `CardInstance`, actions.
- Produces:
  - `start_battle(gs) -> None` — both players draw opening hands (player 0: 4 cards, player 1: 5 cards + gets a "rune"/extra is not modeled; keep it 4 and 5 per 1.2 second-player compensation), set `current=0`, `turn=1`, ramp first player to `max_mana=1, mana=1`.
  - `draw(gs, player, n) -> None` — move top `n` from deck to hand (deck-out sets a pending loss flag via rune damage: on empty deck draw, that player loses `health` by escalating rune penalty; minimal model: each empty-draw deals 1 damage and may end game).
  - `start_turn(gs) -> None` — increment `max_mana` (cap 12), refill `mana`, untap board (`can_attack=True`, `has_attacked=False`), draw 1 (+ `bonus_draw`).
  - `end_turn(gs) -> None` — switch `current`, increment `turn`, call `start_turn` for new current.
  - `check_winner(gs) -> None` — if a player `health <= 0`, set `gs.winner` and `gs.phase=ENDED`.

- [ ] **Step 1: Write failing test**

`tests/test_battle_turn.py`:

```python
import random
from locma.core.state import GameState, Phase
from locma.core.draft import start_draft, apply_draft_pick
from locma.core.battle import start_battle, start_turn, end_turn, check_winner
from locma.data.cards_db import load_cards

def _drafted():
    gs = GameState.new(random.Random(1)); start_draft(gs, load_cards())
    for _ in range(60): apply_draft_pick(gs, 0)
    return gs

def test_start_battle_deals_hands_and_mana():
    gs = _drafted(); start_battle(gs)
    assert len(gs.players[0].hand) == 4
    assert len(gs.players[1].hand) == 5
    assert gs.players[0].max_mana == 1 and gs.players[0].mana == 1

def test_turn_ramps_mana_and_untaps():
    gs = _drafted(); start_battle(gs)
    end_turn(gs)               # to player 1
    assert gs.current == 1
    assert gs.players[1].max_mana == 1
    end_turn(gs)               # back to player 0, turn 3
    assert gs.players[0].max_mana == 2

def test_check_winner():
    gs = _drafted(); start_battle(gs)
    gs.players[1].health = 0
    check_winner(gs)
    assert gs.winner == 0 and gs.phase == Phase.ENDED
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
from locma.core.state import GameState, Phase

MAX_MANA = 12

def draw(gs: GameState, player: int, n: int) -> None:
    p = gs.players[player]
    for _ in range(n):
        if p.deck:
            p.hand.append(p.deck.pop(0))
            if len(p.hand) > 8:
                p.hand.pop()  # overdraw burns the card (1.2 hand cap 8)
        else:
            # deck-out rune penalty: escalating self damage
            p.health -= 1

def start_turn(gs: GameState) -> None:
    p = gs.players[gs.current]
    p.max_mana = min(MAX_MANA, p.max_mana + 1)
    p.mana = p.max_mana
    for c in p.board:
        c.can_attack = True
        c.has_attacked = False
    draw(gs, gs.current, 1 + p.bonus_draw)
    p.bonus_draw = 0

def start_battle(gs: GameState) -> None:
    gs.phase = Phase.BATTLE
    gs.turn = 1
    gs.current = 0
    draw(gs, 0, 4)
    draw(gs, 1, 5)
    p = gs.players[0]
    p.max_mana = 1
    p.mana = 1

def end_turn(gs: GameState) -> None:
    gs.current = gs.opponent(gs.current)
    gs.turn += 1
    start_turn(gs)

def check_winner(gs: GameState) -> None:
    for idx in (0, 1):
        if gs.players[idx].health <= 0:
            gs.winner = gs.opponent(idx)
            gs.phase = Phase.ENDED
            return
```

- [ ] **Step 4: Run → PASS. Commit.**

```bash
git add locma/core/battle.py tests/test_battle_turn.py
git commit -m "feat(core): add battle turn structure, mana, draw, win check"
```

---

### Task 8: Battle — legal actions + summon + item use + pass

**Files:**
- Modify: `locma/core/battle.py`
- Test: `tests/test_battle_actions.py`

**Interfaces:**
- Consumes: actions `Summon, Use, Pass, Attack`, `GameState`.
- Produces:
  - `battle_legal(gs) -> list[Action]` — for `current` player: `Pass()` always; `Summon(iid)` for each creature in hand with `cost <= mana` and board has `< 6` creatures; `Use(item_iid, target)` for each item in hand affordable, with legal targets (green → own creatures; red → enemy creatures; blue → enemy creatures or face `-1`); `Attack(attacker_iid, target)` for each own board creature with `can_attack` and not `has_attacked`, target = enemy Guard creatures only if any Guard exists else any enemy creature or face `-1`.
  - `apply_battle(gs, action) -> None` — mutates state, spends mana, resolves; calls `check_winner`. (Combat math lives in Task 9; here implement Summon, Use for the simplest item application via a shared `_apply_item`, and Pass = `end_turn`.)

- [ ] **Step 1: Write failing test**

`tests/test_battle_actions.py`:

```python
import random
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.core.battle import battle_legal, apply_battle
from locma.core.actions import Summon, Pass, Attack

def _bare_battle():
    gs = GameState.new(random.Random(0)); gs.phase = Phase.BATTLE; gs.turn = 1; gs.current = 0
    gs.players[0].mana = 5; gs.players[0].max_mana = 5
    return gs

def _creature(iid, cost=2, atk=3, dfn=2, ab=""):
    c = Card(100, "X", CardType.CREATURE, cost, atk, dfn, normalize_abilities(ab), 0, 0, 0)
    return CardInstance.from_card(c, iid)

def test_summon_moves_card_to_board_and_spends_mana():
    gs = _bare_battle()
    inst = _creature(1, cost=3)
    gs.players[0].hand.append(inst)
    legal = battle_legal(gs)
    assert any(isinstance(a, Summon) and a.card_instance_id == 1 for a in legal)
    apply_battle(gs, Summon(1))
    assert gs.players[0].mana == 2
    assert len(gs.players[0].board) == 1 and not gs.players[0].hand

def test_pass_ends_turn():
    gs = _bare_battle()
    apply_battle(gs, Pass())
    assert gs.current == 1

def test_guard_restricts_attack_targets():
    gs = _bare_battle()
    atk = _creature(1); atk.can_attack = True
    gs.players[0].board.append(atk)
    guard = _creature(2, ab="G"); plain = _creature(3)
    gs.players[1].board.extend([guard, plain])
    targets = [a.target_id for a in battle_legal(gs) if isinstance(a, Attack) and a.attacker_id == 1]
    assert 2 in targets and 3 not in targets and -1 not in targets
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to `battle.py`)**

```python
from locma.core.actions import Summon, Attack, Use, Pass, Action
from locma.core.cards import CardType

def _find_in_hand(p, iid):
    for c in p.hand:
        if c.instance_id == iid:
            return c
    return None

def _enemy_guards(opp):
    return [c for c in opp.board if c.has("G")]

def battle_legal(gs: GameState) -> list[Action]:
    p = gs.players[gs.current]
    opp = gs.players[gs.opponent(gs.current)]
    actions: list[Action] = [Pass()]
    for c in p.hand:
        if c.card.cost > p.mana:
            continue
        if c.card.type == CardType.CREATURE:
            if len(p.board) < 6:
                actions.append(Summon(c.instance_id))
        elif c.card.type == CardType.GREEN_ITEM:
            for t in p.board:
                actions.append(Use(c.instance_id, t.instance_id))
        elif c.card.type == CardType.RED_ITEM:
            for t in opp.board:
                actions.append(Use(c.instance_id, t.instance_id))
        else:  # BLUE_ITEM
            for t in opp.board:
                actions.append(Use(c.instance_id, t.instance_id))
            actions.append(Use(c.instance_id, -1))
    guards = _enemy_guards(opp)
    targets = guards if guards else opp.board
    for c in p.board:
        if c.can_attack and not c.has_attacked:
            for t in targets:
                actions.append(Attack(c.instance_id, t.instance_id))
            if not guards:
                actions.append(Attack(c.instance_id, -1))
    return actions

def apply_battle(gs: GameState, action: Action) -> None:
    p = gs.players[gs.current]
    if isinstance(action, Pass):
        end_turn(gs); return
    if isinstance(action, Summon):
        c = _find_in_hand(p, action.card_instance_id)
        p.hand.remove(c); p.mana -= c.card.cost
        p.board.append(c)
        _trigger_summon_effects(gs, gs.current, c)
    elif isinstance(action, Use):
        c = _find_in_hand(p, action.item_instance_id)
        p.hand.remove(c); p.mana -= c.card.cost
        _apply_item(gs, c, action.target_id)
    elif isinstance(action, Attack):
        _resolve_attack(gs, action.attacker_id, action.target_id)  # Task 9
    check_winner(gs)

def _trigger_summon_effects(gs, player, c):
    p = gs.players[player]; opp = gs.players[gs.opponent(player)]
    p.health += c.card.player_hp
    opp.health += c.card.enemy_hp
    p.bonus_draw += c.card.card_draw

def _apply_item(gs, item, target_id):
    p = gs.players[gs.current]; opp = gs.players[gs.opponent(gs.current)]
    t = item.card.type
    if t == CardType.GREEN_ITEM:
        tgt = _find_on_board(p, target_id)
        if tgt:
            tgt.attack = max(0, tgt.attack + item.card.attack)
            tgt.defense += item.card.defense
            tgt.abilities = _merge_abilities(tgt.abilities, item.card.abilities, add=True)
        p.health += item.card.player_hp; opp.health += item.card.enemy_hp
        p.bonus_draw += item.card.card_draw
    elif t == CardType.RED_ITEM:
        tgt = _find_on_board(opp, target_id)
        if tgt:
            tgt.attack = max(0, tgt.attack + item.card.attack)
            tgt.abilities = _merge_abilities(tgt.abilities, item.card.abilities, add=False)
            tgt.defense += item.card.defense
            if tgt.defense <= 0:
                opp.board.remove(tgt)
        p.health += item.card.player_hp; opp.health += item.card.enemy_hp
    else:  # BLUE_ITEM
        if target_id == -1:
            opp.health += item.card.defense  # blue items carry negative defense as damage
        else:
            tgt = _find_on_board(opp, target_id)
            if tgt:
                tgt.defense += item.card.defense
                if tgt.defense <= 0:
                    opp.board.remove(tgt)
        p.health += item.card.player_hp; opp.health += item.card.enemy_hp

def _find_on_board(p, iid):
    for c in p.board:
        if c.instance_id == iid:
            return c
    return None

def _merge_abilities(base, mod, add):
    from locma.core.cards import ABILITY_ORDER
    out = []
    for i, ch in enumerate(ABILITY_ORDER):
        present = base[i] != "-"
        changed = mod[i] != "-"
        if add and changed:
            present = True
        if (not add) and changed:
            present = False
        out.append(ch if present else "-")
    return "".join(out)
```

Add helper `_find_on_board` used above; ensure `_resolve_attack` is defined in Task 9 before running attack tests (the summon/pass/guard tests here do not call it).

- [ ] **Step 4: Run → PASS** for summon/pass/guard tests. `uv run pytest tests/test_battle_actions.py -v`

- [ ] **Step 5: Commit**

```bash
git add locma/core/battle.py tests/test_battle_actions.py
git commit -m "feat(core): battle legal actions, summon, item use, pass"
```

---

### Task 9: Battle — combat resolution with keywords

**Files:**
- Modify: `locma/core/battle.py`
- Test: `tests/test_battle_combat.py`

**Interfaces:**
- Produces: `_resolve_attack(gs, attacker_id, target_id) -> None` implementing LOCM 1.2 combat:
  - Find attacker on current board; mark `has_attacked=True`, `can_attack=False`.
  - **Face attack** (`target_id == -1`): deal `attacker.attack` to opponent health; if attacker has Drain `'D'`, heal controller by that amount.
  - **Creature attack:** simultaneous damage. Apply attacker damage to defender, defender damage to attacker, honoring:
    - **Ward `'W'`:** if a unit has Ward, it takes no damage from this instance and loses Ward (`abilities` cleared at that position); damage "fizzles".
    - **Lethal `'L'`:** any nonzero damage from a Lethal unit destroys the target (set defense ≤ 0), unless target warded.
    - **Drain `'D'`:** controller heals by damage actually dealt to the target's defense (not warded amount).
    - **Breakthrough `'B'`:** excess attacker damage beyond defender's defense carries to opponent face.
  - Remove any creature with `defense <= 0` from its board.
  - Call `check_winner`.

- [ ] **Step 1: Write failing tests**

`tests/test_battle_combat.py`:

```python
import random
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.core.battle import _resolve_attack

def _gs():
    gs = GameState.new(random.Random(0)); gs.phase = Phase.BATTLE; gs.current = 0
    return gs

def _c(iid, atk, dfn, ab=""):
    card = Card(1, "X", CardType.CREATURE, 1, atk, dfn, normalize_abilities(ab), 0, 0, 0)
    inst = CardInstance.from_card(card, iid); inst.can_attack = True
    return inst

def test_trade_kills_both():
    gs = _gs()
    a = _c(1, 3, 2); d = _c(2, 2, 2)
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[0].board == [] and gs.players[1].board == []

def test_ward_absorbs_then_drops():
    gs = _gs()
    a = _c(1, 3, 2); d = _c(2, 1, 3, ab="W")
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert d.defense == 3 and d.has("W") is False  # warded, ward consumed
    assert a.defense == 1                            # attacker still took 1

def test_lethal_destroys_big():
    gs = _gs()
    a = _c(1, 1, 5, ab="L"); d = _c(2, 0, 9)
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[1].board == []

def test_breakthrough_tramples_face():
    gs = _gs()
    a = _c(1, 5, 5, ab="B"); d = _c(2, 0, 2)
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[1].health == 30 - 3   # 5 - 2 defense = 3 trample

def test_drain_heals_on_face():
    gs = _gs(); gs.players[0].health = 20
    a = _c(1, 4, 2, ab="D")
    gs.players[0].board.append(a)
    _resolve_attack(gs, 1, -1)
    assert gs.players[1].health == 30 - 4 and gs.players[0].health == 24
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to `battle.py`)**

```python
def _clear_ward(unit):
    from locma.core.cards import ABILITY_ORDER
    i = ABILITY_ORDER.index("W")
    unit.abilities = unit.abilities[:i] + "-" + unit.abilities[i+1:]

def _deal_to_unit(unit, amount, lethal):
    """Returns damage actually applied to defense (0 if warded)."""
    if amount <= 0:
        return 0
    if unit.has("W"):
        _clear_ward(unit)
        return 0
    if lethal:
        unit.defense = 0
        return amount
    unit.defense -= amount
    return amount

def _resolve_attack(gs: GameState, attacker_id: int, target_id: int) -> None:
    p = gs.players[gs.current]; opp = gs.players[gs.opponent(gs.current)]
    atk = _find_on_board(p, attacker_id)
    if atk is None:
        return
    atk.has_attacked = True
    atk.can_attack = False
    if target_id == -1:
        dmg = atk.attack
        opp.health -= dmg
        if atk.has("D") and dmg > 0:
            p.health += dmg
        check_winner(gs)
        return
    dfn = _find_on_board(opp, target_id)
    if dfn is None:
        return
    applied = _deal_to_unit(dfn, atk.attack, atk.has("L"))
    if atk.has("D") and applied > 0:
        p.health += applied
    if atk.has("B") and not dfn.has("W"):
        overflow = atk.attack - max(0, dfn_defense_before(dfn, applied))
    # Breakthrough: excess over the defender's pre-hit defense goes face
    # (recompute cleanly below)
    _deal_to_unit(atk, dfn.attack, dfn.has("L"))
    # remove dead
    if dfn.defense <= 0 and dfn in opp.board:
        opp.board.remove(dfn)
    if atk.defense <= 0 and atk in p.board:
        p.board.remove(atk)
    check_winner(gs)
```

Replace the muddled Breakthrough block with this clean version of the whole creature-vs-creature path (use this as the authoritative body):

```python
def _resolve_attack(gs: GameState, attacker_id: int, target_id: int) -> None:
    p = gs.players[gs.current]; opp = gs.players[gs.opponent(gs.current)]
    atk = _find_on_board(p, attacker_id)
    if atk is None:
        return
    atk.has_attacked = True; atk.can_attack = False
    if target_id == -1:
        dmg = atk.attack
        opp.health -= dmg
        if atk.has("D") and dmg > 0:
            p.health += dmg
        check_winner(gs); return
    dfn = _find_on_board(opp, target_id)
    if dfn is None:
        return
    warded = dfn.has("W")
    def_before = dfn.defense
    applied = _deal_to_unit(dfn, atk.attack, atk.has("L"))  # consumes ward if present
    if atk.has("D") and applied > 0:
        p.health += applied
    if atk.has("B") and not warded:
        overflow = atk.attack - max(0, def_before)
        if overflow > 0:
            opp.health -= overflow
    _deal_to_unit(atk, dfn.attack, dfn.has("L"))
    if dfn.defense <= 0 and dfn in opp.board:
        opp.board.remove(dfn)
    if atk.defense <= 0 and atk in p.board:
        p.board.remove(atk)
    check_winner(gs)
```

Delete the first draft of `_resolve_attack` and the stray `dfn_defense_before` reference; keep only the authoritative body.

- [ ] **Step 4: Run → PASS.** `uv run pytest tests/test_battle_combat.py -v`

- [ ] **Step 5: Commit**

```bash
git add locma/core/battle.py tests/test_battle_combat.py
git commit -m "feat(core): combat resolution with B/C/D/G/L/W keywords"
```

---

### Task 10: Views + engine orchestration

**Files:**
- Create: `locma/core/views.py`, `locma/core/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: everything in `core`, plus `Policy` (Task 11). To avoid a cycle, `engine.run_game` takes any object with `draft_action` and `battle_action` callables (duck-typed).
- Produces:
  - `views.py`: `DraftView` (own picks-so-far counts, the 3 offered cards), `BattleView` (own hp/mana/board/hand, opponent hp/mana/board, opponent hand **count only**, turn). Pure data, no `CardInstance` mutation refs — copy primitives + card ids/stats.
  - `engine.py`:
    - `make_draft_view(gs) -> DraftView`
    - `make_battle_view(gs) -> BattleView`
    - `run_game(policy0, policy1, seed:int, cards=None, max_turns:int=200) -> GameResult` where `GameResult(winner:int, turns:int, seed:int)`. Drives draft then battle to completion; on `max_turns` exceeded, winner = higher health (tiebreak → player 0). Guards against infinite loops by capping battle actions per turn.

- [ ] **Step 1: Write failing test**

`tests/test_engine.py`:

```python
from locma.core.engine import run_game
from locma.policies.random_policy import RandomPolicy

def test_run_game_returns_winner():
    r = run_game(RandomPolicy("a"), RandomPolicy("b"), seed=123)
    assert r.winner in (0, 1)
    assert r.turns >= 1

def test_run_game_is_deterministic():
    r1 = run_game(RandomPolicy("a"), RandomPolicy("b"), seed=999)
    r2 = run_game(RandomPolicy("a"), RandomPolicy("b"), seed=999)
    assert (r1.winner, r1.turns) == (r2.winner, r2.turns)
```

(Depends on Task 11 RandomPolicy — implement Task 11 first or stub a uniform policy inline. Build order: do Task 11 before running this.)

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `views.py`**

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CardView:
    instance_id: int
    card_id: int
    type: int
    cost: int
    attack: int
    defense: int
    abilities: str

@dataclass(frozen=True)
class DraftView:
    round: int
    offered: tuple  # 3 CardView (instance_id unused, -1)

@dataclass(frozen=True)
class BattleView:
    turn: int
    me_health: int
    me_mana: int
    op_health: int
    op_hand_count: int
    my_hand: tuple
    my_board: tuple
    op_board: tuple
```

- [ ] **Step 4: Implement `engine.py`**

```python
from __future__ import annotations
import random
from locma.core.state import GameState, Phase
from locma.core import draft as draftmod
from locma.core import battle as battlemod
from locma.core.views import DraftView, BattleView, CardView
from locma.data.cards_db import load_cards
from dataclasses import dataclass

@dataclass(frozen=True)
class GameResult:
    winner: int
    turns: int
    seed: int

def _cv(inst, hide_id=False):
    return CardView(inst.instance_id if not hide_id else -1, inst.card.id,
                    int(inst.card.type), inst.card.cost, inst.attack, inst.defense, inst.abilities)

def make_draft_view(gs):
    offered = tuple(CardView(-1, c.id, int(c.type), c.cost, c.attack, c.defense, c.abilities)
                    for c in gs.draft_pool[gs.draft_round])
    return DraftView(gs.draft_round, offered)

def make_battle_view(gs):
    me = gs.players[gs.current]; op = gs.players[gs.opponent(gs.current)]
    return BattleView(gs.turn, me.health, me.mana, op.health, len(op.hand),
                      tuple(_cv(c) for c in me.hand),
                      tuple(_cv(c) for c in me.board),
                      tuple(_cv(c) for c in op.board))

def run_game(policy0, policy1, seed, cards=None, max_turns=200) -> GameResult:
    cards = cards or load_cards()
    gs = GameState.new(random.Random(seed))
    draftmod.start_draft(gs, cards)
    pols = (policy0, policy1)
    while gs.phase == Phase.DRAFT:
        view = make_draft_view(gs)
        pick = pols[gs.current].draft_action(view, [0, 1, 2])
        draftmod.apply_draft_pick(gs, pick)
    battlemod.start_battle(gs)
    safety = 0
    while gs.phase == Phase.BATTLE and gs.turn <= max_turns:
        per_turn = 0
        turn_owner = gs.current
        while gs.current == turn_owner and gs.phase == Phase.BATTLE:
            legal = battlemod.battle_legal(gs)
            view = make_battle_view(gs)
            action = pols[gs.current].battle_action(view, legal)
            battlemod.apply_battle(gs, action)
            per_turn += 1
            if per_turn > 100:
                battlemod.end_turn(gs); break
        safety += 1
        if safety > 1000:
            break
    if gs.winner is None:
        h0, h1 = gs.players[0].health, gs.players[1].health
        gs.winner = 0 if h0 >= h1 else 1
    return GameResult(gs.winner, gs.turn, seed)
```

- [ ] **Step 5: Run → PASS (after Task 11). Commit.**

```bash
git add locma/core/views.py locma/core/engine.py tests/test_engine.py
git commit -m "feat(core): views + run_game orchestration"
```

---

### Task 11: Policy protocol + Random + Scripted

**Files:**
- Create: `locma/policies/__init__.py`, `locma/policies/base.py`, `locma/policies/random_policy.py`, `locma/policies/scripted.py`
- Test: `tests/test_policies.py`

**Interfaces:**
- Consumes: views, actions.
- Produces:
  - `base.py`: `class Policy(Protocol)` with `name:str`, `draft_action(view, legal)->int`, `battle_action(view, legal)->Action`, `reset(seed=None)->None`. Plus `class CompositePolicy` taking `draft` and `battle` sub-policies and delegating.
  - `random_policy.py`: `class RandomPolicy` — `__init__(self, name, seed=0)`; uniform choice over legal using its own `random.Random`.
  - `scripted.py`: `class ScriptedPolicy` — draft always picks `0`; battle picks the first non-Pass legal action if any creature can be summoned/attack, else `Pass()` (deterministic lower bound).

- [ ] **Step 1: Write failing test**

`tests/test_policies.py`:

```python
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy
from locma.core.actions import Pass

def test_random_draft_in_range():
    p = RandomPolicy("r", seed=1)
    assert p.draft_action(None, [0, 1, 2]) in (0, 1, 2)

def test_random_battle_returns_legal():
    p = RandomPolicy("r", seed=1)
    legal = [Pass()]
    assert p.battle_action(None, legal) in legal

def test_scripted_draft_picks_zero():
    assert ScriptedPolicy("s").draft_action(None, [0, 1, 2]) == 0
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

`base.py`:

```python
from __future__ import annotations
from typing import Protocol

class Policy(Protocol):
    name: str
    def draft_action(self, view, legal: list[int]) -> int: ...
    def battle_action(self, view, legal: list): ...
    def reset(self, seed: int | None = None) -> None: ...

class CompositePolicy:
    def __init__(self, draft, battle, name=None):
        self.draft = draft; self.battle = battle
        self.name = name or f"{draft.name}+{battle.name}"
    def draft_action(self, view, legal): return self.draft.draft_action(view, legal)
    def battle_action(self, view, legal): return self.battle.battle_action(view, legal)
    def reset(self, seed=None):
        self.draft.reset(seed); self.battle.reset(seed)
```

`random_policy.py`:

```python
from __future__ import annotations
import random

class RandomPolicy:
    def __init__(self, name: str = "random", seed: int = 0):
        self.name = name; self._seed = seed; self._r = random.Random(seed)
    def draft_action(self, view, legal): return self._r.choice(legal)
    def battle_action(self, view, legal): return self._r.choice(legal)
    def reset(self, seed=None):
        self._r = random.Random(self._seed if seed is None else seed)
```

`scripted.py`:

```python
from __future__ import annotations
from locma.core.actions import Pass

class ScriptedPolicy:
    def __init__(self, name: str = "scripted"):
        self.name = name
    def draft_action(self, view, legal): return legal[0]
    def battle_action(self, view, legal):
        for a in legal:
            if not isinstance(a, Pass):
                return a
        return legal[0]
    def reset(self, seed=None): pass
```

- [ ] **Step 4: Run → PASS.** Then run Task 10's `tests/test_engine.py` → PASS.

- [ ] **Step 5: Commit**

```bash
git add locma/policies tests/test_policies.py
git commit -m "feat(policies): Policy protocol, Random, Scripted, Composite"
```

---

### Task 12: Greedy policy

**Files:**
- Create: `locma/policies/greedy.py`
- Test: `tests/test_greedy.py`

**Interfaces:**
- Produces: `class GreedyPolicy` —
  - Draft: pick the offered card with the best heuristic score `score = attack + defense + 0.5*keyword_count - 0.7*max(0,cost-?)`; prefer creatures; concrete formula below.
  - Battle: priority order — (1) lethal face if total available attack ≥ opponent health, attack face; (2) summon the most expensive affordable creature; (3) trade attacks into enemy creatures favoring kills without dying, else attack face; (4) `Pass`.

- [ ] **Step 1: Write failing test**

`tests/test_greedy.py`:

```python
from locma.policies.greedy import GreedyPolicy
from locma.core.views import CardView, DraftView

def test_greedy_prefers_stronger_card():
    weak = CardView(-1, 1, 0, 2, 1, 1, "------")
    strong = CardView(-1, 2, 0, 2, 4, 4, "------")
    mid = CardView(-1, 3, 0, 2, 2, 2, "------")
    view = DraftView(0, (weak, strong, mid))
    assert GreedyPolicy("g").draft_action(view, [0, 1, 2]) == 1

def test_greedy_beats_random_over_many_games():
    from locma.harness.match import run_match
    from locma.policies.random_policy import RandomPolicy
    res = run_match(GreedyPolicy("g"), RandomPolicy("r"), games=60, seed=0)
    assert res.win_rate_a > 0.5
```

(Second test depends on Task 13; run it after Task 13.)

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
from locma.core.actions import Summon, Attack, Use, Pass
from locma.core.cards import ABILITY_ORDER

def _kw_count(abilities): return sum(1 for ch in abilities if ch != "-")

def _score(cv):
    base = cv.attack + cv.defense + 0.5 * _kw_count(cv.abilities)
    if cv.type != 0:  # items slightly deprioritized in draft
        base -= 1.0
    return base

class GreedyPolicy:
    def __init__(self, name: str = "greedy"):
        self.name = name
    def draft_action(self, view, legal):
        scores = [_score(cv) for cv in view.offered]
        return max(legal, key=lambda i: scores[i])
    def battle_action(self, view, legal):
        attacks = [a for a in legal if isinstance(a, Attack)]
        face = [a for a in attacks if a.target_id == -1]
        total_face = 0
        for a in face:
            for c in view.my_board:
                if c.instance_id == a.attacker_id:
                    total_face += c.attack
        if face and total_face >= view.op_health:
            return face[0]
        summons = [a for a in legal if isinstance(a, Summon)]
        if summons:
            def cost_of(a):
                for c in view.my_hand:
                    if c.instance_id == a.card_instance_id:
                        return c.cost
                return 0
            return max(summons, key=cost_of)
        creature_attacks = [a for a in attacks if a.target_id != -1]
        if creature_attacks:
            return creature_attacks[0]
        if face:
            return face[0]
        return Pass()
    def reset(self, seed=None): pass
```

- [ ] **Step 4: Run → PASS** (first test now; second after Task 13).

- [ ] **Step 5: Commit**

```bash
git add locma/policies/greedy.py tests/test_greedy.py
git commit -m "feat(policies): greedy draft+battle heuristic"
```

---

### Task 13: Match harness + JSONL logging

**Files:**
- Create: `locma/harness/__init__.py`, `locma/harness/match.py`, `locma/harness/logging.py`
- Test: `tests/test_match.py`

**Interfaces:**
- Consumes: `run_game`.
- Produces:
  - `match.py`:
    - `@dataclass MatchResult(name_a, name_b, games, wins_a, wins_b, win_rate_a, records:list)`
    - `run_match(policy_a, policy_b, games:int, seed:int=0, jsonl_path=None) -> MatchResult` — plays `games` logical games as **mirrored seed pairs**: for pair `k`, game1 = A as player0 (seed `base+k`), game2 = B as player0 (same seed). A's win counted regardless of seat. Optionally appends per-game records to JSONL.
  - `logging.py`: `write_records(path, records:list[dict]) -> None` appends one JSON object per line.

- [ ] **Step 1: Write failing test**

`tests/test_match.py`:

```python
import json, os
from locma.harness.match import run_match
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy

def test_run_match_counts_and_balances_sides():
    res = run_match(RandomPolicy("a"), RandomPolicy("b"), games=10, seed=0)
    assert res.games == 20  # mirrored pairs => 2 games each
    assert res.wins_a + res.wins_b == 20
    assert 0.0 <= res.win_rate_a <= 1.0

def test_match_is_deterministic():
    r1 = run_match(RandomPolicy("a"), RandomPolicy("b"), games=8, seed=5)
    r2 = run_match(RandomPolicy("a"), RandomPolicy("b"), games=8, seed=5)
    assert (r1.wins_a, r1.wins_b) == (r2.wins_a, r2.wins_b)

def test_jsonl_written(tmp_path):
    p = tmp_path / "out.jsonl"
    run_match(ScriptedPolicy("s"), RandomPolicy("r"), games=3, seed=1, jsonl_path=str(p))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 6
    assert "winner_is_a" in json.loads(lines[0])
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `logging.py`**

```python
from __future__ import annotations
import json

def write_records(path: str, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
```

- [ ] **Step 4: Implement `match.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from locma.core.engine import run_game
from locma.harness.logging import write_records

@dataclass
class MatchResult:
    name_a: str
    name_b: str
    games: int
    wins_a: int
    wins_b: int
    win_rate_a: float
    records: list = field(default_factory=list)

def run_match(policy_a, policy_b, games: int, seed: int = 0, jsonl_path=None) -> MatchResult:
    wins_a = wins_b = 0
    records = []
    for k in range(games):
        s = seed + k
        # game 1: A is player0
        r1 = run_game(policy_a, policy_b, seed=s)
        a_won_1 = (r1.winner == 0)
        # game 2: B is player0 (mirror)
        r2 = run_game(policy_b, policy_a, seed=s)
        a_won_2 = (r2.winner == 1)
        for won, gr, a_seat in ((a_won_1, r1, 0), (a_won_2, r2, 1)):
            if won: wins_a += 1
            else: wins_b += 1
            records.append({"seed": gr.seed, "a_seat": a_seat,
                            "turns": gr.turns, "winner_is_a": bool(won)})
    total = games * 2
    res = MatchResult(policy_a.name, policy_b.name, total, wins_a, wins_b,
                      wins_a / total if total else 0.0, records)
    if jsonl_path:
        write_records(jsonl_path, records)
    return res
```

- [ ] **Step 5: Run → PASS.** Now run Task 12's second test (`test_greedy_beats_random_over_many_games`) → should PASS.

- [ ] **Step 6: Commit**

```bash
git add locma/harness tests/test_match.py
git commit -m "feat(harness): mirrored match runner + JSONL logging"
```

---

### Task 14: Stats — Wilson CI, binomial test, SPRT

**Files:**
- Create: `locma/stats/__init__.py`, `locma/stats/intervals.py`, `locma/stats/sprt.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `intervals.py`: `wilson_ci(wins:int, n:int, z:float=1.96) -> tuple[float,float]`; `binomial_test(wins:int, n:int, p0:float=0.5) -> float` (two-sided p-value via `scipy.stats.binomtest`).
  - `sprt.py`: `@dataclass SprtResult(decision:str, llr:float, n:int)` and `sprt(wins:int, n:int, p0:float, p1:float, alpha:float=0.05, beta:float=0.05) -> SprtResult` where `decision ∈ {"accept_h1","accept_h0","continue"}` using the Wald log-likelihood-ratio bounds.

- [ ] **Step 1: Write failing test**

`tests/test_stats.py`:

```python
from locma.stats.intervals import wilson_ci, binomial_test
from locma.stats.sprt import sprt

def test_wilson_ci_bounds():
    lo, hi = wilson_ci(60, 100)
    assert 0.0 <= lo < 0.6 < hi <= 1.0

def test_binomial_clear_signal():
    assert binomial_test(90, 100, 0.5) < 0.01
    assert binomial_test(50, 100, 0.5) > 0.5

def test_sprt_accepts_h1_when_dominant():
    r = sprt(95, 100, p0=0.5, p1=0.65)
    assert r.decision == "accept_h1"

def test_sprt_continue_when_ambiguous():
    r = sprt(11, 20, p0=0.5, p1=0.65)
    assert r.decision == "continue"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `intervals.py`**

```python
from __future__ import annotations
import math
from scipy.stats import binomtest

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = wins / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def binomial_test(wins: int, n: int, p0: float = 0.5) -> float:
    return binomtest(wins, n, p0, alternative="two-sided").pvalue
```

- [ ] **Step 4: Implement `sprt.py`**

```python
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass
class SprtResult:
    decision: str
    llr: float
    n: int

def sprt(wins: int, n: int, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.05) -> SprtResult:
    losses = n - wins
    llr = (wins * math.log(p1/p0)) + (losses * math.log((1-p1)/(1-p0)))
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if llr >= upper:
        return SprtResult("accept_h1", llr, n)
    if llr <= lower:
        return SprtResult("accept_h0", llr, n)
    return SprtResult("continue", llr, n)
```

- [ ] **Step 5: Run → PASS. Commit.**

```bash
git add locma/stats tests/test_stats.py
git commit -m "feat(stats): Wilson CI, binomial test, SPRT"
```

---

### Task 15: Ratings (Elo) + tournament

**Files:**
- Create: `locma/stats/ratings.py`, `locma/harness/tournament.py`
- Test: `tests/test_tournament.py`

**Interfaces:**
- Produces:
  - `ratings.py`: `elo_update(ra:float, rb:float, score_a:float, k:float=32) -> tuple[float,float]`; `elo_from_results(pairs:list[tuple[str,str,float]], start:float=1500) -> dict[str,float]` (score_a in {0,0.5,1} per game).
  - `tournament.py`: `@dataclass TournamentResult(policies:list[str], win_matrix:dict, ratings:dict, p_vs_reference:dict)`; `run_tournament(policies:list, games:int=50, seed:int=0, reference:str|None=None) -> TournamentResult` — round-robin all pairs via `run_match`, builds win-rate matrix `(a,b)->win_rate_a`, Elo ratings, and (if `reference` given) binomial p-value of each policy vs the reference.

- [ ] **Step 1: Write failing test**

`tests/test_tournament.py`:

```python
from locma.stats.ratings import elo_update
from locma.harness.tournament import run_tournament
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy

def test_elo_winner_gains():
    ra, rb = elo_update(1500, 1500, 1.0)
    assert ra > 1500 > rb

def test_tournament_structure():
    pols = [RandomPolicy("r"), ScriptedPolicy("s")]
    res = run_tournament(pols, games=6, seed=0, reference="r")
    assert ("r", "s") in res.win_matrix
    assert set(res.ratings) == {"r", "s"}
    assert "s" in res.p_vs_reference
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `ratings.py`**

```python
from __future__ import annotations

def elo_update(ra: float, rb: float, score_a: float, k: float = 32) -> tuple[float, float]:
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    ra2 = ra + k * (score_a - ea)
    rb2 = rb + k * ((1 - score_a) - eb)
    return ra2, rb2

def elo_from_results(pairs, start: float = 1500) -> dict:
    ratings: dict = {}
    for a, b, score_a in pairs:
        ratings.setdefault(a, start); ratings.setdefault(b, start)
        ratings[a], ratings[b] = elo_update(ratings[a], ratings[b], score_a)
    return ratings
```

- [ ] **Step 4: Implement `tournament.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from locma.harness.match import run_match
from locma.stats.ratings import elo_from_results
from locma.stats.intervals import binomial_test

@dataclass
class TournamentResult:
    policies: list
    win_matrix: dict
    ratings: dict
    p_vs_reference: dict

def run_tournament(policies, games: int = 50, seed: int = 0, reference=None) -> TournamentResult:
    names = [p.name for p in policies]
    win_matrix = {}
    elo_pairs = []
    totals = {n: [0, 0] for n in names}  # wins, games vs reference
    for a, b in combinations(policies, 2):
        res = run_match(a, b, games=games, seed=seed)
        win_matrix[(a.name, b.name)] = res.win_rate_a
        win_matrix[(b.name, a.name)] = res.wins_b / res.games
        for _ in range(res.wins_a): elo_pairs.append((a.name, b.name, 1.0))
        for _ in range(res.wins_b): elo_pairs.append((a.name, b.name, 0.0))
        if reference in (a.name, b.name):
            other = b if a.name == reference else a
            wins_other = res.wins_b if a.name == reference else res.wins_a
            totals[other.name][0] += wins_other
            totals[other.name][1] += res.games
    ratings = elo_from_results(elo_pairs)
    for n in names:
        ratings.setdefault(n, 1500)
    p_vs_reference = {}
    if reference:
        for n in names:
            if n == reference: continue
            w, g = totals[n]
            p_vs_reference[n] = binomial_test(w, g, 0.5) if g else 1.0
    return TournamentResult(names, win_matrix, ratings, p_vs_reference)
```

- [ ] **Step 5: Run → PASS. Commit.**

```bash
git add locma/stats/ratings.py locma/harness/tournament.py tests/test_tournament.py
git commit -m "feat(harness): Elo ratings + round-robin tournament"
```

---

### Task 16: CLI (Typer)

**Files:**
- Create: `locma/cli/__init__.py`, `locma/cli/app.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces a Typer `app` with commands:
  - `play A B --games N --seed S` — prints win rate, Wilson CI, binomial p-value.
  - `tournament NAMES... --games N` — prints Rich table of ratings + win matrix.
  - `eval X --vs random --p0 0.5 --p1 0.6 --max-games N` — runs SPRT loop, prints verdict.
  - `fetch-cards` / `fetch-art` — call data layer (Task 17).
  - A `POLICIES` registry mapping names → constructors: `random, scripted, greedy` (and `sb3:<path>` later).

- [ ] **Step 1: Write failing test**

`tests/test_cli.py`:

```python
from typer.testing import CliRunner
from locma.cli.app import app

runner = CliRunner()

def test_play_command_runs():
    r = runner.invoke(app, ["play", "scripted", "random", "--games", "5", "--seed", "0"])
    assert r.exit_code == 0
    assert "win rate" in r.stdout.lower()

def test_eval_command_runs():
    r = runner.invoke(app, ["eval", "greedy", "--vs", "random", "--max-games", "40"])
    assert r.exit_code == 0
    assert "verdict" in r.stdout.lower() or "accept" in r.stdout.lower()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `app.py`**

```python
from __future__ import annotations
import typer
from rich.console import Console
from rich.table import Table
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy
from locma.policies.greedy import GreedyPolicy
from locma.harness.match import run_match
from locma.harness.tournament import run_tournament
from locma.stats.intervals import wilson_ci, binomial_test
from locma.stats.sprt import sprt

app = typer.Typer(help="Legends of Code & Magic 1.2 explore kit")
console = Console()

def make_policy(spec: str):
    table = {"random": RandomPolicy, "scripted": ScriptedPolicy, "greedy": GreedyPolicy}
    if spec in table:
        return table[spec](spec)
    raise typer.BadParameter(f"unknown policy '{spec}'")

@app.command()
def play(a: str, b: str, games: int = 100, seed: int = 0):
    res = run_match(make_policy(a), make_policy(b), games=games, seed=seed)
    lo, hi = wilson_ci(res.wins_a, res.games)
    p = binomial_test(res.wins_a, res.games, 0.5)
    console.print(f"[bold]{a}[/] vs [bold]{b}[/]  win rate A = {res.win_rate_a:.3f} "
                  f"(95% CI {lo:.3f}-{hi:.3f}), p={p:.4g}, n={res.games}")

@app.command()
def tournament(names: list[str], games: int = 50, seed: int = 0, reference: str = "random"):
    pols = [make_policy(n) for n in names]
    res = run_tournament(pols, games=games, seed=seed, reference=reference)
    t = Table(title="Ratings")
    t.add_column("policy"); t.add_column("elo", justify="right"); t.add_column("p vs ref", justify="right")
    for n in sorted(res.ratings, key=lambda k: -res.ratings[k]):
        t.add_row(n, f"{res.ratings[n]:.0f}", f"{res.p_vs_reference.get(n, float('nan')):.4g}")
    console.print(t)

@app.command()
def eval(x: str, vs: str = "random", p0: float = 0.5, p1: float = 0.6,
         max_games: int = 1000, batch: int = 20, seed: int = 0):
    px, py = make_policy(x), make_policy(vs)
    wins = n = 0
    k = 0
    while n < max_games:
        res = run_match(px, py, games=batch, seed=seed + k); k += batch
        wins += res.wins_a; n += res.games
        r = sprt(wins, n, p0, p1)
        if r.decision != "continue":
            break
    lo, hi = wilson_ci(wins, n)
    console.print(f"verdict: [bold]{r.decision}[/]  winrate={wins/n:.3f} "
                  f"(CI {lo:.3f}-{hi:.3f}), n={n}")

@app.command("fetch-cards")
def fetch_cards_cmd():
    from locma.data.fetch import fetch_cards
    path = fetch_cards()
    console.print(f"cards at {path}")

@app.command("fetch-art")
def fetch_art_cmd():
    from locma.data.fetch import fetch_art
    n = fetch_art()
    console.print(f"fetched {n} art assets (best-effort)")
```

- [ ] **Step 4: Run → PASS.** `uv run pytest tests/test_cli.py -v`

- [ ] **Step 5: Commit**

```bash
git add locma/cli tests/test_cli.py
git commit -m "feat(cli): play, tournament, eval, fetch commands"
```

---

### Task 17: Data fetch commands (cards refresh + best-effort art)

**Files:**
- Create: `locma/data/fetch.py`, `locma/data/assets/manifest.json` (empty `{}`)
- Test: `tests/test_fetch.py`

**Interfaces:**
- Produces:
  - `fetch_cards(dest=None) -> str` — downloads cardlist from the confirmed source URL (recorded in Task 3) to `locma/data/cardlist.txt`, verifies 160 lines parse, returns path. On network failure, logs and returns the existing vendored path.
  - `fetch_art(dest=None) -> int` — best-effort: iterate card ids, attempt to download images from the confirmed art source into `assets/`, update `manifest.json` (id→{file,url}); never raises — counts successes. Returns count.

- [ ] **Step 1: Write failing test**

`tests/test_fetch.py`:

```python
from locma.data.fetch import fetch_art

def test_fetch_art_never_raises(monkeypatch):
    # force all downloads to fail; must not raise, returns int
    import locma.data.fetch as F
    monkeypatch.setattr(F, "_download", lambda url, path: False)
    assert isinstance(fetch_art(), int)
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
import json, os, urllib.request
from importlib import resources
from locma.data.cards_db import parse_cardlist, load_cards

CARDLIST_URL = "https://raw.githubusercontent.com/ronaldosvieira/gym-locm/master/gym_locm/engine/resources/cardlist.txt"
ART_URL_TEMPLATE = ""  # confirm during implementation; empty disables art fetch

def _download(url: str, path: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False

def _data_dir() -> str:
    return str(resources.files("locma.data"))

def fetch_cards(dest=None) -> str:
    path = dest or os.path.join(_data_dir(), "cardlist.txt")
    tmp = path + ".tmp"
    if _download(CARDLIST_URL, tmp):
        text = open(tmp, encoding="utf-8").read()
        if len(parse_cardlist(text)) == 160:
            os.replace(tmp, path)
        else:
            os.remove(tmp)
    else:
        if os.path.exists(tmp): os.remove(tmp)
    return path

def fetch_art(dest=None) -> int:
    if not ART_URL_TEMPLATE:
        return 0
    art_dir = dest or os.path.join(_data_dir(), "assets")
    os.makedirs(art_dir, exist_ok=True)
    manifest_path = os.path.join(art_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
    count = 0
    for c in load_cards():
        fname = f"{c.id}.png"
        fpath = os.path.join(art_dir, fname)
        if os.path.exists(fpath):
            continue
        url = ART_URL_TEMPLATE.format(id=c.id)
        if _download(url, fpath):
            manifest[str(c.id)] = {"file": fname, "url": url}
            count += 1
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return count
```

- [ ] **Step 4: Run → PASS. Commit.**

```bash
git add locma/data/fetch.py locma/data/assets/manifest.json tests/test_fetch.py
git commit -m "feat(data): cards refresh + best-effort art fetch"
```

---

### Task 18: Gym env + action encoding (optional `[ml]`)

**Files:**
- Create: `locma/envs/__init__.py`, `locma/envs/encode.py`, `locma/envs/battle_env.py`
- Test: `tests/test_env.py` (skipped if gymnasium not installed)

**Interfaces:**
- Produces:
  - `encode.py`: `OBS_SIZE:int`; `encode_battle(view) -> np.ndarray` (float32, fixed length); `ACTION_SIZE:int`; `action_mask(legal) -> np.ndarray[bool]`; `index_to_action(idx, legal) -> Action` (maps a discrete index onto the legal list; out-of-range → Pass). Keep v1 mapping simple: action index = position in a canonicalized legal list of fixed max length `ACTION_SIZE`; mask marks valid positions.
  - `battle_env.py`: `class BattleEnv(gymnasium.Env)` wrapping `run_game`-style stepping for the agent seat with a fixed opponent `Policy`; `reset()`, `step(action_idx)`, `action_masks()`. Reward = +1 win, -1 loss, 0 otherwise.

- [ ] **Step 1: Write failing test** (guarded)

`tests/test_env.py`:

```python
import pytest
gym = pytest.importorskip("gymnasium")
from locma.envs.battle_env import BattleEnv
from locma.policies.random_policy import RandomPolicy

def test_env_reset_step():
    env = BattleEnv(opponent=RandomPolicy("opp"), seed=0)
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    mask = env.action_masks()
    assert mask.any()
    import numpy as np
    idx = int(np.argmax(mask))
    obs, reward, terminated, truncated, info = env.step(idx)
    assert reward in (-1.0, 0.0, 1.0)
```

- [ ] **Step 2: Run → FAIL** (or skip if gymnasium absent; install `[ml]` to develop this task).

- [ ] **Step 3: Implement `encode.py`** (fixed-length canonical mapping)

```python
from __future__ import annotations
import numpy as np
from locma.core.actions import Pass

ACTION_SIZE = 64  # max canonical legal actions considered per decision
OBS_SIZE = 6 + 6*7 + 6*7 + 8*7  # turn/hp/mana scalars + board+hand slots; see below

def encode_battle(view) -> np.ndarray:
    vec = [float(view.turn), float(view.me_health), float(view.me_mana),
           float(view.op_health), float(view.op_hand_count), 0.0]
    def card_feats(seq, n):
        out = []
        for i in range(n):
            if i < len(seq):
                c = seq[i]
                out += [1.0, c.cost, c.attack, c.defense,
                        float(c.abilities.count("G") > 0),
                        float(c.abilities.count("L") > 0),
                        float(c.abilities.count("W") > 0)]
            else:
                out += [0.0]*7
        return out
    vec += card_feats(view.my_board, 6)
    vec += card_feats(view.op_board, 6)
    vec += card_feats(view.my_hand, 8)
    return np.asarray(vec, dtype=np.float32)

def action_mask(legal) -> np.ndarray:
    m = np.zeros(ACTION_SIZE, dtype=bool)
    m[:min(len(legal), ACTION_SIZE)] = True
    return m

def index_to_action(idx, legal):
    if 0 <= idx < len(legal):
        return legal[idx]
    return Pass()
```

Ensure `OBS_SIZE == len(encode_battle(sample))`; add an assertion test in `test_env.py` if needed.

- [ ] **Step 4: Implement `battle_env.py`**

```python
from __future__ import annotations
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from locma.core.state import GameState, Phase
from locma.core import draft as draftmod, battle as battlemod
from locma.core.engine import make_battle_view, make_draft_view
from locma.envs.encode import encode_battle, action_mask, index_to_action, OBS_SIZE, ACTION_SIZE
from locma.data.cards_db import load_cards

class BattleEnv(gym.Env):
    def __init__(self, opponent, seed: int = 0, agent_seat: int = 0):
        self.opponent = opponent; self.base_seed = seed; self.agent_seat = agent_seat
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self._cards = load_cards(); self._ep = 0

    def _opp_play_until_agent(self):
        while self.gs.phase == Phase.BATTLE and self.gs.current != self.agent_seat:
            legal = battlemod.battle_legal(self.gs)
            view = make_battle_view(self.gs)
            self.gs and battlemod.apply_battle(self.gs, self.opponent.battle_action(view, legal))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.base_seed + self._ep; self._ep += 1
        self.gs = GameState.new(random.Random(s))
        draftmod.start_draft(self.gs, self._cards)
        while self.gs.phase == Phase.DRAFT:
            dv = make_draft_view(self.gs)
            pol = self.opponent  # opponent drafts both seats in v1 battle-only env
            draftmod.apply_draft_pick(self.gs, pol.draft_action(dv, [0,1,2]))
        battlemod.start_battle(self.gs)
        self._opp_play_until_agent()
        return encode_battle(make_battle_view(self.gs)), {}

    def action_masks(self):
        return action_mask(battlemod.battle_legal(self.gs))

    def step(self, idx):
        legal = battlemod.battle_legal(self.gs)
        battlemod.apply_battle(self.gs, index_to_action(int(idx), legal))
        if self.gs.phase != Phase.ENDED:
            self._opp_play_until_agent()
        terminated = self.gs.phase == Phase.ENDED
        reward = 0.0
        if terminated:
            reward = 1.0 if self.gs.winner == self.agent_seat else -1.0
        obs = encode_battle(make_battle_view(self.gs)) if not terminated else np.zeros(OBS_SIZE, np.float32)
        return obs, reward, terminated, False, {}
```

- [ ] **Step 5: Run → PASS** (with `[ml]` installed). Commit.

```bash
git add locma/envs tests/test_env.py
git commit -m "feat(envs): gymnasium BattleEnv with action masking"
```

---

### Task 19: SB3 policy wrapper + reference training script

**Files:**
- Create: `locma/policies/sb3_policy.py`, `train.py`
- Test: `tests/test_sb3.py` (skipped without `[ml]`)

**Interfaces:**
- Produces:
  - `sb3_policy.py`: `class SB3Policy` — `__init__(self, model_path, name=None)` loads a MaskablePPO model lazily; `battle_action(view, legal)` encodes obs, builds mask, predicts, maps to action; `draft_action` delegates to a provided draft policy (default `RandomPolicy`).
  - `train.py`: minimal MaskablePPO training over `BattleEnv` vs `RandomPolicy`, saves `model.zip`. Runnable: `python train.py --steps 50000`.

- [ ] **Step 1: Write failing test** (guarded)

`tests/test_sb3.py`:

```python
import pytest
pytest.importorskip("sb3_contrib")
from locma.policies.sb3_policy import SB3Policy

def test_sb3_policy_constructs_without_model():
    # name set; model loads lazily so construction must not require a file
    p = SB3Policy(model_path="nonexistent.zip", name="ppo")
    assert p.name == "ppo"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `sb3_policy.py`**

```python
from __future__ import annotations
import numpy as np
from locma.policies.random_policy import RandomPolicy
from locma.envs.encode import encode_battle, action_mask, index_to_action

class SB3Policy:
    def __init__(self, model_path: str, name=None, draft=None):
        self.model_path = model_path; self.name = name or "sb3"
        self.draft = draft or RandomPolicy("sb3-draft")
        self._model = None
    def _ensure(self):
        if self._model is None:
            from sb3_contrib import MaskablePPO
            self._model = MaskablePPO.load(self.model_path)
    def draft_action(self, view, legal): return self.draft.draft_action(view, legal)
    def battle_action(self, view, legal):
        self._ensure()
        obs = encode_battle(view)
        mask = action_mask(legal)
        idx, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return index_to_action(int(idx), legal)
    def reset(self, seed=None): self.draft.reset(seed)
```

- [ ] **Step 4: Implement `train.py`**

```python
from __future__ import annotations
import argparse
from sb3_contrib import MaskablePPO
from locma.envs.battle_env import BattleEnv
from locma.policies.random_policy import RandomPolicy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--out", default="model.zip")
    args = ap.parse_args()
    env = BattleEnv(opponent=RandomPolicy("opp"), seed=0)
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run → PASS (construction test). Commit.**

```bash
git add locma/policies/sb3_policy.py train.py tests/test_sb3.py
git commit -m "feat(ml): SB3Policy wrapper + reference MaskablePPO training"
```

---

### Task 20: Full-suite verification + README quickstart

**Files:**
- Modify: `README.md`
- Test: run the whole suite.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all non-`[ml]` tests PASS; `[ml]` tests pass if extra installed, else skipped.

- [ ] **Step 2: Manual CLI smoke**

Run:
```bash
uv run locma play greedy random --games 50 --seed 0
uv run locma eval greedy --vs random --max-games 200
uv run locma tournament random scripted greedy --games 30
```
Expected: greedy win rate vs random > 0.5; eval verdict `accept_h1`; tournament prints a ratings table with greedy on top.

- [ ] **Step 3: Update README quickstart**

Append install + usage (uv sync, the three commands above, how to train with `[ml]`).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add quickstart and verify full suite"
```

---

## Self-Review

**Spec coverage:**
- §2 architecture/layout → Tasks 1–19 follow the package map. ✓
- §3 engine (cards, instance, state, draft, battle keywords, determinism) → Tasks 2,4,5,6,7,8,9. ✓
- §4 policy interface + 4 baselines → Tasks 11 (Random/Scripted/Composite), 12 (Greedy), 19 (SB3). ✓
- §5 eval/tournament (mirrored matches, parallelism, JSONL, Wilson/binomial/SPRT/Elo, win matrix, p-vs-reference) → Tasks 13,14,15. **Parallelism note:** v1 ships sequential `run_match`/`run_tournament`; parallel fan-out via `concurrent.futures` is a safe later optimization (engine is pure/seeded). Flagged here rather than silently dropped.
- §6 Gym/SB3 (env, masking, SB3Policy, train.py) → Tasks 18,19. ✓
- §7 data (vendored cards, fetch-cards, best-effort art, manifest) → Tasks 3,17. ✓
- §8 CLI commands → Task 16. ✓
- TrueSkill (§5 "optional") → deferred; Elo ships in Task 15. Listed as optional in spec, acceptable to defer.

**Placeholder scan:** No "TBD"/"add error handling" placeholders; the two genuinely external unknowns (exact cardlist column order, art URL) are handled as explicit discovery steps with verifications (160-count assert; art disabled until URL confirmed), not silent gaps.

**Type consistency:** `Card`/`CardInstance` fields, `CardView` shape, `Action` variants, `MatchResult`/`GameResult`/`TournamentResult`/`SprtResult` names and signatures are consistent across tasks. `battle_legal`/`apply_battle`/`_resolve_attack`/`make_battle_view`/`make_draft_view` names match between engine, env, and harness.

**Known build-order coupling:** Task 10 tests depend on Task 11; noted inline. Greedy's second test depends on Task 13; noted inline.
