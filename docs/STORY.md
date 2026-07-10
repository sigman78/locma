# The story so far: teaching a machine to play a card game

This project builds AI to play *Legends of Code & Magic* (LOCM) — a small
two-phase card game: first you **draft** a 30-card deck by picking 1-of-3
offered cards, 30 times; then you **battle** with it. Simple rules, deep
tactics — a good testbed for reinforcement learning (RL). What follows is the
short version of a few months of experiments: what worked, what didn't, and
why. Numbers are win rate against a fixed panel of strong scripted
opponents ("avg-hard3"); 0.50 is a coin flip, higher is better.

## Chapter 1 — First steps: scripts, then learning

We started with hand-written scripted strategies (always attack the
strongest thing, always draft the highest-stat card, etc.) as a measuring
stick, then trained a neural net with PPO (a standard RL algorithm) to play
reactively — one observation in, one action out, no lookahead. An early bug
(the action space didn't line up with what the game actually allowed) quietly
capped the net below the baselines; fixing it let PPO catch and pass them.
But then progress stalled. No amount of training budget, opponent variety,
reward shaping, bigger networks, or self-play moved the number. Something
structural was missing.

## Chapter 2 — The wall: reactive nets can't plan

We had two flavors of search-based opponents: one that (unfairly) sees the
opponent's whole hand and picks the best move by tree search, and a fair
version that only samples plausible hidden hands. Both thrash every reactive
net. We tried to teach a reactive net to imitate search moves directly
(behavior cloning) — best case, it copied the *teacher's own moves* only
**37-40%** of the time, no matter how strong, fair, or deterministic the
teacher was. A simple scripted heuristic clones at 95%. The conclusion: a
search move is the *output* of lookahead, and lookahead has no compact
one-shot form — you cannot compress planning into a bigger or better-trained
reflex. If you want the ceiling, you have to plan **at play time**, not train
harder.

## Chapter 3 — The deck is a lever too

Somewhat by accident, we noticed swapping which scripted policy drafted the
deck moved the win rate more than most training changes did. A curve-aware,
creature-majority "balanced" drafter consistently beat naive "always take the
biggest stats" drafting. Lesson banked for later: deck construction is a
second, mostly independent lever from battle skill.

## Chapter 4 — Planning at play time: the single biggest jump

Chapter 2's conclusion, taken literally: bolt a shallow beam search onto the
*same* trained network, using its own value estimate to score a few
turns ahead before acting (called `vbeam`). Reactive net alone: **0.657**.
Same net, searching its own turn before committing: **0.863**. A **+0.206**
jump — still the largest single result in the project, and it confirmed the
wall in Chapter 2 was real and crossable, just not from the training side.

## Chapter 5 — Cheap wins, then diminishing returns

Once search was the lever, we mined it. Training three battle nets on
slightly different decks and averaging their value estimates (an ensemble)
lifted the planner to **0.926** — for *zero* extra training, just more
compute at eval time. But stacking more diversity sources (more critics, more
opponent variety, combined mechanisms) never stacked the gains further — one
saturating resource, not a ladder. A recurring pattern showed up here for the
first of three times: a policy network mostly doesn't care *which* decks it
trained on, but the value network (the critic) does.

## Chapter 6 — Even a strong net has blind spots

A scripted "always keep a fat defensive board" strategy beat our best
reactive net **54%** of the time — a strategy the net never saw enough of in
training. The planner shrugged it off (loses only ~19-25% of the time to the
same trick) — search self-corrects against surprises a fixed reactive policy
can't. Adding that exploit into the training mix helped, but only to parity
(50/50), not a real fix — a loose thread that got tied off much later, from
an unexpected direction (Chapter 8).

## Chapter 7 — Six ways to "absorb" the planner into a reactive net, all closed

If play-time search works, can you distill its judgment back into a fast
reactive net — ranking losses, imitating full plans, richer observations,
recurrent memory, ensembles of critics as teachers? We ran the program to
its end: every escalation hit the same **capacity wall** — a frozen feature
extractor could separate "clearly good vs. bad" moves (91% accuracy) but not
fine margins between close options (72%), and that gap is exactly where the
planner's edge lives. After six negative, confidence-interval-backed
attempts, we closed the program: **training-side absorption doesn't work;
play-time search is the only confirmed way to get planning.**

## Chapter 8 — An outside idea pays off big: let RL draft the deck

Revisiting a competition-winning approach (ByteRL, no search, just very
careful RL) suggested two untried levers. First, matching their reward
discounting — tested, and it made things worse (our training runs are much
shorter, so their trick doesn't transfer). Second, and much bigger: instead
of a scripted drafter, train a *second* RL agent to draft the deck itself,
rewarded by how often the resulting deck wins. Result: reactive win rate
**0.683 -> 0.791**, planner **0.926 -> 0.978**. The learned decks pack in
more spells (previously almost none) *and* a much younger, cheaper curve —
a combination no amount of tuning the old scripted drafter's knobs could
reach (turning the "use more items" dial alone made things worse).
As a bonus, the better deck also mostly closed Chapter 6's blind spot
(**0.514 -> 0.408** against the board-keeping exploit) — the fix was in the
cards, not the training curriculum.

## Chapter 9 — Getting the deck-building win for free

A learned drafter is a neural net you have to run at every pick. Could a
cheap lookup table get most of the benefit? Fitting a static "card
priority list" against how the net actually drafted real games failed
completely (real games mix card quality with deck context in ways that
confused the fit). But asking the net a cleaner question — "all else equal,
which of these three cards do you prefer?" — produced a clean, confident
ranking (72% predictable, vs. 50% for the confounded version). Combined with
hand-recalibrated deck-shape targets (read straight off the learned decks'
statistics), that ranking recovered **67%** of the learned drafter's reactive
gain and matched its planner result — with **no neural net running at draft
time**. Published as a usable, free draft option, not (yet) a full recipe
of record.

## Chapter 10 — Deeper beats wider: the planner's one-turn horizon gives out

The planner searches exactly one turn before committing. Real tree search
(MCTS) plans many turns ahead — and when we let the two play each other
directly, MCTS started *beating* the planner once it had enough thinking
budget, even the fair version that only samples plausible hidden hands. So the
planner's ceiling wasn't information, it was **horizon**. We tested this on a
net-guided determinized search (`netdmcts`), which spends a fixed pool of
neural evaluations across `K` sampled hidden-world "determinizations", each
searched for `I` iterations. Historically we'd set `K=8` — eight shallow trees.
Holding the total budget fixed (`K*I=320`) and sweeping the split, the answer
was clean and monotone: **one deep tree beats eight shallow ones**, in both
critic families, by 20-30 win-rate points. The old `K=8` default was near the
*bottom* of the curve. Concentrated into a single 320-iteration tree
(`netdmcts:1,320` on the shared critics, same learned draft both sides), fair
net-guided search beat the planner head-to-head **0.575** (confirmed on 200
fresh games, CI [0.506, 0.642]) — the first fair, matched-draft config to clear
the planner, and the new play-time-search recipe of record. The margin is
modest, but the direction is unambiguous: after training-side absorption
(Chapter 7) closed, deeper search is where the next planning gains live.

## Chapter 11 — Buying the depth cheaply: one opponent reply is enough

Chapter 10 said depth is the frontier, but its winning tool — a 320-iteration
determinized tree — was expensive (~17 seconds a game). If the *reason* depth
works is that the planner never sees the opponent's reply, maybe we don't need a
deep tree at all: maybe we just need **one** genuine reply. So we built `rbeam`
("reply-aware turn beam"): take the planner's few best whole-turn plans, and for
each, sample a handful of plausible hidden worlds, end the turn, let the
opponent play its *own* strongest reply in that world, and only then score the
resulting position. Pick the plan that looks best after the opponent gets to
answer. It stays fair — we commit to one plan and average over what the opponent
might hold. Sweeping how many plans and how many worlds to consider, a clean
rule emerged: **grow both together** (a balanced 4×4 beat lopsided splits of the
same cost). At 4×4 it beat the planner head-to-head **0.640** (confirmed, 200
fresh games) — a bigger margin than the deep tree's 0.575 — and beat the deep
`netdmcts` tree itself **0.548** (confirmed over 500 games, once we ran enough
to resolve a genuine coin-flip-looking result) at **~2.6× less compute**. One
real reply, it turns out, captures most of what many turns of tree search were
buying — and it became the new play-time-search recipe of record.

## Conclusion

| stage | reactive win rate | planner win rate |
|---|---|---|
| first working PPO net | 0.657 | 0.863 (with search bolted on) |
| + opponent-hardened training, ensemble critics | 0.683 | 0.926 |
| + RL-learned draft | 0.791 | 0.978 |
| + free lookup-table draft (no net at draft time) | ~0.75 (partial) | ~0.978 (matched, pilot-scale) |
| + deep single-tree netdmcts (play-time search) | — | beats planner 0.575 head-to-head |
| + rbeam, one opponent-reply ply | — | beats planner 0.640 / beats netdmcts 0.548, ~2.6x cheaper |

Three lessons carried the whole project. **Planning beats training** — every
attempt to train the search advantage into a reactive net failed, while
bolting search onto training-time gains at play time worked every time we
tried it (+0.206 alone) — and Chapters 10-11 sharpened it further: *deeper*
search beats *wider* search, so the planner's one-turn horizon, not its
information, is the live frontier — and most of that depth turned out to be
buyable cheaply, with a single genuine opponent reply rather than a deep tree.
**The deck is not a side quest** — it turned out to
be the second-biggest lever in the whole project, on par with several
training-side improvements combined, and the place a late, big win still
came from. And **knowing when to stop is progress** — recognizing a real
capacity wall (Chapter 7) after six careful, confidence-interval-backed
negative results saved months chasing a dead end, and freed the time that
found Chapter 8's win instead.
