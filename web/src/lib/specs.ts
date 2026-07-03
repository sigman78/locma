// Policy-spec knowledge for the UI: what each registry base is, what its
// positional parameters mean, and a parser that explains any spec string.
// Kept client-side because this is presentation copy; the server registry
// (locma/policies/registry.py) remains the source of truth for behavior.

export interface ParamInfo {
  name: string
  meaning: string
  default: string
}

export interface BaseInfo {
  label: string
  blurb: string
  params: ParamInfo[]
}

export const SPEC_INFO: Record<string, BaseInfo> = {
  random: {
    label: 'Random',
    blurb: 'Uniform random legal action — the sanity floor.',
    params: [],
  },
  scripted: {
    label: 'Scripted',
    blurb: 'Random draft + fixed aggressive battle script (items, attack face/Guard, summon).',
    params: [],
  },
  greedy: {
    label: 'Greedy',
    blurb: 'Stat-based draft + greedy lethal/trade battle heuristic.',
    params: [],
  },
  'max-guard': {
    label: 'Max-Guard',
    blurb: 'Draft prefers Guard creatures; aggressive ground battle.',
    params: [],
  },
  'max-attack': {
    label: 'Max-Attack',
    blurb: 'Draft prefers highest attack; aggressive ground battle.',
    params: [],
  },
  mcts: {
    label: 'MCTS (cheating)',
    blurb: 'Perfect-information MCTS — it PEEKS at your hand. Strong but unfair; slow.',
    params: [
      { name: 'iterations', meaning: 'search iterations per move', default: '100' },
      { name: 'c', meaning: 'UCB exploration constant', default: '1.41' },
      { name: 'seed', meaning: 'search RNG seed', default: '0' },
      { name: 'rollout_turns', meaning: 'heuristic rollout depth in turns (0 = random)', default: '3' },
    ],
  },
  dmcts: {
    label: 'Determinized MCTS (fair)',
    blurb: 'Samples K possible hidden hands, searches each, votes. Fair (public info only).',
    params: [
      { name: 'K', meaning: 'determinizations (sampled hidden worlds)', default: '15' },
      { name: 'I', meaning: 'MCTS iterations per world', default: '30' },
      { name: 'seed', meaning: 'search RNG seed', default: '0' },
      { name: 'rollout_turns', meaning: 'heuristic rollout depth in turns', default: '3' },
    ],
  },
  azlite: {
    label: 'AZ-lite',
    blurb: 'PUCT-guided search with a heuristic policy/value oracle (no net needed).',
    params: [
      { name: 'iterations', meaning: 'search iterations per move', default: '100' },
      { name: 'c_puct', meaning: 'PUCT exploration constant', default: '1.5' },
      { name: 'seed', meaning: 'search RNG seed', default: '0' },
      { name: 'rollout_turns', meaning: 'heuristic rollout depth (0 = value oracle only)', default: '0' },
    ],
  },
  netdmcts: {
    label: 'Net-guided DMCTS',
    blurb: 'Determinized PUCT guided by a trained net (needs the [ml] extra + a model).',
    params: [
      { name: 'K', meaning: 'determinizations (sampled hidden worlds)', default: '15' },
      { name: 'I', meaning: 'PUCT iterations per world', default: '80' },
      { name: 'c_puct', meaning: 'PUCT exploration constant', default: '1.5' },
      { name: 'model', meaning: 'checkpoint path or depot: ref', default: 'model.zip' },
    ],
  },
  vbeam: {
    label: 'V-beam planner',
    blurb:
      'Beam-searches whole own turns and scores stopping points with the net’s value ' +
      'head — the strongest known policy (avg-hard3 0.863 with depot:b0). Needs [ml].',
    params: [
      { name: 'model', meaning: 'checkpoint path or depot: ref (token-obs net)', default: 'model.zip' },
      { name: 'width', meaning: 'beam width — candidate plans kept per depth', default: '8' },
      { name: 'max_actions', meaning: 'max actions planned within one turn', default: '20' },
    ],
  },
  ppo: {
    label: 'PPO (reactive)',
    blurb:
      'Reactive MaskablePPO net, one forward pass per action (avg-hard3 0.657 with ' +
      'depot:b0). Needs [ml].',
    params: [{ name: 'model', meaning: 'checkpoint path or depot: ref', default: 'model.zip' }],
  },
}

export interface ExplainedParam extends ParamInfo {
  value: string
  isDefault: boolean
}

export interface Explanation {
  base: string
  known: boolean
  blurb: string
  params: ExplainedParam[]
}

/** Explain a registry spec string `base[:p1,p2,...]` — parameter values are
 * matched positionally against SPEC_INFO, defaults filled for the rest. */
export function explainSpec(spec: string): Explanation {
  const trimmed = spec.trim()
  const ci = trimmed.indexOf(':')
  const base = ci === -1 ? trimmed : trimmed.slice(0, ci)
  const info = SPEC_INFO[base]
  if (!info) {
    return { base, known: false, blurb: 'Unknown policy base — check the spec.', params: [] }
  }
  const raw = ci === -1 || ci === trimmed.length - 1 ? [] : trimmed.slice(ci + 1).split(',')
  // model-path params may themselves contain ':' but never ',' (depot refs are comma-free)
  const params = info.params.map((p, i) => {
    const given = (raw[i] ?? '').trim()
    return { ...p, value: given || p.default, isDefault: !given }
  })
  return { base, known: true, blurb: info.blurb, params }
}
