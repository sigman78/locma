import type { ActionDict, EventDict } from "./replay";

export interface Splash {
  seat: number;
  target: number | "face";
  amount: number;
  fatal: boolean;
}

export interface Lunge {
  seat: number;
  iid: number;
  toward: "face" | { seat: number; iid: number };
}

export interface Cast {
  seat: number;
}

/** Cards freshly drawn at a start-of-turn beat, highlighted in the owner's hand. */
export interface Drawn {
  seat: number;
  iids: number[];
}

export interface Fx {
  lunge: Lunge | null;
  cast: Cast | null;
  splashes: Splash[];
  drawn: Drawn | null;
}

export function computeFx(
  events: EventDict[],
  action: ActionDict | null,
  seat: number,
): Fx {
  const splashes: Splash[] = [];
  const damaged = new Set<number>();
  for (const e of events) {
    if (e.t === "damage") {
      splashes.push({
        seat: e.seat,
        target: e.target,
        amount: e.amount,
        fatal: e.fatal,
      });
      if (typeof e.target === "number") damaged.add(e.target);
    }
  }
  // Units removed without a damage event (e.g. red/blue item kills) still animate death.
  for (const e of events) {
    if (e.t === "unit_died" && !damaged.has(e.iid)) {
      splashes.push({ seat: e.seat, target: e.iid, amount: 0, fatal: true });
    }
  }

  let lunge: Lunge | null = null;
  let cast: Cast | null = null;
  if (action?.t === "attack") {
    lunge = {
      seat,
      iid: action.a,
      toward:
        action.target === -1 ? "face" : { seat: 1 - seat, iid: action.target },
    };
  } else if (action?.t === "use") {
    cast = { seat };
  }
  return { lunge, cast, splashes, drawn: null };
}
