import { describe, expect, it } from "vitest";
import { computeFx } from "./fx";
import type { EventDict } from "./replay";

describe("computeFx", () => {
  it("numeric splash from a unit damage event", () => {
    const events: EventDict[] = [
      { t: "damage", seat: 1, target: 7, amount: 3, fatal: false },
    ];
    const fx = computeFx(events, { t: "attack", a: 0, target: 7 }, 0);
    expect(fx.splashes).toContainEqual({
      seat: 1,
      target: 7,
      amount: 3,
      fatal: false,
    });
  });

  it("fatal splash from unit_died without a damage event (item removal)", () => {
    const events: EventDict[] = [{ t: "unit_died", seat: 1, iid: 7 }];
    const fx = computeFx(events, { t: "use", item: 5, target: 7 }, 0);
    expect(fx.splashes).toContainEqual({
      seat: 1,
      target: 7,
      amount: 0,
      fatal: true,
    });
  });

  it("face splash from a face damage event", () => {
    const events: EventDict[] = [
      { t: "damage", seat: 1, target: "face", amount: 4, fatal: false },
    ];
    const fx = computeFx(events, { t: "attack", a: 0, target: -1 }, 0);
    expect(fx.splashes).toContainEqual({
      seat: 1,
      target: "face",
      amount: 4,
      fatal: false,
    });
  });

  it("lunge toward an enemy minion derives from the action", () => {
    const fx = computeFx([], { t: "attack", a: 1, target: 7 }, 0);
    expect(fx.lunge).toEqual({ seat: 0, iid: 1, toward: { seat: 1, iid: 7 } });
  });

  it("lunge toward face when target is -1", () => {
    const fx = computeFx([], { t: "attack", a: 1, target: -1 }, 0);
    expect(fx.lunge).toEqual({ seat: 0, iid: 1, toward: "face" });
  });

  it("cast for use actions; no lunge", () => {
    const fx = computeFx([], { t: "use", item: 5, target: -1 }, 0);
    expect(fx.cast).toEqual({ seat: 0 });
    expect(fx.lunge).toBeNull();
  });

  it("no lunge/cast/splash on pass", () => {
    expect(computeFx([], { t: "pass" }, 0)).toEqual({
      lunge: null,
      cast: null,
      splashes: [],
      drawn: null,
    });
  });

  it("combat-lethal emits a single splash (damage+unit_died deduped)", () => {
    const events: EventDict[] = [
      { t: "damage", seat: 1, target: 7, amount: 5, fatal: true },
      { t: "unit_died", seat: 1, iid: 7 },
    ];
    const fx = computeFx(events, { t: "attack", a: 1, target: 7 }, 0);
    const forSeven = fx.splashes.filter((s) => s.target === 7);
    expect(forSeven).toEqual([{ seat: 1, target: 7, amount: 5, fatal: true }]);
  });
});
