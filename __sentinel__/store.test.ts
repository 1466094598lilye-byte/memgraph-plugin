/**
 * [T-Sys] System Tester — store.ts correctness
 * Tests addTurn validation, getAllTurns deep copy safety, memo dedup.
 */

import { describe, it, expect } from "vitest";
import { addTurn, getAllTurns, addMemos, getAllMemos, totalTurns } from "../lib/store.js";

const EMBEDDING_DIM = 384;
const validEmbedding = () => Array.from({ length: EMBEDDING_DIM }, (_, i) => Math.sin(i));

describe("[T-Sys] addTurn validation", () => {
  it("rejects embedding of wrong dimension", () => {
    expect(() => addTurn("u", "a", [1, 2, 3], "test-dim")).toThrow(/384-dim/);
  });

  it("rejects embedding with NaN", () => {
    const emb = validEmbedding();
    emb[42] = NaN;
    expect(() => addTurn("u", "a", emb, "test-nan")).toThrow(/NaN/i);
  });

  it("rejects embedding with Infinity", () => {
    const emb = validEmbedding();
    emb[0] = Infinity;
    expect(() => addTurn("u", "a", emb, "test-inf")).toThrow(/Infinity/i);
  });

  it("rejects non-array embedding", () => {
    expect(() => addTurn("u", "a", "not-an-array" as any, "test-type")).toThrow();
  });

  it("accepts valid 384-dim embedding", () => {
    const id = addTurn("user msg", "assistant msg", validEmbedding(), "test-valid-" + Date.now());
    expect(typeof id).toBe("number");
    expect(id).toBeGreaterThanOrEqual(0);
  });

  it("enforces rate limit (2s between calls per session)", () => {
    const session = "test-rate-" + Date.now();
    addTurn("u", "a", validEmbedding(), session);
    // Second call within 2s should throw
    expect(() => addTurn("u2", "a2", validEmbedding(), session)).toThrow(/rate limited/i);
  });
});

describe("[T-Sys] getAllTurns deep copy", () => {
  it("returns a copy — mutating result does not corrupt store", () => {
    const turns = getAllTurns();
    if (turns.length > 0) {
      const originalText = turns[0].user_text;
      turns[0].user_text = "MUTATED";
      const turnsAgain = getAllTurns();
      expect(turnsAgain[0].user_text).toBe(originalText);
    }
  });

  it("embedding arrays are also deep copied", () => {
    const turns = getAllTurns();
    if (turns.length > 0) {
      const original0 = turns[0].embedding[0];
      turns[0].embedding[0] = 999999;
      const turnsAgain = getAllTurns();
      expect(turnsAgain[0].embedding[0]).toBe(original0);
    }
  });
});

describe("[T-Sys] addMemos deduplication", () => {
  it("updates existing memo with same key", () => {
    addMemos([{ key: "test_dedup", value: "first", turn_id: 0 }]);
    addMemos([{ key: "test_dedup", value: "second", turn_id: 1 }]);
    const memos = getAllMemos();
    const found = memos.filter((m) => m.key === "test_dedup");
    expect(found.length).toBe(1);
    expect(found[0].value).toBe("second");
  });
});

describe("[T-Sys] totalTurns", () => {
  it("returns a non-negative number", () => {
    expect(totalTurns()).toBeGreaterThanOrEqual(0);
  });
});
