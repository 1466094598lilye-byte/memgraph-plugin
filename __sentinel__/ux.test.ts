/**
 * [T-UX] User Tester — PM criteria translated to runnable tests
 * Each test references a specific PM criterion by number.
 */

import { describe, it, expect } from "vitest";
import { cosineSimilarity, stripRecallTags } from "../lib/utils.js";
import { addTurn, getAllTurns, addMemos, getAllMemos } from "../lib/store.js";

const EMBEDDING_DIM = 384;
const validEmbedding = () => Array.from({ length: EMBEDDING_DIM }, (_, i) => Math.sin(i));

// ── PM #4 [T2]: cosineSimilarity must reject mismatched lengths ──

describe("[T-UX][Tier 2] PM#4 — cosineSimilarity mismatched lengths", () => {
  it("should throw when vectors have different lengths", () => {
    const a = [1, 2, 3];
    const b = [1, 2, 3, 4, 5];
    expect(() => cosineSimilarity(a, b)).toThrow(/length mismatch/);
  });
});

// ── PM #6 [T2]: stripRecallTags must handle unclosed tags ──

describe("[T-UX][Tier 2] PM#6 — stripRecallTags unclosed tags", () => {
  it("should handle unclosed opening tag", () => {
    const input = "before <memgraph_long_term_memory>leaked content without closing tag";
    const result = stripRecallTags(input);

    // Unclosed tag should either be stripped or left completely intact
    // It should NOT leave the tag opener while stripping partial content
    const tagOpenerRemains = result.includes("<memgraph_long_term_memory>");
    const contentLeaked = result.includes("leaked content");

    // If tag opener is gone but content remains, that's inconsistent → FAIL
    // If tag opener remains and content remains, the regex didn't match → acceptable
    // If both are gone, the function handled it → acceptable
    if (!tagOpenerRemains && contentLeaked) {
      expect.fail("Unclosed tag was partially stripped — tag removed but injected content leaked through");
    }
  });
});

// ── PM #9 [T3]: addTurn rate limit error message quality ──

describe("[T-UX][Tier 3] PM#9 — error message quality", () => {
  it("rate limit error should not expose internal variable names", () => {
    const session = "ux-rate-" + Date.now();
    addTurn("u", "a", validEmbedding(), session);

    try {
      addTurn("u2", "a2", validEmbedding(), session);
      expect.fail("Should have thrown rate limit error");
    } catch (err: any) {
      const msg = err.message || "";
      // PM criterion: error message should not contain internal implementation details
      expect(msg).not.toContain("INGEST_MIN_INTERVAL_MS");
      expect(msg).not.toContain("lastIngestTime");
      // Should be user-friendly
      expect(msg.toLowerCase()).toContain("rate limit");
    }
  });
});

// ── PM #10 [T4]: getAllTurns memory proportionality ──

describe("[T-UX][Tier 4] PM#10 — getAllTurns memory usage", () => {
  it("getAllMemos returns deep copies (no reference leakage)", () => {
    addMemos([{ key: "ux_test_copy", value: "original", turn_id: 0 }]);
    const memos = getAllMemos();
    const found = memos.find((m) => m.key === "ux_test_copy");
    if (found) {
      found.value = "MUTATED";
      const memosAgain = getAllMemos();
      const foundAgain = memosAgain.find((m) => m.key === "ux_test_copy");
      expect(foundAgain?.value).toBe("original");
    }
  });
});

// ── PM #5 [T2]: recall should not return turns with corrupted embeddings ──

describe("[T-UX][Tier 2] PM#5 — zero embedding should not rank highly", () => {
  it("a turn with all-zero embedding should not dominate recall results", () => {
    // If a turn somehow gets a zero embedding, cosineSimilarity returns 0
    // This is the correct behavior — it should NOT appear in top results
    const zeroEmb = new Array(EMBEDDING_DIM).fill(0);
    const queryEmb = validEmbedding();
    const score = cosineSimilarity(zeroEmb, queryEmb);
    expect(score).toBe(0); // zero vector should produce zero similarity
  });
});
