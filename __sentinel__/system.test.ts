/**
 * [T-Sys] System Tester — Code Correctness
 * Tests each exported function against its contract.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { cosineSimilarity, stripRecallTags, combineTurnText } from "../lib/utils.js";

// ── utils.ts ──

describe("[T-Sys] cosineSimilarity", () => {
  it("returns 1.0 for identical vectors", () => {
    const v = [1, 0, 0];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0);
  });

  it("returns 0.0 for orthogonal vectors", () => {
    const a = [1, 0, 0];
    const b = [0, 1, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0);
  });

  it("returns -1.0 for opposite vectors", () => {
    const a = [1, 0];
    const b = [-1, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0);
  });

  it("returns 0.0 for zero vector (not NaN)", () => {
    const zero = [0, 0, 0];
    const v = [1, 2, 3];
    expect(cosineSimilarity(zero, v)).toBe(0);
    expect(Number.isNaN(cosineSimilarity(zero, v))).toBe(false);
  });
});

describe("[T-Sys] stripRecallTags", () => {
  it("strips closed memgraph tags", () => {
    const input = "hello <memgraph_long_term_memory>secret</memgraph_long_term_memory> world";
    expect(stripRecallTags(input)).toBe("hello  world");
  });

  it("returns input unchanged when no tags present", () => {
    expect(stripRecallTags("no tags here")).toBe("no tags here");
  });

  it("strips multiple tag occurrences", () => {
    const input = "<memgraph_long_term_memory>a</memgraph_long_term_memory> mid <memgraph_long_term_memory>b</memgraph_long_term_memory>";
    expect(stripRecallTags(input)).toBe("mid");
  });

  it("handles empty string", () => {
    expect(stripRecallTags("")).toBe("");
  });
});

describe("[T-Sys] combineTurnText", () => {
  it("combines user and assistant text with newline", () => {
    expect(combineTurnText("hello", "world")).toBe("hello\nworld");
  });

  it("trims result", () => {
    expect(combineTurnText("", "answer")).toBe("answer");
  });
});
