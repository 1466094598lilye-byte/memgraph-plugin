/**
 * Utility functions for MemGraph plugin
 */

/** Cosine similarity between two vectors of equal length */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`cosineSimilarity: vector length mismatch (${a.length} vs ${b.length})`);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/** Strip <memgraph_long_term_memory> tags and their content (handles nested tags) */
export function stripRecallTags(text: string): string {
  let result = text;
  // Match innermost tags first (no nested open tags inside), repeat until clean
  const innerTag = /<memgraph_long_term_memory>[^<]*(?:<(?!memgraph_long_term_memory>|\/memgraph_long_term_memory>)[^<]*)*<\/memgraph_long_term_memory>/g;
  let prev = "";
  while (result !== prev) {
    prev = result;
    result = result.replace(innerTag, "");
  }
  // Clean up orphaned unclosed opening tags — strip everything after them
  result = result.replace(/<memgraph_long_term_memory>[\s\S]*/g, "");
  // Clean up orphaned closing tags
  result = result.replace(/<\/memgraph_long_term_memory>/g, "");
  return result.trim();
}

/** Combine user + assistant text for embedding */
export function combineTurnText(userText: string, assistantText: string): string {
  return `${userText}\n${assistantText}`.trim();
}
