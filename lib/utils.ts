/**
 * Utility functions for MemGraph plugin
 */

/** Cosine similarity between two vectors of equal length */
export function cosineSimilarity(a: number[], b: number[]): number {
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

/** Strip <memgraph_long_term_memory> tags and their content */
export function stripRecallTags(text: string): string {
  return text
    .replace(/<memgraph_long_term_memory>[\s\S]*?<\/memgraph_long_term_memory>/g, "")
    .trim();
}

/** Combine user + assistant text for embedding */
export function combineTurnText(userText: string, assistantText: string): string {
  return `${userText}\n${assistantText}`.trim();
}
