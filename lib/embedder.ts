/**
 * Local embedding using Transformers.js (ONNX backend)
 * Model: all-MiniLM-L6-v2 — 384-dim sentence embeddings
 * Lazy-loaded on first call.
 */

let extractor: any = null;

async function getExtractor() {
  if (extractor) return extractor;
  // Dynamic import to avoid blocking plugin load
  const { pipeline } = await import("@huggingface/transformers");
  extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
    // Use local cache, quantized for speed
    quantized: true,
  });
  return extractor;
}

/** Generate a 384-dim embedding for a text string */
export async function embed(text: string): Promise<number[]> {
  const ext = await getExtractor();
  const output = await ext(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array).slice(0, 384);
}

/** Batch embed multiple texts */
export async function embedBatch(texts: string[]): Promise<number[][]> {
  const ext = await getExtractor();
  const results: number[][] = [];
  for (const text of texts) {
    const output = await ext(text, { pooling: "mean", normalize: true });
    results.push(Array.from(output.data as Float32Array).slice(0, 384));
  }
  return results;
}
