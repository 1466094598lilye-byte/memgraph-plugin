/**
 * MemGraph Plugin — Fully local, embedding-driven long-term memory
 *
 * No external server needed. All logic runs in-process:
 *   - Embedding: local ONNX model (all-MiniLM-L6-v2) via Transformers.js
 *   - Storage: JSON file persistence
 *   - Memo extraction: cheap LLM API (DeepSeek) — optional
 *   - Recall: cosine similarity + focus decay, only from past sessions
 *
 * Anti-token-explosion:
 *   - Circular ingestion prevention (strip recall tags before ingest)
 *   - Quality gate (skip low-value agent chatter)
 *   - Session filtering (don't recall current session — it's in context)
 *   - Recall cooldown (30s) + query cache
 */

import { embed } from "./lib/embedder.js";
import { addTurn, totalTurns } from "./lib/store.js";
import { extractMemos } from "./lib/memo.js";
import { checkContext, recall } from "./lib/recall.js";
import { stripRecallTags, combineTurnText } from "./lib/utils.js";

// ── State ──
const contextTurnIds = new Map<string, Set<number>>();
const pendingPrompt = new Map<string, string>();

// ── Recall cooldown + cache ──
const RECALL_COOLDOWN_MS = 30_000; // 30 seconds
const lastRecallTime = new Map<string, number>();
const recallCache = new Map<string, { query: string; result: string; time: number }>();

function getContextSet(sessionKey: string): Set<number> {
  if (!contextTurnIds.has(sessionKey)) {
    contextTurnIds.set(sessionKey, new Set());
  }
  return contextTurnIds.get(sessionKey)!;
}

// ── Ingest quality gate ──

const MIN_MEANINGFUL_LENGTH = 40;

const NOISE_PATTERNS: RegExp[] = [
  /^```[\s\S]*```$/,                       // pure code block
  /^\s*\{[\s\S]*\}\s*$/,                   // raw JSON blob
  /^(ok|done|sure|got it|understood|好的|明白|收到|完成|好|是的|没问题)\s*[.!。！]?\s*$/i,
  /^(let me|i'll|i will|让我|我来|接下来|我先)/i,
  /^(reading|searching|checking|looking|running|executing)/i,
  /^\$ .+/,                                 // shell command
  /^(Error|Warning|ECONNREFUSED|TypeError|SyntaxError)/i,
];

function isWorthIngesting(userText: string, assistantText: string): boolean {
  const combined = `${userText} ${assistantText}`.trim();
  if (combined.length < MIN_MEANINGFUL_LENGTH) return false;

  for (const pat of NOISE_PATTERNS) {
    if (pat.test(userText.trim()) || pat.test(assistantText.trim())) {
      return false;
    }
  }

  const codeBlockContent = assistantText.match(/```[\s\S]*?```/g)?.join("") || "";
  if (codeBlockContent.length > 0 && codeBlockContent.length / assistantText.length > 0.8) {
    return false;
  }

  return true;
}

// ── Plugin ──

export default function register(api: any) {
  const log = api.logger;

  // ── Tool: memgraph_recall (manual, always available) ──

  api.registerTool(
    (ctx: any) => {
      return {
        name: "memgraph_recall",
        description:
          "Query long-term memory from previous sessions. Use to recall facts, preferences, decisions, or conversations from past sessions no longer in context.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Natural language query for what to recall from past sessions.",
            },
            top_k: {
              type: "number",
              description: "Number of relevant turns to retrieve (default: 10)",
              minimum: 1,
              maximum: 50,
            },
          },
          required: ["query"],
        },
        async handler(params: { query: string; top_k?: number }) {
          try {
            const sessionId = ctx.sessionKey || ctx.sessionId || "default";
            const text = await recall(params.query, sessionId, params.top_k ?? 10);
            return {
              content: [
                {
                  type: "text" as const,
                  text: text || "(no memories found)",
                },
              ],
            };
          } catch (err: any) {
            return {
              content: [{ type: "text" as const, text: `MemGraph error: ${err?.message || err}` }],
            };
          }
        },
      };
    },
    { names: ["memgraph_recall"], optional: true },
  );

  // ── Hook: cache user prompt from llm_input ──

  api.on("llm_input", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || event.sessionId || "default";
    if (event.prompt) {
      pendingPrompt.set(key, event.prompt);
    }
  });

  // ── Hook: auto ingest after each LLM output ──

  api.on("llm_output", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || event.sessionId || "default";
    let userText = pendingPrompt.get(key) || "";
    pendingPrompt.delete(key);

    const assistantTexts: string[] = event.assistantTexts || [];
    let assistantText = assistantTexts.join("\n");

    if (!userText && !assistantText) return;
    if (assistantTexts.every((t: string) => !t.trim())) return;
    if (!userText.trim()) return;

    // Strip recalled memory to prevent circular ingestion
    userText = stripRecallTags(userText);
    assistantText = stripRecallTags(assistantText);
    if (!userText && !assistantText) return;

    // Quality gate: skip low-value agent chatter
    if (!isWorthIngesting(userText, assistantText)) {
      log.info(`[memgraph] skipped ingest — low-value content`);
      return;
    }

    try {
      // Generate embedding for this turn
      const combined = combineTurnText(userText, assistantText);
      const embedding = await embed(combined);

      // Store the turn
      const sessionId = ctx.sessionKey || ctx.sessionId || "default";
      const turnId = addTurn(userText, assistantText, embedding, sessionId);

      // Track in context
      getContextSet(key).add(turnId);

      log.info(`[memgraph] ingested turn ${turnId}, total=${totalTurns()}`);

      // Fire-and-forget memo extraction (async, non-blocking)
      extractMemos(userText, assistantText, turnId).catch(() => {});
    } catch (err: any) {
      log.warn(`[memgraph] ingest failed: ${err?.message || err}`);
    }
  });

  // ── Hook: clear context window on new session ──

  api.on("session_start", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || event.sessionKey || "default";
    contextTurnIds.set(key, new Set());
    lastRecallTime.delete(key);
    recallCache.delete(key);
    log.info(`[memgraph] session_start → cleared context for ${key}`);
  });

  // ── Hook: clear context window after compaction ──

  api.on("after_compaction", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || "default";
    contextTurnIds.set(key, new Set());
    log.info(`[memgraph] after_compaction → cleared context for ${key}`);
  });

  // ── Hook: smart recall with cooldown + session filtering ──

  api.on("before_prompt_build", async (event: any, ctx: any) => {
    const key = ctx.sessionKey || "default";
    const query = event.prompt || "";
    if (!query) return;

    const now = Date.now();

    // Cooldown: skip if last recall was < 30s ago
    const lastTime = lastRecallTime.get(key) || 0;
    if (now - lastTime < RECALL_COOLDOWN_MS) {
      return;
    }

    // Cache: if same query, reuse result
    const cached = recallCache.get(key);
    if (cached && cached.query === query && now - cached.time < RECALL_COOLDOWN_MS * 4) {
      if (cached.result) {
        return {
          prependContext: `<memgraph_long_term_memory>\n${cached.result}\n</memgraph_long_term_memory>`,
        };
      }
      return;
    }

    const sessionId = ctx.sessionKey || ctx.sessionId || "default";
    const ctxSet = getContextSet(key);

    try {
      // Check if recall is needed (any top matches from other sessions?)
      const check = await checkContext(query, sessionId, ctxSet, 5);

      if (!check.needsRecall) {
        lastRecallTime.set(key, now);
        recallCache.set(key, { query, result: "", time: now });
        return;
      }

      log.info(`[memgraph] Smart recall triggered: ${check.reason}`);

      // Full recall — only past sessions
      const text = await recall(query, sessionId, 10);
      lastRecallTime.set(key, now);
      recallCache.set(key, { query, result: text, time: now });

      if (!text) return;

      log.info(`[memgraph] Injected ${text.length} chars of long-term memory`);

      return {
        prependContext: `<memgraph_long_term_memory>\n${text}\n</memgraph_long_term_memory>`,
      };
    } catch (err: any) {
      log.warn(`[memgraph] smart recall failed: ${err?.message || err}`);
    }
  });
}
