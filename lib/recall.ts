/**
 * Recall engine — cosine search + focus decay + session filtering
 * Only recalls memories from OTHER sessions (current session is in context window).
 */

import { embed } from "./embedder.js";
import { getAllTurns, getAllMemos, type Turn } from "./store.js";
import { cosineSimilarity } from "./utils.js";

// ── Focus decay ──
// Boost recent turns, decay older ones. Half-life = 50 turns.
const DECAY_HALF_LIFE = 50;

function focusDecay(turnId: number, maxTurnId: number): number {
  const age = maxTurnId - turnId;
  return Math.pow(0.5, age / DECAY_HALF_LIFE);
}

interface ScoredTurn {
  turn: Turn;
  score: number; // cosine * focus_decay
}

/**
 * Check if a query points to turns outside the current context.
 * Returns true only if top matches belong to OTHER sessions.
 */
// Quick pre-filter: skip embedding if query is clearly not about past context
const RECALL_HINT_PATTERNS = [
  /之前|上次|以前|记得|记忆|过去|历史/,        // Chinese
  /before|previous|last time|remember|recall|earlier|history|past/i,  // English
  /\?$|？$/,                                    // questions are more likely to need recall
];

function queryMightNeedRecall(query: string): boolean {
  // Short operational commands never need recall
  if (query.length < 20) return false;
  // Explicit recall hints always qualify
  if (RECALL_HINT_PATTERNS.some((p) => p.test(query))) return true;
  // Longer queries (>100 chars) might be conversational — let them through
  if (query.length > 100) return true;
  // Default: skip — saves an embed() call
  return false;
}

export async function checkContext(
  query: string,
  currentSessionId: string,
  contextTurnIds: Set<number>,
  topK: number = 5,
): Promise<{ needsRecall: boolean; reason: string }> {
  // Pre-filter: avoid expensive embed() call for operational queries
  if (!queryMightNeedRecall(query)) {
    return { needsRecall: false, reason: "query doesn't suggest past context needed" };
  }

  const turns = getAllTurns();
  if (turns.length === 0) {
    return { needsRecall: false, reason: "no turns stored" };
  }

  const queryEmb = await embed(query);
  const maxTurnId = turns.length - 1;

  // Score all turns
  const scored: ScoredTurn[] = turns.map((turn) => ({
    turn,
    score: cosineSimilarity(queryEmb, turn.embedding) * focusDecay(turn.turn_id, maxTurnId),
  }));

  // Top-k by score
  scored.sort((a, b) => b.score - a.score);
  const topMatches = scored.slice(0, topK);

  // Only consider matches with strong similarity — 0.3 is noise floor,
  // 0.55 filters out "vaguely related" matches that waste tokens
  const MIN_SIMILARITY = 0.55;
  const meaningfulMatches = topMatches.filter((s) => s.score >= MIN_SIMILARITY);

  if (meaningfulMatches.length === 0) {
    return { needsRecall: false, reason: "no meaningful matches above threshold" };
  }

  // Check: do any meaningful matches come from other sessions AND are not in context?
  const outsideContext = meaningfulMatches.filter(
    (s) => s.turn.session_id !== currentSessionId && !contextTurnIds.has(s.turn.turn_id),
  );

  if (outsideContext.length === 0) {
    return { needsRecall: false, reason: "all top matches are in current session or context" };
  }

  return {
    needsRecall: true,
    reason: `${outsideContext.length}/${topK} top matches from other sessions`,
  };
}

/**
 * Full recall: retrieve relevant memories from past sessions.
 * Excludes current session's turns (they're already in the context window).
 */
export async function recall(
  query: string,
  currentSessionId: string,
  topK: number = 10,
): Promise<string> {
  const turns = getAllTurns();
  const memos = getAllMemos();

  if (turns.length === 0 && memos.length === 0) {
    return "";
  }

  // ── Memo section (always included, small and critical) ──
  let memoSection = "";
  if (memos.length > 0) {
    const memoLines = memos.map((m) => `- ${m.key}: ${m.value}`);
    memoSection = `[Key Facts]\n${memoLines.join("\n")}`;
  }

  // ── Turn recall (only from other sessions) ──
  const otherSessionTurns = turns.filter((t) => t.session_id !== currentSessionId);

  if (otherSessionTurns.length === 0) {
    return memoSection; // Only memos, no past session turns
  }

  const queryEmb = await embed(query);
  const maxTurnId = turns.length - 1;

  const scored: ScoredTurn[] = otherSessionTurns.map((turn) => ({
    turn,
    score: cosineSimilarity(queryEmb, turn.embedding) * focusDecay(turn.turn_id, maxTurnId),
  }));

  scored.sort((a, b) => b.score - a.score);
  const topTurns = scored.slice(0, topK);

  // Sort by time for readable output
  topTurns.sort((a, b) => a.turn.turn_id - b.turn.turn_id);

  const turnLines = topTurns.map((s) => {
    const t = s.turn;
    return `[Turn ${t.turn_id} | ${t.session_id}]\nUser: ${t.user_text}\nAssistant: ${t.assistant_text}`;
  });

  const turnSection = turnLines.length > 0 ? `[Relevant Past Turns]\n${turnLines.join("\n\n")}` : "";

  return [memoSection, turnSection].filter(Boolean).join("\n\n");
}
