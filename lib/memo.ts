/**
 * Memo extraction — calls a cheap LLM (DeepSeek) to extract key-value facts
 * from conversation turns. Async fire-and-forget, failures are silent.
 */

import { addMemos, type MemoEntry } from "./store.js";

const API_KEY = process.env.MEMGRAPH_LLM_API_KEY || process.env.OPENAI_API_KEY || "";
const BASE_URL =
  process.env.MEMGRAPH_LLM_BASE_URL ||
  process.env.OPENAI_BASE_URL ||
  "https://api.deepseek.com";
const MODEL = process.env.MEMGRAPH_LLM_MODEL || "deepseek-chat";

const EXTRACT_PROMPT = `You are a memory extraction assistant. Given a conversation turn (user message + assistant response), extract key facts worth remembering long-term.

Output a JSON array of objects with "key" and "value" fields. Keys should be short identifiers (e.g., "user_name", "project_tech_stack", "deadline"). Values should be concise facts.

Rules:
- Only extract concrete, specific facts (names, numbers, dates, decisions, preferences)
- Skip vague or generic information
- Skip implementation details and code specifics
- If nothing worth extracting, return an empty array []
- Output ONLY the JSON array, no other text`;

/**
 * Extract memos from a conversation turn.
 * Fire-and-forget — does not throw on failure.
 */
export async function extractMemos(
  userText: string,
  assistantText: string,
  turnId: number,
): Promise<void> {
  if (!API_KEY) return; // No API key configured, skip memo extraction

  const turnContent = `User: ${userText}\nAssistant: ${assistantText}`;

  try {
    const resp = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: "system", content: EXTRACT_PROMPT },
          { role: "user", content: turnContent },
        ],
        temperature: 0,
        max_tokens: 512,
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!resp.ok) return;

    const data = await resp.json();
    const content = data.choices?.[0]?.message?.content?.trim();
    if (!content) return;

    // Parse JSON array from response (handle markdown code blocks)
    const jsonStr = content.replace(/^```json?\n?/i, "").replace(/\n?```$/, "");
    const entries: Array<{ key: string; value: string }> = JSON.parse(jsonStr);

    if (!Array.isArray(entries) || entries.length === 0) return;

    const memos: MemoEntry[] = entries
      .filter((e) => e.key && e.value)
      .map((e) => ({
        key: String(e.key),
        value: String(e.value),
        turn_id: turnId,
      }));

    if (memos.length > 0) {
      addMemos(memos);
    }
  } catch {
    // Silent failure — memo extraction is best-effort
  }
}
