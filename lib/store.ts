/**
 * In-memory data store with JSON file persistence.
 * Stores turns (with embeddings) and memos (key-value facts).
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";

export interface Turn {
  turn_id: number;
  user_text: string;
  assistant_text: string;
  embedding: number[]; // 384-dim
  session_id: string;
  timestamp: string;
}

export interface MemoEntry {
  key: string;
  value: string;
  turn_id: number;
}

interface StoreData {
  turns: Turn[];
  memos: MemoEntry[];
}

const EMBEDDING_DIM = 384;
const DATA_DIR = join(dirname(__dirname), "data");
const STORE_PATH = join(DATA_DIR, "store.json");

/** Ingest rate limit — min interval between addTurn calls per session */
const INGEST_MIN_INTERVAL_MS = 2_000; // 2 seconds
const lastIngestTime = new Map<string, number>();

let store: StoreData = { turns: [], memos: [] };
let loaded = false;
let saveTimer: ReturnType<typeof setTimeout> | null = null;

function ensureLoaded() {
  if (loaded) return;
  loaded = true;
  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }
  if (existsSync(STORE_PATH)) {
    try {
      const raw = readFileSync(STORE_PATH, "utf-8");
      store = JSON.parse(raw);
    } catch {
      store = { turns: [], memos: [] };
    }
  }
}

/** Debounced save — writes at most once per 500ms */
function scheduleSave() {
  if (saveTimer) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    try {
      if (!existsSync(DATA_DIR)) {
        mkdirSync(DATA_DIR, { recursive: true });
      }
      writeFileSync(STORE_PATH, JSON.stringify(store), "utf-8");
    } catch {
      // silent — persistence failure shouldn't break the plugin
    }
  }, 500);
}

/** Add a turn and return its turn_id */
export function addTurn(
  userText: string,
  assistantText: string,
  embedding: number[],
  sessionId: string,
): number {
  if (!Array.isArray(embedding) || embedding.length !== EMBEDDING_DIM) {
    throw new Error(
      `addTurn: embedding must be a ${EMBEDDING_DIM}-dim array, got ${Array.isArray(embedding) ? embedding.length : typeof embedding}`,
    );
  }
  // Reject NaN / Infinity — one poisoned value corrupts all cosine similarity
  for (let i = 0; i < embedding.length; i++) {
    if (!Number.isFinite(embedding[i])) {
      throw new Error(
        `addTurn: embedding[${i}] is ${embedding[i]} — must be a finite number`,
      );
    }
  }

  // Rate limit — human conversation can't produce hundreds of turns per minute
  const now = Date.now();
  const lastTime = lastIngestTime.get(sessionId) || 0;
  if (now - lastTime < INGEST_MIN_INTERVAL_MS) {
    throw new Error(
      `addTurn: rate limited — min ${INGEST_MIN_INTERVAL_MS}ms between ingests (session: ${sessionId})`,
    );
  }
  lastIngestTime.set(sessionId, now);

  ensureLoaded();
  const turn_id = store.turns.length;
  // Deep copy embedding to prevent external mutation
  const embCopy = Float64Array.from(embedding);
  store.turns.push({
    turn_id,
    user_text: userText,
    assistant_text: assistantText,
    embedding: Array.from(embCopy),
    session_id: sessionId,
    timestamp: new Date().toISOString(),
  });
  scheduleSave();
  return turn_id;
}

/** Normalize memo key — strip zero-width and invisible unicode characters */
function normalizeKey(key: string): string {
  return key.normalize("NFKC").replace(/[\u200B-\u200F\u2028-\u202F\uFEFF]/g, "");
}

/** Add memo entries */
export function addMemos(entries: MemoEntry[]) {
  ensureLoaded();
  for (const entry of entries) {
    entry.key = normalizeKey(entry.key);
    // Deduplicate by normalized key — update existing if same key
    const existing = store.memos.findIndex((m) => normalizeKey(m.key) === entry.key);
    if (existing >= 0) {
      store.memos[existing] = entry;
    } else {
      store.memos.push(entry);
    }
  }
  scheduleSave();
}

/** Get all turns (deep copy — safe from external mutation) */
export function getAllTurns(): Turn[] {
  ensureLoaded();
  return store.turns.map((t) => ({
    ...t,
    embedding: [...t.embedding],
  }));
}

/** Get all memos (deep copy — safe from external mutation) */
export function getAllMemos(): MemoEntry[] {
  ensureLoaded();
  return store.memos.map((m) => ({ ...m }));
}

/** Get total turn count */
export function totalTurns(): number {
  ensureLoaded();
  return store.turns.length;
}

/** Force save (for shutdown) */
export function flush() {
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
  try {
    if (!existsSync(DATA_DIR)) {
      mkdirSync(DATA_DIR, { recursive: true });
    }
    writeFileSync(STORE_PATH, JSON.stringify(store), "utf-8");
  } catch {
    // silent
  }
}
