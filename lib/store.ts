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

const DATA_DIR = join(dirname(__dirname), "data");
const STORE_PATH = join(DATA_DIR, "store.json");

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
  ensureLoaded();
  const turn_id = store.turns.length;
  store.turns.push({
    turn_id,
    user_text: userText,
    assistant_text: assistantText,
    embedding,
    session_id: sessionId,
    timestamp: new Date().toISOString(),
  });
  scheduleSave();
  return turn_id;
}

/** Add memo entries */
export function addMemos(entries: MemoEntry[]) {
  ensureLoaded();
  for (const entry of entries) {
    // Deduplicate by key — update existing if same key
    const existing = store.memos.findIndex((m) => m.key === entry.key);
    if (existing >= 0) {
      store.memos[existing] = entry;
    } else {
      store.memos.push(entry);
    }
  }
  scheduleSave();
}

/** Get all turns */
export function getAllTurns(): Turn[] {
  ensureLoaded();
  return store.turns;
}

/** Get all memos */
export function getAllMemos(): MemoEntry[] {
  ensureLoaded();
  return store.memos;
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
