// generate.js
/**
 * XARVIS AI — GENERATE FETCH LAYER
 * Handles all content generation tool requests.
 * Fixed: real error propagation, consistent error format.
 */

import { CONFIG } from "./config.js";

// ─────────────────────────────────────────────────────────────
// SHARED GENERATE CALLER
// All tool functions route through here
// ─────────────────────────────────────────────────────────────
async function callGenerate(payload) {
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);

  try {
    const res = await fetch(`${CONFIG.API_BASE}${CONFIG.ROUTES.GENERATE}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
      signal:  controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }

    const data = await res.json();

    if (!data.reply) {
      throw new Error("Empty response from server");
    }

    return data.reply;

  } catch (err) {
    clearTimeout(timeoutId);

    if (err.name === "AbortError") {
      throw new Error("Request timed out. The server may be waking up — try again in 30 seconds.");
    }

    throw err;
  }
}

// ─────────────────────────────────────────────────────────────
// NAMED EXPORTS — one per tool
// ─────────────────────────────────────────────────────────────
export const generateViral    = (topic, platform, memory) =>
  callGenerate({ type: "viral",    topic, platform, memory });

export const generatePostNext = (memory) =>
  callGenerate({ type: "postnext", memory });

export const generateCalendar = (memory) =>
  callGenerate({ type: "calendar", memory });

export const analyzeFeedback  = (content, memory) =>
  callGenerate({ type: "feedback", content, memory });
