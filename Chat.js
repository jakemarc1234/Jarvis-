// chat.js
/**
 * XARVIS AI — CHAT FETCH LAYER
 * Handles streaming and non-streaming chat requests.
 * Fixed: history passing, SSE parsing, real error propagation.
 */

import { CONFIG } from "./config.js";

// ─────────────────────────────────────────────────────────────
// SAFE FETCH — timeout + smart retry
// Only retries on genuine network failures, NOT on 4xx/5xx
// ─────────────────────────────────────────────────────────────
async function fetchWithRetry(url, options = {}, retries = CONFIG.RETRY_ATTEMPTS) {
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);

  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let errorMessage = `Server error ${res.status}`;
      try {
        const data = await res.json();
        errorMessage = data?.error || errorMessage;
      } catch {
        // Response wasn't JSON (Render may return HTML on 502/503)
      }

      const error = new Error(errorMessage);
      error.status = res.status;
      throw error;
    }

    return res;

  } catch (err) {
    clearTimeout(timeoutId);

    if (err.name === "AbortError") {
      throw new Error(
        "Request timed out. The server may be waking up from sleep — try again in 30 seconds."
      );
    }

    // Retry only on genuine network-level failures (no status = fetch never completed)
    const isNetworkFailure = !err.status && err.message?.toLowerCase().includes("fetch");
    if (isNetworkFailure && retries > 0) {
      console.warn(`[Xarvis] Network failure, retrying... (${retries} left)`);
      await new Promise((r) => setTimeout(r, CONFIG.RETRY_DELAY));
      return fetchWithRetry(url, options, retries - 1);
    }

    throw err;
  }
}

// ─────────────────────────────────────────────────────────────
// SEND CHAT — Non-streaming fallback
// ─────────────────────────────────────────────────────────────
export async function sendChat(message, history = []) {
  const res = await fetchWithRetry(`${CONFIG.API_BASE}${CONFIG.ROUTES.CHAT}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      message,
      history: history.slice(-CONFIG.MAX_HISTORY),
    }),
  });

  const data = await res.json().catch(() => ({}));

  if (!data.reply) {
    throw new Error("Empty response from server");
  }

  return data.reply;
}

// ─────────────────────────────────────────────────────────────
// STREAM CHAT — SSE with robust parsing
// ─────────────────────────────────────────────────────────────
export async function streamChat(message, history = [], onChunk) {
  const res = await fetchWithRetry(`${CONFIG.API_BASE}${CONFIG.ROUTES.STREAM}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      message,
      history: history.slice(-CONFIG.MAX_HISTORY),
    }),
  });

  if (!res.body) {
    throw new Error("Server does not support streaming. Will use standard mode.");
  }

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer    = "";
  let fullText  = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on double-newline (SSE event boundary)
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const event of events) {
        const dataLine = event.split("\n").find((l) => l.startsWith("data: "));
        if (!dataLine) continue;

        const raw = dataLine.slice(6).trim();
        if (!raw || raw === "[DONE]") continue;

        try {
          const parsed = JSON.parse(raw);

          if (parsed.type === "delta" && parsed.content) {
            fullText += parsed.content;
            onChunk?.(fullText);
          }

          if (parsed.type === "done")  return fullText;

          if (parsed.type === "error") {
            throw new Error(parsed.message || "Stream error from server");
          }
        } catch (parseErr) {
          // Only re-throw real errors, not JSON parse failures on malformed chunks
          if (parseErr.message?.includes("Stream error")) throw parseErr;
        }
      }
    }
  } finally {
    try { reader.releaseLock(); } catch { /* already released */ }
  }

  return fullText;
}
