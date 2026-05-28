// agents.js
/**
 * XARVIS AI — AGENT RUNNER
 * Executes the full planning agent.
 * Tries the dedicated /api/agent/plan endpoint first,
 * falls back to /api/generate if unavailable.
 */

import { CONFIG } from "./config.js";

// ─────────────────────────────────────────────────────────────
// RUN AGENT
// onChunk(fullTextSoFar) is called on each streaming delta
// ─────────────────────────────────────────────────────────────
export async function runAgent(goal, memory = {}, onChunk) {
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);

  // ── ATTEMPT 1: Dedicated agent/plan endpoint (streaming) ──
  try {
    const res = await fetch(`${CONFIG.API_BASE}${CONFIG.ROUTES.STREAM}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        message: buildAgentPrompt(goal, memory),
        history: [],
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok)   throw new Error(`Stream endpoint returned ${res.status}`);
    if (!res.body) throw new Error("No stream body");

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = "";
    let fullText  = "";

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
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
            if (parsed.type === "error") throw new Error(parsed.message || "Stream error");
          } catch (parseErr) {
            if (parseErr.message?.includes("Stream error")) throw parseErr;
          }
        }
      }
    } finally {
      try { reader.releaseLock(); } catch {}
    }

    return fullText;

  } catch (streamErr) {
    clearTimeout(timeoutId);

    if (streamErr.name === "AbortError") {
      throw new Error("Request timed out. Server may be waking up — try again in 30 seconds.");
    }

    console.warn("[Xarvis Agent] Stream failed, falling back to /api/agent/plan:", streamErr.message);
  }

  // ── ATTEMPT 2: Non-streaming fallback via /api/agent/plan ──
  try {
    const res2 = await fetch(`${CONFIG.API_BASE}${CONFIG.ROUTES.AGENT}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ goal, memory }),
    });

    if (!res2.ok) {
      const err = await res2.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res2.status}`);
    }

    const data = await res2.json();
    const text = data.reply || "";
    onChunk?.(text);
    return text;

  } catch (fallbackErr) {
    throw new Error(fallbackErr.message || "Agent failed. Please try again.");
  }
}

// ─────────────────────────────────────────────────────────────
// AGENT PROMPT BUILDER
// ─────────────────────────────────────────────────────────────
function buildAgentPrompt(goal, memory = {}) {
  return `AGENT MODE. Creator goal: "${goal}"
${memory.niche ? `Niche: ${memory.niche} | Platform: ${memory.platform || "YouTube Shorts"}` : ""}

Build a complete, numbered execution roadmap:

🎯 GOAL BREAKDOWN
[3 measurable milestones with deadlines]

📅 PHASE 1 — WEEKS 1-2: Foundation
[5 specific daily actions]

📅 PHASE 2 — WEEKS 3-6: Momentum
[5 specific traction-building actions]

📅 PHASE 3 — WEEKS 7-12: Scale
[5 acceleration actions]

⚡ QUICK WINS (Do in next 24 hours)
[3 things to do TODAY]

🚧 BIGGEST OBSTACLES & HOW TO BEAT THEM
[Top 3 with specific countermeasures]

💰 MONETIZATION PATH
[Revenue streams with realistic timelines]

Be extremely specific. No fluff. This is their blueprint.`;
}
