// server.js
/**
 * XARVIS AI — SERVER v4.1
 *
 * Changes from v4.0:
 * - Added /api/agent/plan route (was 404, causing silent fallback)
 * - Fixed /api/chat/stream to use Groq's real streaming API
 *   (previously faked streaming with character-by-character setTimeout)
 * - Added /api/health GET (was missing, caused frontend ping errors)
 * - Extracted prompt builders into a PROMPTS object for easy editing
 * - Added consistent error format: { success: false, error: string }
 * - Kept all existing routes intact — no breaking changes
 */

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import Groq from "groq-sdk";

dotenv.config();

const app = express();

// ─────────────────────────────────────────────
// MIDDLEWARE
// ─────────────────────────────────────────────
app.use(express.json());
app.use(cors({
  origin: "*",
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"],
}));

// ─────────────────────────────────────────────
// STARTUP CHECKS
// ─────────────────────────────────────────────
console.log("🚀 Starting Xarvis AI Server v4.1...");
console.log("✅ API KEY EXISTS:", !!process.env.GROQ_API_KEY);

if (!process.env.GROQ_API_KEY) {
  console.error("❌ FATAL: Missing GROQ_API_KEY in environment");
  process.exit(1);
}

// ─────────────────────────────────────────────
// GROQ CLIENT
// ─────────────────────────────────────────────
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const MODEL = "llama-3.3-70b-versatile";

// ─────────────────────────────────────────────
// SYSTEM PROMPT
// ─────────────────────────────────────────────
const SYSTEM_PROMPT = `You are Xarvis AI — a powerful AI co-founder for creators, entrepreneurs, and builders.

Your personality:
- Direct, intelligent, and strategic
- High-energy but concise
- Always actionable — no fluff, no filler

When giving plans or roadmaps, use clear numbered sections and bullet points.
When giving scripts or hooks, format them cleanly so they can be copy-pasted.`;

// ─────────────────────────────────────────────
// PROMPT BUILDERS
// All prompt logic lives here — easy to edit
// without touching route handlers
// ─────────────────────────────────────────────
const PROMPTS = {

  viral({ topic, platform = "YouTube Shorts", memory = {} }) {
    return `Create a viral ${platform} content package for this idea: "${topic}"
${memory.niche ? `Creator niche: ${memory.niche}` : ""}
${memory.tone ? `Content tone: ${memory.tone}` : ""}

Include ALL of the following:

🎣 HOOK (first 3 seconds)
[A scroll-stopping opening line]

📌 TITLE OPTIONS (3 variations)
[A/B/C titles optimised for click-through]

📝 FULL SCRIPT
[Complete script with timestamps and transitions]

🖼️ THUMBNAIL CONCEPT
[Visual description: what to show, text overlay, colour palette]

📊 RETENTION STRATEGY
[3 techniques to keep viewers watching to the end]

Be specific. Write like a top creator, not a marketing bot.`;
  },

  postnext({ memory = {} }) {
    const niche    = memory.niche    || "general content creation";
    const platform = memory.platform || "YouTube Shorts";
    const goal     = memory.goal     || "grow audience";
    const tone     = memory.tone     || "engaging";

    return `You are a viral content strategist. Based on current trends, pick the SINGLE best post idea for:

Niche: ${niche}
Platform: ${platform}
Goal: ${goal}
Tone: ${tone}

Output format:
📌 WINNING IDEA: [one clear title]
🎯 WHY NOW: [why this will perform today specifically]
🎣 HOOK: [opening line]
📋 ANGLE: [unique perspective that separates this from similar content]
⚡ QUICK TIPS: [3 execution tips]

Be direct. One idea. Make it count.`;
  },

  calendar({ memory = {} }) {
    const niche    = memory.niche    || "general";
    const platform = memory.platform || "YouTube Shorts";
    const goal     = memory.goal     || "grow and engage";
    const tone     = memory.tone     || "varied";

    return `Create a 7-day content calendar for a ${niche} creator on ${platform}.
Goal this week: ${goal}
Content tone: ${tone}

For each day include:
- 📅 Day + Topic
- 🎣 Hook/Opening line
- 📐 Format (tutorial/story/reaction/etc)
- ⏱️ Ideal length
- 💡 One differentiation tip

Make every day distinct. No repetitive formats back-to-back.`;
  },

  feedback({ content, memory = {} }) {
    return `Analyze this content idea or script and provide detailed feedback:

"${content}"

${memory.niche ? `Creator niche: ${memory.niche}` : ""}

Score it and explain each category:

📊 VIRALITY SCORE: X/10
📈 Hook Strength: X/10 — [why]
🔁 Shareability: X/10 — [why]
🎯 Niche Fit: X/10 — [why]
⏱️ Retention Potential: X/10 — [why]

🚨 TOP 3 WEAKNESSES
[Specific problems, not generic advice]

✅ HOW TO FIX THEM
[Concrete rewrites or adjustments]

💎 UPGRADED VERSION
[Rewrite the hook/title to be stronger]`;
  },

  agent({ goal, memory = {} }) {
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
  },
};

// ─────────────────────────────────────────────
// GROQ HELPERS
// ─────────────────────────────────────────────

/**
 * Non-streaming Groq call.
 * Returns the full reply string.
 */
async function askGroq(messages, maxTokens = 1200) {
  const completion = await groq.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      ...messages,
    ],
    temperature: 0.72,
    max_tokens: maxTokens,
  });

  return completion?.choices?.[0]?.message?.content || "No response generated.";
}

/**
 * Streaming Groq call.
 * Writes SSE events directly to the Express response object.
 * Caller is responsible for setting SSE headers before calling this.
 *
 * Event format:
 *   data: {"type":"delta","content":"..."}
 *   data: {"type":"done"}
 *   data: {"type":"error","message":"..."}
 */
async function streamGroq(messages, res, maxTokens = 1200) {
  const stream = await groq.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      ...messages,
    ],
    temperature: 0.72,
    max_tokens: maxTokens,
    stream: true,
  });

  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content;
    if (delta) {
      res.write(`data: ${JSON.stringify({ type: "delta", content: delta })}\n\n`);
    }
  }

  res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
  res.end();
}

// ─────────────────────────────────────────────
// TEST GROQ ON STARTUP
// ─────────────────────────────────────────────
async function testGroq() {
  try {
    const res = await groq.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: "Say: Groq OK" }],
      max_tokens: 10,
    });
    console.log("✅ Groq connected:", res?.choices?.[0]?.message?.content);
  } catch (err) {
    console.error("❌ FATAL: Groq connection failed:", err.message);
    process.exit(1);
  }
}

await testGroq();

// ─────────────────────────────────────────────
// ROUTES — HEALTH
// ─────────────────────────────────────────────
app.get("/", (req, res) => {
  res.json({ status: "online", message: "🚀 Xarvis AI v4.1" });
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true, model: MODEL, timestamp: new Date().toISOString() });
});

// ─────────────────────────────────────────────
// ROUTES — CHAT (non-streaming)
// ─────────────────────────────────────────────
app.post("/api/chat", async (req, res) => {
  try {
    const { message, history = [], systemPrompt } = req.body;

    if (!message?.trim()) {
      return res.status(400).json({ success: false, error: "message is required" });
    }

    const messages = [
      ...(systemPrompt
        ? [
            { role: "user",      content: `[Context for this session: ${systemPrompt}]` },
            { role: "assistant", content: "Got it, I'll keep that context in mind." },
          ]
        : []),
      ...history.slice(-10),
      { role: "user", content: message },
    ];

    const reply = await askGroq(messages, 900);
    res.json({ success: true, reply });

  } catch (err) {
    console.error("❌ /api/chat error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────
// ROUTES — CHAT (streaming via SSE)
// ─────────────────────────────────────────────
app.post("/api/chat/stream", async (req, res) => {
  try {
    const { message, history = [] } = req.body;

    if (!message?.trim()) {
      return res.status(400).json({ success: false, error: "message is required" });
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    const messages = [
      ...history.slice(-10),
      { role: "user", content: message },
    ];

    await streamGroq(messages, res, 900);

  } catch (err) {
    console.error("❌ /api/chat/stream error:", err.message);
    if (!res.headersSent) {
      res.setHeader("Content-Type", "text/event-stream");
    }
    res.write(`data: ${JSON.stringify({ type: "error", message: err.message })}\n\n`);
    res.end();
  }
});

// ─────────────────────────────────────────────
// ROUTES — GENERATE
// ─────────────────────────────────────────────
app.post("/api/generate", async (req, res) => {
  try {
    const { type, topic, platform, content, goal, memory = {}, prompt } = req.body;

    let userPrompt;

    switch (type) {
      case "viral":
        userPrompt = PROMPTS.viral({ topic: topic || prompt, platform, memory });
        break;
      case "postnext":
        userPrompt = PROMPTS.postnext({ memory });
        break;
      case "calendar":
        userPrompt = PROMPTS.calendar({ memory });
        break;
      case "feedback":
        userPrompt = PROMPTS.feedback({ content: content || prompt, memory });
        break;
      case "agent":
      case "agent_plan":
        userPrompt = PROMPTS.agent({ goal: goal || prompt, memory });
        break;
      default:
        userPrompt = prompt || topic || goal || "Help me with content creation.";
    }

    const reply = await askGroq([{ role: "user", content: userPrompt }], 1400);
    res.json({ success: true, reply });

  } catch (err) {
    console.error("❌ /api/generate error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────
// ROUTES — AGENT PLAN
// ─────────────────────────────────────────────
app.post("/api/agent/plan", async (req, res) => {
  try {
    const { goal, memory = {}, niche, platform } = req.body;

    if (!goal?.trim()) {
      return res.status(400).json({ success: false, error: "goal is required" });
    }

    const mergedMemory = {
      niche:    niche    || memory.niche    || "",
      platform: platform || memory.platform || "YouTube Shorts",
      goal:     memory.goal  || "",
      tone:     memory.tone  || "",
    };

    const userPrompt = PROMPTS.agent({ goal, memory: mergedMemory });
    const reply = await askGroq([{ role: "user", content: userPrompt }], 1600);
    res.json({ success: true, reply });

  } catch (err) {
    console.error("❌ /api/agent/plan error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─────────────────────────────────────────────
// 404
// ─────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ success: false, error: `Route not found: ${req.method} ${req.path}` });
});

// ─────────────────────────────────────────────
// START
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Xarvis AI v4.1 running on port ${PORT}`);
  console.log(`   Health: http://localhost:${PORT}/api/health`);
});
