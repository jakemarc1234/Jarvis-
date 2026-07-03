/**
 * XARVIS AI — SERVER v4.2
 *
 * Changes from v4.1:
 * - Added Viral Clip Finder pipeline (Phase 1: analysis only, no export)
 *   Upload -> extract audio (ffmpeg) -> transcribe (Groq Whisper) ->
 *   analyze (Claude) -> structured JSON -> poll for result
 * - New deps: multer, fluent-ffmpeg, @ffmpeg-installer/ffmpeg, @anthropic-ai/sdk
 * - New in-memory job store (Map) — NOT persistent, Phase 1 only
 * - Everything from v4.1 is unchanged
 */

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import Groq from "groq-sdk";
import Anthropic from "@anthropic-ai/sdk";
import multer from "multer";
import ffmpeg from "fluent-ffmpeg";
import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
import fs from "fs";
import path from "path";
import os from "os";
import crypto from "crypto";

dotenv.config();

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

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
console.log("🚀 Starting Xarvis AI Server v4.2...");
console.log("✅ GROQ KEY EXISTS:", !!process.env.GROQ_API_KEY);
console.log("✅ ANTHROPIC KEY EXISTS:", !!process.env.ANTHROPIC_API_KEY);

if (!process.env.GROQ_API_KEY) {
  console.error("❌ FATAL: Missing GROQ_API_KEY in environment");
  process.exit(1);
}
if (!process.env.ANTHROPIC_API_KEY) {
  console.error("❌ FATAL: Missing ANTHROPIC_API_KEY in environment");
  process.exit(1);
}

// ─────────────────────────────────────────────
// CLIENTS
// ─────────────────────────────────────────────
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const MODEL = "llama-3.3-70b-versatile";
const CLAUDE_MODEL = "claude-sonnet-4-6"; // update if Anthropic ships a newer default

// ─────────────────────────────────────────────
// SYSTEM PROMPT (existing chat/generate tools — unchanged)
// ─────────────────────────────────────────────
const SYSTEM_PROMPT = `You are Xarvis AI — a powerful AI co-founder for creators, entrepreneurs, and builders.

Your personality:
- Direct, intelligent, and strategic
- High-energy but concise
- Always actionable — no fluff, no filler

When giving plans or roadmaps, use clear numbered sections and bullet points.
When giving scripts or hooks, format them cleanly so they can be copy-pasted.`;

// ─────────────────────────────────────────────
// EXISTING PROMPT BUILDERS (unchanged, collapsed for brevity — keep yours as-is)
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
    const niche = memory.niche || "general content creation";
    const platform = memory.platform || "YouTube Shorts";
    const goal = memory.goal || "grow audience";
    const tone = memory.tone || "engaging";
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
    const niche = memory.niche || "general";
    const platform = memory.platform || "YouTube Shorts";
    const goal = memory.goal || "grow and engage";
    const tone = memory.tone || "varied";
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

  // ── NEW: clip analysis prompt ──
  clipAnalysis({ transcript }) {
    return `You are an expert YouTube Shorts, TikTok, and Instagram Reels editor.
Your job is to read a timestamped transcript from a long-form video and identify the BEST moments that should become short-form clips.

Do NOT summarize the transcript. Think like a creator trying to get millions of views.
Your goal is to find moments that immediately grab attention and keep people watching.

Prioritize clips that contain one or more of these:
- Strong hook
- Controversial opinion
- Emotional moment
- Funny moment
- Valuable lesson
- Story payoff
- Surprise
- Argument or disagreement
- Viral quote
- Curiosity gap
- High attention moment

Avoid: introductions, sponsor reads, filler, repetition, low-energy conversation, long explanations with no payoff.

Rules:
- Start and end timestamps MUST be copied exactly from the transcript's timestamp markers — never estimate or interpolate.
- Clips must be between 15 and 90 seconds unless a moment genuinely requires more time — justify any exception in the reason field.
- If two candidate clips overlap by more than 50% or hinge on the same core moment, keep only the strongest one.
- Return up to 10 clips. If fewer than 10 moments meet a real quality bar, return fewer — do not pad with weak clips.
- "attention_score" (1-100) reflects attention-grabbing potential based on transcript signals, NOT a guarantee of virality.
- "reason" must reference the specific line or moment, not generic commentary.

Return ONLY valid JSON in this exact shape, nothing else — no markdown fences, no preamble:

{
  "clips": [
    {
      "rank": 1,
      "title": "",
      "start": "00:00:00",
      "end": "00:00:00",
      "category": "",
      "reason": "",
      "hook": "",
      "attention_score": 0,
      "confidence": 0
    }
  ]
}

Transcript:
"""
${transcript}
"""`;
  },
};

// ─────────────────────────────────────────────
// GROQ HELPERS (existing — unchanged)
// ─────────────────────────────────────────────
async function askGroq(messages, maxTokens = 1200) {
  const completion = await groq.chat.completions.create({
    model: MODEL,
    messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
    temperature: 0.72,
    max_tokens: maxTokens,
  });
  return completion?.choices?.[0]?.message?.content || "No response generated.";
}

async function streamGroq(messages, res, maxTokens = 1200) {
  const stream = await groq.chat.completions.create({
    model: MODEL,
    messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
    temperature: 0.72,
    max_tokens: maxTokens,
    stream: true,
  });
  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content;
    if (delta) res.write(`data: ${JSON.stringify({ type: "delta", content: delta })}\n\n`);
  }
  res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
  res.end();
}

// ─────────────────────────────────────────────
// CLIP PIPELINE HELPERS
// ─────────────────────────────────────────────
const UPLOAD_DIR = path.join(os.tmpdir(), "xarvis-uploads");
const MAX_UPLOAD_BYTES = 200 * 1024 * 1024; // 200MB cap for Phase 1

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

// In-memory job store. Phase 1 only — resets on server restart.
// Shape: { id, status, stage, error, clips, createdAt }
const jobs = new Map();

function createJob() {
  const id = crypto.randomUUID();
  const job = {
    id,
    status: "processing", // processing | done | error
    stage: "queued",      // queued | extracting_audio | transcribing | analyzing | done
    error: null,
    clips: null,
    createdAt: Date.now(),
  };
  jobs.set(id, job);
  return job;
}

/** Extract audio from a video file to mono 16kHz mp3 (small + Whisper-friendly). */
function extractAudio(videoPath, audioPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .noVideo()
      .audioChannels(1)
      .audioFrequency(16000)
      .audioBitrate("64k")
      .format("mp3")
      .on("end", resolve)
      .on("error", reject)
      .save(audioPath);
  });
}

/** Transcribe audio with Groq's hosted Whisper, requesting word/segment-level timestamps. */
async function transcribeAudio(audioPath) {
  const transcription = await groq.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: "whisper-large-v3",
    response_format: "verbose_json",
    temperature: 0,
  });

  // verbose_json returns { text, segments: [{ start, end, text }, ...] }
  const segments = transcription.segments || [];
  const formatted = segments
    .map((s) => `[${formatTimestamp(s.start)} - ${formatTimestamp(s.end)}] ${s.text.trim()}`)
    .join("\n");

  return formatted || transcription.text || "";
}

function formatTimestamp(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

/** Send the timestamped transcript to Claude and parse structured clip JSON. */
async function analyzeTranscript(transcript) {
  const message = await anthropic.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 4000,
    messages: [{ role: "user", content: PROMPTS.clipAnalysis({ transcript }) }],
  });

  const raw = message.content?.[0]?.text || "";
  const cleaned = raw.replace(/```json|```/g, "").trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (e) {
    throw new Error("Claude returned non-JSON output — could not parse clip results.");
  }

  if (!parsed.clips || !Array.isArray(parsed.clips)) {
    throw new Error("Claude response missing a valid 'clips' array.");
  }

  return parsed.clips;
}

/** Runs the full pipeline in the background; updates the job record as it goes. */
async function runClipPipeline(job, videoPath) {
  const audioPath = videoPath + ".mp3";

  try {
    job.stage = "extracting_audio";
    await extractAudio(videoPath, audioPath);

    job.stage = "transcribing";
    const transcript = await transcribeAudio(audioPath);

    if (!transcript.trim()) {
      throw new Error("Transcription returned no speech content.");
    }

    job.stage = "analyzing";
    const clips = await analyzeTranscript(transcript);

    job.clips = clips;
    job.status = "done";
    job.stage = "done";
  } catch (err) {
    console.error(`❌ Clip pipeline failed [${job.id}]:`, err.message);
    job.status = "error";
    job.error = err.message;
  } finally {
    // Best-effort cleanup — Phase 1 has no persistent storage anyway
    fs.unlink(videoPath, () => {});
    fs.unlink(audioPath, () => {});
  }
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
  res.json({ status: "online", message: "🚀 Xarvis AI v4.2" });
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true, model: MODEL, timestamp: new Date().toISOString() });
});

// ─────────────────────────────────────────────
// ROUTES — CHAT / GENERATE / AGENT (unchanged from v4.1)
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
            { role: "user", content: `[Context for this session: ${systemPrompt}]` },
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
    const messages = [...history.slice(-10), { role: "user", content: message }];
    await streamGroq(messages, res, 900);
  } catch (err) {
    console.error("❌ /api/chat/stream error:", err.message);
    if (!res.headersSent) res.setHeader("Content-Type", "text/event-stream");
    res.write(`data: ${JSON.stringify({ type: "error", message: err.message })}\n\n`);
    res.end();
  }
});

app.post("/api/generate", async (req, res) => {
  try {
    const { type, topic, platform, content, goal, memory = {}, prompt } = req.body;
    let userPrompt;
    switch (type) {
      case "viral": userPrompt = PROMPTS.viral({ topic: topic || prompt, platform, memory }); break;
      case "postnext": userPrompt = PROMPTS.postnext({ memory }); break;
      case "calendar": userPrompt = PROMPTS.calendar({ memory }); break;
      case "feedback": userPrompt = PROMPTS.feedback({ content: content || prompt, memory }); break;
      case "agent":
      case "agent_plan": userPrompt = PROMPTS.agent({ goal: goal || prompt, memory }); break;
      default: userPrompt = prompt || topic || goal || "Help me with content creation.";
    }
    const reply = await askGroq([{ role: "user", content: userPrompt }], 1400);
    res.json({ success: true, reply });
  } catch (err) {
    console.error("❌ /api/generate error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post("/api/agent/plan", async (req, res) => {
  try {
    const { goal, memory = {}, niche, platform } = req.body;
    if (!goal?.trim()) {
      return res.status(400).json({ success: false, error: "goal is required" });
    }
    const mergedMemory = {
      niche: niche || memory.niche || "",
      platform: platform || memory.platform || "YouTube Shorts",
      goal: memory.goal || "",
      tone: memory.tone || "",
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
// ROUTES — VIRAL CLIP FINDER (NEW, Phase 1)
// ─────────────────────────────────────────────

/**
 * POST /api/clips/upload
 * multipart/form-data, field name: "video"
 * Returns immediately with a jobId; processing happens in the background.
 */
app.post("/api/clips/upload", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: "No video file received (field name must be 'video')." });
    }

    const job = createJob();
    res.json({ success: true, jobId: job.id });

    // Fire and forget — client polls /api/clips/status/:jobId
    runClipPipeline(job, req.file.path);

  } catch (err) {
    console.error("❌ /api/clips/upload error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * GET /api/clips/status/:jobId
 * Poll this while status === "processing".
 */
app.get("/api/clips/status/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ success: false, error: "Job not found." });
  }
  res.json({
    success: true,
    status: job.status,
    stage: job.stage,
    error: job.error,
  });
});

/**
 * GET /api/clips/result/:jobId
 * Call once status === "done". Returns the ranked clip array.
 */
app.get("/api/clips/result/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ success: false, error: "Job not found." });
  }
  if (job.status !== "done") {
    return res.status(409).json({ success: false, error: `Job is not finished yet (status: ${job.status}).` });
  }
  res.json({ success: true, clips: job.clips });
});

// ─────────────────────────────────────────────
// 404
// ─────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ success: false, error: `Route not found: ${req.method} ${req.path}` });
});

// Multer error handler (file too large, etc.) — must have 4 args to be recognized by Express
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ success: false, error: `Upload error: ${err.message}` });
  }
  console.error("❌ Unhandled error:", err.message);
  res.status(500).json({ success: false, error: err.message });
});

// ─────────────────────────────────────────────
// START
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Xarvis AI v4.2 running on port ${PORT}`);
  console.log(`   Health: http://localhost:${PORT}/api/health`);
});
