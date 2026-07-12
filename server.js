/**
 * XARVIS AI — SERVER v4.6
 *
 * Changes from v4.5:
 * - FIX: yt-dlp failures only ever showed "Command failed with exit code 1:
 *   <the command>" — the actual reason (YouTube's response) lives in
 *   err.stderr, which was being discarded. extractYtDlpError() now pulls
 *   the real stderr text.
 * - FIX: added the android player-client + mobile user-agent workaround
 *   (YTDLP_COMMON_OPTS) to both metadata and download calls. YouTube's
 *   "Sign in to confirm you're not a bot" check disproportionately blocks
 *   cloud-hosted IPs (Render, AWS, GCP) — this is the most common real
 *   cause of yt-dlp failures on hosted servers, and the android client
 *   skips the browser JS challenge that triggers it.
 * - NEW: friendlyYoutubeError() maps known failure patterns (bot-check,
 *   private/unavailable, age-restricted, removed) to specific, honest
 *   user-facing messages instead of a generic "download failed."
 * - Still Groq-only. yt-dlp is a download tool, not an AI provider.
 */

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import Groq from "groq-sdk";
import multer from "multer";
import ffmpeg from "fluent-ffmpeg";
import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
import ytDlpExec from "yt-dlp-exec";
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
console.log("🚀 Starting Xarvis AI Server v4.6...");
console.log("✅ GROQ KEY EXISTS:", !!process.env.GROQ_API_KEY);

if (!process.env.GROQ_API_KEY) {
  console.error("❌ FATAL: Missing GROQ_API_KEY in environment");
  process.exit(1);
}

// ─────────────────────────────────────────────
// CLIENTS
// ─────────────────────────────────────────────
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const MODEL = "llama-3.3-70b-versatile";
const ANALYSIS_MODEL = process.env.CLIP_ANALYSIS_MODEL || MODEL;

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

  // ── clipAnalysisChunk — FIX: now requests attention_type ──
  clipAnalysisChunk({ chunk }) {
    return `You are a world-class short-form video editor for YouTube Shorts, TikTok, and Instagram Reels.
Your job is to analyze a timestamped transcript chunk from a long-form video and identify ONLY the most viral, high-retention moments.
Do NOT summarize the content.
Think like an editor trying to find moments that will get maximum watch time and engagement.
---
## INPUT
You will receive a transcript with timestamps like:
[00:01:12] text...
[00:01:18] text...
---
## TASK
Find the BEST clip-worthy moments inside this chunk.
Only select moments that would make someone stop scrolling.
Prioritize:
- Strong hooks
- Emotional spikes
- Controversial opinions
- Funny moments
- Big realizations
- Story twists
- High tension or conflict
- Unexpected facts
- Relatable struggles
- Viral quotes
- Curiosity gaps
---
## DO NOT INCLUDE
- intros
- filler talk
- greetings
- explanations with no payoff
- sponsor content
- repeated ideas
---
## ATTENTION TYPE
Classify each clip with EXACTLY one attention_type from this list:
emotional_shift, opinion_change, contradiction, surprise_reveal, conflict_start, punchline, story_twist, curiosity_gap
Pick whichever single category best explains why the moment grabs attention.
---
## OUTPUT RULES
Return ONLY JSON. No markdown fences, no preamble.
Return MAX 3 clips for this chunk.
Start/end timestamps MUST be copied exactly from the chunk's timestamp markers — never estimate or interpolate.
If nothing in this chunk is strong enough, return an empty clips array.
Each clip must include:
- start_time
- end_time
- title (short, viral style)
- hook (first 3 seconds)
- reason (why it will perform)
- attention_type (one of the categories above)
- viral_score (1–100)
- confidence (1–100)
---
## OUTPUT FORMAT
{
  "clips": [
    {
      "start_time": "00:01:12",
      "end_time": "00:01:45",
      "title": "Most people quit too early",
      "hook": "You're probably doing this wrong...",
      "reason": "Strong emotional + relatable failure moment",
      "attention_type": "surprise_reveal",
      "viral_score": 92,
      "confidence": 85
    }
  ]
}
---
Be extremely selective. Only return moments worth turning into short-form content.

Transcript chunk:
"""
${chunk}
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
const CHUNK_DURATION_SECONDS = 5 * 60; // ~5 min windows per analysis call
const JOB_TTL_MS = 30 * 60 * 1000; // 30 min — after this a finished/dead job is swept

// YouTube ingestion caps — long/huge videos would make transcription +
// chunked analysis take unreasonably long for Phase 1, so we cap duration
// up front (checked against yt-dlp's own metadata, before downloading).
const MAX_YOUTUBE_DURATION_SECONDS = 2 * 60 * 60; // 2 hours
const YOUTUBE_URL_PATTERN = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/|live\/)|youtu\.be\/)[\w-]{6,}/i;

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

// In-memory job store. Phase 1 only — resets on server restart/redeploy.
// Shape: { id, status, stage, error, clips, createdAt }
const jobs = new Map();

// Sweep old jobs periodically so a dead/forgotten job doesn't sit in memory
// forever and so a client polling one gets an explicit "expired" error
// instead of an indefinite 404 with no explanation.
setInterval(() => {
  const now = Date.now();
  for (const [id, job] of jobs) {
    if (now - job.createdAt > JOB_TTL_MS) jobs.delete(id);
  }
}, 5 * 60 * 1000).unref();

function createJob() {
  const id = crypto.randomUUID();
  const job = {
    id,
    status: "processing", // processing | done | error
    stage: "queued",      // queued | retrieving_video | extracting_audio | transcribing | analyzing | done
    error: null,
    clips: null,
    createdAt: Date.now(),
  };
  jobs.set(id, job);
  return job;
}

/**
 * yt-dlp-exec throws an error whose top-level .message is just
 * "Command failed with exit code N: <the command that was run>" — the
 * actual reason (YouTube's response, a bad URL, etc.) lives in .stderr.
 * Without this, every failure looked identical and undebuggable.
 */
function extractYtDlpError(err) {
  const stderr = (err && (err.stderr || err.shortMessage)) || "";
  const cleaned = String(stderr)
    .split("\n")
    .filter((line) => line.trim() && !line.trim().startsWith("[debug]"))
    .join(" ")
    .trim();
  return cleaned || err?.message || String(err);
}

// Common workaround for YouTube's bot-check ("Sign in to confirm you're not
// a bot") which disproportionately hits cloud-hosted IPs like Render's —
// the android player client skips the browser JS challenge entirely.
const YTDLP_COMMON_OPTS = {
  noWarnings: true,
  noCheckCertificates: true,
  extractorArgs: "youtube:player_client=android",
  addHeader: ["referer:youtube.com", "user-agent:com.google.android.youtube/19.09.37 (Linux; U; Android 14) gzip"],
};

/**
 * Fetch metadata only (no download) so we can reject unreasonably long
 * videos before spending time/bandwidth on them.
 */
async function fetchYoutubeMetadata(url) {
  try {
    return await ytDlpExec(url, {
      ...YTDLP_COMMON_OPTS,
      dumpSingleJson: true,
      preferFreeFormats: true,
      skipDownload: true,
    });
  } catch (err) {
    throw new Error(extractYtDlpError(err));
  }
}

/**
 * Downloads a YouTube video to disk via yt-dlp, capped to a resolution that
 * keeps file size reasonable (audio is all that matters downstream, but we
 * keep a low-res video stream too so ffmpeg has a normal container to pull
 * audio from).
 */
async function downloadYoutubeVideo(url, outputPath) {
  try {
    await ytDlpExec(url, {
      ...YTDLP_COMMON_OPTS,
      output: outputPath,
      format: "worst[ext=mp4]/worst",
      noPlaylist: true,
      maxFilesize: `${MAX_UPLOAD_BYTES}`,
    });
  } catch (err) {
    throw new Error(extractYtDlpError(err));
  }
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
      .on("error", (err) => {
        const raw = err.message || "";
        const friendly = raw.includes("does not contain any stream") || raw.includes("Invalid data found")
          ? "That file doesn't look like a playable video — it may be corrupted, or not actually a video file. Try a different file."
          : `Audio extraction failed: ${raw}`;
        reject(new Error(friendly));
      })
      .save(audioPath);
  });
}

/**
 * Transcribe audio with Groq's hosted Whisper, requesting segment-level
 * timestamps explicitly. FIX: previously relied on verbose_json alone to
 * include segments — now requests timestamp_granularities: ["segment"]
 * so the requirement isn't implicit/version-dependent.
 */
async function transcribeAudio(audioPath) {
  let transcription;
  try {
    transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(audioPath),
      model: "whisper-large-v3",
      response_format: "verbose_json",
      timestamp_granularities: ["segment"],
      temperature: 0,
    });
  } catch (err) {
    throw new Error(`Transcription failed: ${err.message}`);
  }

  const segments = transcription.segments || [];
  const formatted = segments
    .map((s) => `[${formatTimestamp(s.start)} - ${formatTimestamp(s.end)}] ${s.text.trim()}`)
    .join("\n");

  return { formatted: formatted || transcription.text || "", segments };
}

function formatTimestamp(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

function timestampToSeconds(ts) {
  if (!ts || typeof ts !== "string") return null;
  const parts = ts.split(":").map((v) => parseInt(v, 10));
  if (parts.some((v) => Number.isNaN(v))) return null;
  while (parts.length < 3) parts.unshift(0);
  const [h, m, s] = parts;
  return h * 3600 + m * 60 + s;
}

function chunkTranscript(segments) {
  const chunks = [];
  let current = [];
  let windowStart = 0;

  for (const seg of segments) {
    if (seg.start - windowStart >= CHUNK_DURATION_SECONDS && current.length) {
      chunks.push(current.join("\n"));
      current = [];
      windowStart = seg.start;
    }
    current.push(`[${formatTimestamp(seg.start)} - ${formatTimestamp(seg.end)}] ${seg.text.trim()}`);
  }
  if (current.length) chunks.push(current.join("\n"));

  return chunks;
}

const VALID_ATTENTION_TYPES = new Set([
  "emotional_shift", "opinion_change", "contradiction", "surprise_reveal",
  "conflict_start", "punchline", "story_twist", "curiosity_gap",
]);

/** Run one chunk through Groq and parse its clip JSON. Returns [] on any failure (logged, non-fatal). */
async function analyzeTranscriptChunk(chunk, chunkIndex) {
  try {
    const completion = await groq.chat.completions.create({
      model: ANALYSIS_MODEL,
      messages: [{ role: "user", content: PROMPTS.clipAnalysisChunk({ chunk }) }],
      temperature: 0.4,
      max_tokens: 1200,
    });

    const raw = completion?.choices?.[0]?.message?.content || "";
    const cleaned = raw.replace(/```json|```/g, "").trim();

    const parsed = JSON.parse(cleaned);
    if (!parsed.clips || !Array.isArray(parsed.clips)) return [];

    return parsed.clips;
  } catch (err) {
    console.error(`⚠️ Chunk ${chunkIndex} analysis failed (skipping):`, err.message);
    return [];
  }
}

/**
 * Runs chunked analysis across the whole transcript, normalizes field names
 * to the app-wide clip shape (now including attention_type — FIX), drops
 * heavily-overlapping lower-score clips, and returns the top 10 ranked by
 * score.
 */
async function analyzeTranscript(segments) {
  const chunks = chunkTranscript(segments);
  if (!chunks.length) return [];

  const results = [];
  for (let i = 0; i < chunks.length; i++) {
    const clips = await analyzeTranscriptChunk(chunks[i], i);
    results.push(...clips);
  }

  const normalized = results
    .map((c) => ({
      title: c.title || "Untitled clip",
      start: c.start_time || c.start,
      end: c.end_time || c.end,
      hook: c.hook || "",
      reason: c.reason || "",
      attention_type: VALID_ATTENTION_TYPES.has(c.attention_type) ? c.attention_type : null,
      attention_score: Number(c.viral_score ?? c.attention_score ?? 0),
      confidence: Number(c.confidence ?? 0),
    }))
    .filter((c) => c.start && c.end)
    .sort((a, b) => b.attention_score - a.attention_score);

  const accepted = [];
  for (const clip of normalized) {
    const start = timestampToSeconds(clip.start);
    const end = timestampToSeconds(clip.end);
    if (start === null || end === null || end <= start) continue;

    const overlapsExisting = accepted.some((existing) => {
      const exStart = timestampToSeconds(existing.start);
      const exEnd = timestampToSeconds(existing.end);
      const overlap = Math.max(0, Math.min(end, exEnd) - Math.max(start, exStart));
      const shorterLen = Math.min(end - start, exEnd - exStart);
      return shorterLen > 0 && overlap / shorterLen > 0.5;
    });

    if (!overlapsExisting) accepted.push(clip);
    if (accepted.length >= 10) break;
  }

  return accepted.map((c, i) => ({ rank: i + 1, ...c }));
}

/**
 * Runs the full pipeline in the background; updates the job record as it
 * goes. `source` is either { kind: 'upload', videoPath } for a local file
 * already on disk (from multer), or { kind: 'youtube', url } — in which
 * case this function downloads the video itself as the first stage. Both
 * paths converge on the same extract → transcribe → analyze pipeline so
 * there's a single analysis implementation to maintain.
 */
/**
 * Turns yt-dlp's raw stderr into an honest, specific message. YouTube's
 * bot-check ("Sign in to confirm you're not a bot") is the single most
 * common failure on cloud-hosted IPs (Render, AWS, GCP, etc.) — it isn't a
 * bug in this app, it's YouTube rate-limiting the server's IP range, so the
 * user needs to know that rather than see a raw exit code.
 */
function friendlyYoutubeError(rawMessage, action) {
  const msg = String(rawMessage || "");
  if (/sign in to confirm|not a bot|confirm you.?re not a bot/i.test(msg)) {
    return `YouTube is blocking this server's connection with a bot-check (this happens on cloud-hosted servers, not because of anything wrong with the link). Please upload the video file directly instead — that path doesn't go through YouTube at all.`;
  }
  if (/private video|video unavailable/i.test(msg)) {
    return `That video is private or unavailable. Double-check the link, or upload the file directly.`;
  }
  if (/age[- ]restrict/i.test(msg)) {
    return `That video is age-restricted, which YouTube blocks server-side downloads for. Upload the file directly instead.`;
  }
  if (/copyright|removed/i.test(msg)) {
    return `That video appears to have been removed or is unavailable. Double-check the link.`;
  }
  return `Could not ${action} (${msg.slice(0, 220)}). Double-check the link is public and correct, or upload the file directly.`;
}

async function runClipPipeline(job, source) {
  let videoPath = source.kind === "upload" ? source.videoPath : null;
  let audioPath = null;

  try {
    if (source.kind === "youtube") {
      job.stage = "retrieving_video";
      console.log(`[job ${job.id}] retrieving video from YouTube: ${source.url}`);

      const meta = await fetchYoutubeMetadata(source.url).catch((err) => {
        throw new Error(friendlyYoutubeError(err.message, "read that video's info"));
      });

      if (meta?.duration && meta.duration > MAX_YOUTUBE_DURATION_SECONDS) {
        throw new Error(`That video is ${Math.round(meta.duration / 60)} minutes long — Phase 1 supports up to ${MAX_YOUTUBE_DURATION_SECONDS / 60} minutes. Try a shorter video or upload a trimmed file instead.`);
      }

      videoPath = path.join(UPLOAD_DIR, `${job.id}.mp4`);
      await downloadYoutubeVideo(source.url, videoPath).catch((err) => {
        throw new Error(friendlyYoutubeError(err.message, "download that video"));
      });

      if (!fs.existsSync(videoPath) || fs.statSync(videoPath).size === 0) {
        throw new Error("Download completed but produced an empty file — the video may be age-restricted, private, or region-locked.");
      }
    }

    audioPath = videoPath + ".mp3";

    job.stage = "extracting_audio";
    console.log(`[job ${job.id}] extracting audio...`);
    await extractAudio(videoPath, audioPath);

    job.stage = "transcribing";
    console.log(`[job ${job.id}] transcribing...`);
    const { formatted, segments } = await transcribeAudio(audioPath);

    if (!formatted.trim() || !segments.length) {
      throw new Error("Transcription returned no speech content.");
    }

    job.stage = "analyzing";
    console.log(`[job ${job.id}] analyzing ${segments.length} segments...`);
    const clips = await analyzeTranscript(segments);

    job.clips = clips;
    job.status = "done";
    job.stage = "done";
    console.log(`[job ${job.id}] done — ${clips.length} clips found.`);
  } catch (err) {
    console.error(`❌ [job ${job.id}] failed at stage "${job.stage}":`, err.message);
    job.status = "error";
    job.error = err.message;
  } finally {
    if (videoPath) fs.unlink(videoPath, () => {});
    if (audioPath) fs.unlink(audioPath, () => {});
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
  res.json({ status: "online", message: "🚀 Xarvis AI v4.6" });
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
// ROUTES — VIRAL CLIP FINDER (Groq-only, Phase 1)
// ─────────────────────────────────────────────

app.post("/api/clips/upload", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: "No video file received (field name must be 'video')." });
    }

    const job = createJob();
    res.json({ success: true, jobId: job.id });

    runClipPipeline(job, { kind: "upload", videoPath: req.file.path });

  } catch (err) {
    console.error("❌ /api/clips/upload error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post("/api/clips/from-url", async (req, res) => {
  try {
    const { url } = req.body;
    if (!url || typeof url !== "string" || !YOUTUBE_URL_PATTERN.test(url.trim())) {
      return res.status(400).json({ success: false, error: "That doesn't look like a valid YouTube URL." });
    }

    const job = createJob();
    res.json({ success: true, jobId: job.id });

    runClipPipeline(job, { kind: "youtube", url: url.trim() });

  } catch (err) {
    console.error("❌ /api/clips/from-url error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.get("/api/clips/status/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ success: false, error: "Job not found — it may have expired. Please upload again." });
  }
  res.json({
    success: true,
    status: job.status,
    stage: job.stage,
    error: job.error,
  });
});

app.get("/api/clips/result/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ success: false, error: "Job not found — it may have expired. Please upload again." });
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

app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    const msg = err.code === "LIMIT_FILE_SIZE"
      ? "That file is over the 200MB limit — try a shorter clip or compress it first."
      : `Upload error: ${err.message}`;
    return res.status(400).json({ success: false, error: msg });
  }
  console.error("❌ Unhandled error:", err.message);
  res.status(500).json({ success: false, error: err.message });
});

// ─────────────────────────────────────────────
// START
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Xarvis AI v4.6 running on port ${PORT}`);
  console.log(`   Health: http://localhost:${PORT}/api/health`);
});
