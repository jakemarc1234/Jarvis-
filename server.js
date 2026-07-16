/**
 * XARVIS AI — SERVER v5.1
 *
 * Changes from v4.8:
 * - ROOT-CAUSE FIX: removed the "yt-dlp-exec" npm dependency entirely.
 *   That package relies on its own postinstall script to download the
 *   actual yt-dlp binary from GitHub during `npm install` — that step
 *   wasn't completing reliably on Render's build (invisible failure,
 *   zero build-log warning), producing an ENOENT for
 *   node_modules/yt-dlp-exec/bin/yt-dlp at request time instead.
 * - NEW: ensureYtDlpBinary() downloads the yt-dlp_linux standalone binary
 *   (PyInstaller build, no python3 dependency) ourselves at server
 *   startup, to a path we control, verifies the file size looks real
 *   (catches "downloaded an HTML error page" silently truncating to a
 *   tiny file), and logs success/failure loudly in the boot logs — not
 *   discovered later on a user's first request.
 * - NEW: runYtDlpBinary()/ytdlpOptsToArgs() spawn that binary directly via
 *   child_process.execFile with our own CLI-flag translation, replacing
 *   every yt-dlp-exec call site. Same external behavior (same flags, same
 *   player-client fallback order, same cookie/proxy support from v4.8),
 *   just no longer dependent on a third-party package's binary resolution.
 * - If the binary fails to download at boot (e.g. GitHub unreachable from
 *   Render's network), YouTube ingestion cleanly reports that specific
 *   failure per-request; file upload is entirely unaffected either way.
 * - Still Groq-only. yt-dlp is a download tool, not an AI provider.
 */

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import Groq from "groq-sdk";
import multer from "multer";
import ffmpeg from "fluent-ffmpeg";
import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
import axios from "axios";
import { execFile } from "child_process";
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
console.log("🚀 Starting Xarvis AI Server v5.1...");
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

  // ── generateHooks — final pass, only over the clips that survived
  // ranking/dedup. Given a real (verbatim, non-hallucinated) transcript
  // excerpt for each clip plus its title/reason, writes one scroll-stopping
  // hook per clip. Kept as a single batched call rather than per-clip so
  // it's one extra API round-trip regardless of how many clips survive.
  generateHooks({ clips }) {
    const list = clips
      .map((c, i) => `${i}. Title: "${c.title}"\nReason: ${c.reason}\nTranscript: "${c.transcript_preview}"`)
      .join("\n\n");
    return `You write scroll-stopping hooks for short-form video clips (YouTube Shorts, TikTok, Reels).
For each numbered clip below, write ONE hook — the line that would appear as on-screen text or be spoken in the first 2 seconds to stop someone from scrolling past.
Base the hook on the actual transcript excerpt provided — it should feel like a natural extract or tease of what's said, not a generic marketing line.
Return ONLY JSON, no markdown fences, no preamble, in this exact shape:
{ "hooks": ["hook for clip 0", "hook for clip 1", ...] }
The array must have exactly ${clips.length} entries, in the same order as the clips below.

${list}`;
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

const SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: MAX_UPLOAD_BYTES },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname || "").toLowerCase();
    if (!SUPPORTED_VIDEO_EXTENSIONS.includes(ext)) {
      // Passing an Error here (rather than throwing) is how multer reports
      // a clean rejection instead of a generic stream failure — it surfaces
      // through the same `err` path as MulterError in the error middleware.
      cb(new Error(`Unsupported file type "${ext || "unknown"}". Supported formats: ${SUPPORTED_VIDEO_EXTENSIONS.join(", ")}.`));
      return;
    }
    cb(null, true);
  },
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
    stage: "queued",      // queued | retrieving_video | validating | extracting_audio | transcribing | analyzing | generating_hooks | done
    error: null,
    clips: null,
    createdAt: Date.now(),
  };
  jobs.set(id, job);
  return job;
}

/**
 * ─────────────────────────────────────────────
 * YT-DLP BINARY — self-managed, not delegated to a third-party postinstall
 * ─────────────────────────────────────────────
 * yt-dlp-exec (and similar wrapper packages) rely on their own postinstall
 * script to fetch the yt-dlp binary from GitHub during `npm install`. That
 * step running (or not) is invisible to us and evidently isn't completing
 * reliably on Render's build — hence the ENOENT at request time with zero
 * warning at build time. Instead of trying to make someone else's install
 * hook behave, we fetch and verify the binary ourselves, at server startup,
 * to a path we control, with errors that show up immediately and loudly in
 * Render's boot logs rather than silently surfacing later on some user's
 * first request.
 *
 * We use `yt-dlp_linux` specifically — the PyInstaller-built standalone
 * binary — rather than the generic `yt-dlp` release, because the generic
 * one is a Python zipapp that requires `python3` on PATH, which a Node
 * buildpack image has no guaranteed to have. `yt-dlp_linux` has no such
 * dependency; it's a single self-contained executable.
 */
const YTDLP_BIN_DIR = path.join(os.tmpdir(), "xarvis-bin");
const YTDLP_BIN_PATH = path.join(YTDLP_BIN_DIR, "yt-dlp");
const YTDLP_DOWNLOAD_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux";

let ytdlpBinaryReady = false;

async function ensureYtDlpBinary() {
  try {
    if (fs.existsSync(YTDLP_BIN_PATH) && fs.statSync(YTDLP_BIN_PATH).size > 0) {
      ytdlpBinaryReady = true;
      console.log("✅ yt-dlp binary already present at", YTDLP_BIN_PATH);
      return;
    }

    console.log(`⬇️  yt-dlp binary not found — downloading from ${YTDLP_DOWNLOAD_URL} ...`);
    fs.mkdirSync(YTDLP_BIN_DIR, { recursive: true });

    const response = await axios.get(YTDLP_DOWNLOAD_URL, {
      responseType: "stream",
      maxRedirects: 5,
      timeout: 60_000,
    });

    await new Promise((resolve, reject) => {
      const writer = fs.createWriteStream(YTDLP_BIN_PATH);
      response.data.pipe(writer);
      writer.on("finish", resolve);
      writer.on("error", reject);
      response.data.on("error", reject);
    });

    fs.chmodSync(YTDLP_BIN_PATH, 0o755);

    const size = fs.statSync(YTDLP_BIN_PATH).size;
    if (size < 1_000_000) { // the real binary is tens of MB — a tiny file means we downloaded an error page, not the binary
      throw new Error(`Downloaded file is only ${size} bytes — expected a multi-megabyte binary. The download likely failed or was redirected to an error page.`);
    }

    ytdlpBinaryReady = true;
    console.log(`✅ yt-dlp binary downloaded and made executable (${(size / 1024 / 1024).toFixed(1)}MB) at ${YTDLP_BIN_PATH}`);
  } catch (err) {
    ytdlpBinaryReady = false;
    console.error("❌ FATAL for YouTube ingestion: could not obtain the yt-dlp binary:", err.message);
    console.error("   File uploads will still work — only YouTube link ingestion is affected.");
  }
}

/**
 * Runs the yt-dlp binary directly via child_process, translating our option
 * object into CLI flags ourselves. This is the only place that needs to
 * know the CLI flag names — everything upstream just passes a plain object.
 */
function ytdlpOptsToArgs(url, opts) {
  const args = [];
  if (opts.dumpSingleJson) args.push("--dump-single-json");
  if (opts.noWarnings) args.push("--no-warnings");
  if (opts.noCheckCertificates) args.push("--no-check-certificates");
  if (opts.preferFreeFormats) args.push("--prefer-free-formats");
  if (opts.skipDownload) args.push("--skip-download");
  if (opts.noPlaylist) args.push("--no-playlist");
  if (opts.extractorArgs) args.push("--extractor-args", opts.extractorArgs);
  if (opts.cookies) args.push("--cookies", opts.cookies);
  if (opts.proxy) args.push("--proxy", opts.proxy);
  if (opts.format) args.push("--format", opts.format);
  if (opts.output) args.push("--output", opts.output);
  if (opts.maxFilesize) args.push("--max-filesize", String(opts.maxFilesize));
  args.push(url);
  return args;
}

function runYtDlpBinary(url, opts) {
  console.log("YT-DLP DIAGNOSTIC — runYtDlpBinary() called for URL:", url);
  return new Promise((resolve, reject) => {
    if (!ytdlpBinaryReady) {
      console.log("YT-DLP DIAGNOSTIC — ytdlpBinaryReady is false, rejecting before execFile runs at all.");
      reject(new Error("The yt-dlp binary isn't available on this server (it failed to download at startup — check the Render logs for the exact download error). YouTube link ingestion can't run until that's resolved; file upload is unaffected."));
      return;
    }
    const args = ytdlpOptsToArgs(url, opts);
    console.log("YT-DLP COMMAND:", YTDLP_BIN_PATH, args.join(" "));
    execFile(YTDLP_BIN_PATH, args, { maxBuffer: 1024 * 1024 * 50, timeout: 5 * 60 * 1000 }, (error, stdout, stderr) => {
      if (error) {
        console.log("YT-DLP DIAGNOSTIC — command failed");
        console.log("YT-DLP DIAGNOSTIC — full command:", YTDLP_BIN_PATH, args.join(" "));
        console.log("YT-DLP DIAGNOSTIC — error.message:", error.message);
        console.log("YT-DLP DIAGNOSTIC — error.code (exit code):", error.code);
        console.log("YT-DLP DIAGNOSTIC — error.signal:", error.signal);
        console.log("YT-DLP DIAGNOSTIC — full stderr:\n" + stderr);
        console.log("YT-DLP DIAGNOSTIC — full stdout:\n" + stdout);
        error.stderr = stderr;
        reject(error);
        return;
      }
      if (opts.dumpSingleJson) {
        try {
          resolve(JSON.parse(stdout));
        } catch (parseErr) {
          console.log("YT-DLP DIAGNOSTIC — JSON parse failed. Raw stdout was:\n" + stdout);
          reject(new Error(`yt-dlp returned output that wasn't valid JSON: ${parseErr.message}`));
        }
      } else {
        resolve(stdout);
      }
    });
  });
}

/**
 * yt-dlp's stderr is where the actual reason for a failure lives (YouTube's
 * response, a bad URL, etc.) — the raw Node error.message alone is just
 * "Command failed with exit code N".
 */
function extractYtDlpError(err) {
  const stderr = (err && err.stderr) || "";
  const cleaned = String(stderr)
    .split("\n")
    .filter((line) => line.trim() && !line.trim().startsWith("[debug]"))
    .join(" ")
    .trim();
  return cleaned || err?.message || String(err);
}

function isBotCheckError(err) {
  return /sign in to confirm|not a bot|confirm you.?re not a bot/i.test(extractYtDlpError(err));
}

/**
 * Real fix for YouTube's bot-check: authenticate as a logged-in browser
 * session via cookies, same as every production tool does. Set the
 * YTDLP_COOKIES env var on Render to the full contents of a cookies.txt
 * file exported from a logged-in YouTube session in your own browser (the
 * "Get cookies.txt LOCALLY" extension does this in one click — export
 * while on youtube.com). Without this set, yt-dlp falls back to
 * unauthenticated requests, which is what cloud IPs get bot-checked on.
 */
const YTDLP_COOKIES_PATH = path.join(os.tmpdir(), "xarvis-yt-cookies.txt");
let ytdlpCookiesReady = false;
if (process.env.YTDLP_COOKIES && process.env.YTDLP_COOKIES.trim()) {
  try {
    fs.writeFileSync(YTDLP_COOKIES_PATH, process.env.YTDLP_COOKIES.trim() + "\n");
    ytdlpCookiesReady = true;
    console.log("✅ YouTube cookies file configured — authenticated downloads enabled.");
  } catch (err) {
    console.error("⚠️ Could not write YTDLP_COOKIES to disk:", err.message);
  }
} else {
  console.log("ℹ️  YTDLP_COOKIES not set — YouTube downloads will rely on player-client spoofing only, which YouTube's bot-check can still block. See server.js comments for how to set it.");
}
if (process.env.YTDLP_PROXY && process.env.YTDLP_PROXY.trim()) {
  console.log("✅ YTDLP_PROXY configured — YouTube downloads will route through it.");
}

// Player clients to try in order — success is inconsistent and shifts as
// YouTube patches things, so we no longer bet on a single one.
const YTDLP_CLIENT_FALLBACK_ORDER = ["android", "ios", "web_embedded", "tv_embedded"];

function ytdlpBaseOpts(playerClient) {
  const opts = {
    noWarnings: true,
    noCheckCertificates: true,
    extractorArgs: `youtube:player_client=${playerClient}`,
  };
  if (ytdlpCookiesReady) opts.cookies = YTDLP_COOKIES_PATH;
  // Option B from the audit: a residential/mobile proxy from a provider
  // (Bright Data, Oxylabs, Smartproxy, etc.) sidesteps the datacenter-IP
  // bot-check entirely, independent of cookies. Set YTDLP_PROXY to the
  // full proxy URL, e.g. http://user:pass@host:port — no other code
  // changes needed if this is the direction you choose.
  if (process.env.YTDLP_PROXY && process.env.YTDLP_PROXY.trim()) {
    opts.proxy = process.env.YTDLP_PROXY.trim();
  }
  return opts;
}

/**
 * Runs a yt-dlp call, retrying with the next player client only when the
 * failure looks like YouTube's bot-check (retrying on every kind of error —
 * a genuinely private video, say — would just waste time failing the same
 * way four times over).
 */
async function runYtDlpWithFallback(url, extraOpts) {
  let lastErr;
  for (const client of YTDLP_CLIENT_FALLBACK_ORDER) {
    try {
      return await runYtDlpBinary(url, { ...ytdlpBaseOpts(client), ...extraOpts });
    } catch (err) {
      lastErr = err;
      if (!isBotCheckError(err)) throw err; // a different kind of failure — no point trying other clients
      console.warn(`[yt-dlp] client "${client}" hit bot-check, trying next client...`);
    }
  }
  throw lastErr;
}

/**
 * Fetch metadata only (no download) so we can reject unreasonably long
 * videos before spending time/bandwidth on them.
 */
async function fetchYoutubeMetadata(url) {
  try {
    return await runYtDlpWithFallback(url, {
      dumpSingleJson: true,
      preferFreeFormats: true,
      skipDownload: true,
    });
  } catch (err) {
    throw new Error(extractYtDlpError(err));
  }
}

/**
 * Downloads a YouTube video's audio via yt-dlp. FIX: previously requested
 * "worst[ext=mp4]/worst" — a muxed video+audio format in an mp4 container.
 * Many videos no longer serve progressive/muxed formats at all (YouTube
 * increasingly only exposes separate video-only and audio-only adaptive
 * streams), so that selector had nothing to match and yt-dlp failed with
 * "Requested format is not available." We never needed video in the first
 * place — extractAudio() immediately discards it — so requesting
 * "bestaudio/best" is both the fix and strictly less work: audio-only
 * streams are essentially always available, and downloads are smaller.
 *
 * outputPath should be a template ending in `.%(ext)s` rather than a fixed
 * extension, since the actual container yt-dlp picks (m4a/webm/opus) isn't
 * known in advance. ffmpeg auto-detects the real container from file
 * content in extractAudio(), so a "wrong" extension on disk doesn't matter.
 */
async function downloadYoutubeVideo(url, outputPath) {
  try {
    await runYtDlpWithFallback(url, {
      output: outputPath,
      format: "bestaudio/best",
      noPlaylist: true,
      maxFilesize: `${MAX_UPLOAD_BYTES}`,
    });
  } catch (err) {
    throw new Error(extractYtDlpError(err));
  }
}

/**
 * Real "validating file" stage: probes the file with ffprobe to confirm it
 * has at least one usable stream before committing to extraction/
 * transcription/analysis. Catches corrupted files, mislabeled non-video
 * files, and empty uploads with one clean error here instead of a raw
 * ffmpeg stack trace surfacing three stages later.
 */
function validateVideoFile(videoPath) {
  return new Promise((resolve, reject) => {
    const stat = fs.existsSync(videoPath) ? fs.statSync(videoPath) : null;
    if (!stat || stat.size === 0) {
      reject(new Error("The uploaded file is empty."));
      return;
    }
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        reject(new Error("That file doesn't look like a playable video — it may be corrupted, encrypted, or not actually a video file. Try a different file."));
        return;
      }
      const hasAudioOrVideoStream = (metadata?.streams || []).some(
        (s) => s.codec_type === "audio" || s.codec_type === "video"
      );
      if (!hasAudioOrVideoStream) {
        reject(new Error("That file doesn't contain any audio or video stream Xarvis can read."));
        return;
      }
      const hasAudio = (metadata?.streams || []).some((s) => s.codec_type === "audio");
      if (!hasAudio) {
        reject(new Error("That video doesn't have an audio track — Xarvis needs spoken audio to find viral moments."));
        return;
      }
      resolve(metadata);
    });
  });
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
// How many chunk-analysis calls run at once. Capped rather than unlimited
// because Groq enforces per-account rate limits — firing all chunks of a
// long video simultaneously would trade one bottleneck for 429s. 4 is a
// conservative default that still cuts a 6-chunk (30min) video's analysis
// time roughly in half to a third versus fully sequential.
const CHUNK_ANALYSIS_CONCURRENCY = Number(process.env.CHUNK_ANALYSIS_CONCURRENCY) || 4;

async function mapWithConcurrency(items, limit, fn) {
  const results = new Array(items.length);
  let nextIndex = 0;
  async function worker() {
    while (nextIndex < items.length) {
      const i = nextIndex++;
      results[i] = await fn(items[i], i);
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, worker);
  await Promise.all(workers);
  return results;
}

/**
 * Pulls the actual verbatim transcript text for a clip's time range directly
 * from Whisper's segments — not asked of the LLM, so it's guaranteed
 * accurate rather than a paraphrase or hallucination. This is what powers
 * the "Transcript Preview" on each result card.
 */
function extractTranscriptExcerpt(segments, startSec, endSec) {
  const text = segments
    .filter((s) => s.end >= startSec && s.start <= endSec)
    .map((s) => s.text.trim())
    .join(" ")
    .trim();
  if (text.length <= 240) return text;
  return text.slice(0, 240).trim() + "…";
}

/**
 * Final pass, run once over only the clips that survived ranking/dedup
 * (not per-chunk-candidate, which would waste tokens on discarded clips).
 * Falls back to using the transcript excerpt itself as the "hook" if this
 * call fails — never blocks the whole job on one extra API call failing.
 */
async function generateHooksForClips(clips) {
  if (!clips.length) return clips;
  try {
    const completion = await groq.chat.completions.create({
      model: ANALYSIS_MODEL,
      messages: [{ role: "user", content: PROMPTS.generateHooks({ clips }) }],
      temperature: 0.6,
      max_tokens: 800,
    });
    const raw = completion?.choices?.[0]?.message?.content || "";
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(cleaned);
    const hooks = Array.isArray(parsed.hooks) ? parsed.hooks : [];
    return clips.map((c, i) => ({ ...c, hook: hooks[i] || c.transcript_preview || c.title }));
  } catch (err) {
    console.error("⚠️ Hook generation failed, falling back to transcript excerpt as hook:", err.message);
    return clips.map((c) => ({ ...c, hook: c.transcript_preview || c.title }));
  }
}

async function analyzeTranscript(segments, job) {
  const chunks = chunkTranscript(segments);
  if (!chunks.length) return [];

  const perChunkResults = await mapWithConcurrency(
    chunks,
    CHUNK_ANALYSIS_CONCURRENCY,
    (chunk, i) => analyzeTranscriptChunk(chunk, i)
  );
  const results = perChunkResults.flat();

  const normalized = results
    .map((c) => ({
      title: c.title || "Untitled clip",
      start: c.start_time || c.start,
      end: c.end_time || c.end,
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

    if (!overlapsExisting) accepted.push({ ...clip, transcript_preview: extractTranscriptExcerpt(segments, start, end) });
    if (accepted.length >= 10) break;
  }

  const ranked = accepted.map((c, i) => ({ rank: i + 1, ...c }));
  if (job) job.stage = "generating_hooks";
  return generateHooksForClips(ranked);
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
    return ytdlpCookiesReady
      ? `YouTube is still blocking this server's connection even with cookies configured — the cookies may have expired (YouTube session cookies typically last a few weeks). Export a fresh cookies.txt and update YTDLP_COOKIES on Render, or upload the video file directly for now.`
      : `YouTube is blocking this server's connection with a bot-check — this happens on cloud-hosted servers and needs authenticated cookies to fix reliably (ask about setting up YTDLP_COOKIES). For now, please upload the video file directly instead — that path doesn't go through YouTube at all.`;
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

      const outputTemplate = path.join(UPLOAD_DIR, `${job.id}.%(ext)s`);
      await downloadYoutubeVideo(source.url, outputTemplate).catch((err) => {
        throw new Error(friendlyYoutubeError(err.message, "download that video"));
      });

      // yt-dlp resolves %(ext)s to whatever container the chosen audio
      // stream actually came in (m4a/webm/opus/etc.) — find that real file
      // rather than assuming a fixed extension.
      const downloadedFile = fs.readdirSync(UPLOAD_DIR).find((f) => f.startsWith(`${job.id}.`));
      if (!downloadedFile) {
        throw new Error("Download completed but the resulting file couldn't be found — the video may be age-restricted, private, or region-locked.");
      }
      videoPath = path.join(UPLOAD_DIR, downloadedFile);

      if (fs.statSync(videoPath).size === 0) {
        throw new Error("Download completed but produced an empty file — the video may be age-restricted, private, or region-locked.");
      }
    }

    audioPath = videoPath + ".mp3";

    job.stage = "validating";
    console.log(`[job ${job.id}] validating file...`);
    await validateVideoFile(videoPath);

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
    const clips = await analyzeTranscript(segments, job);

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
await ensureYtDlpBinary();

// ─────────────────────────────────────────────
// ROUTES — HEALTH
// ─────────────────────────────────────────────
app.get("/", (req, res) => {
  res.json({ status: "online", message: "🚀 Xarvis AI v5.1" });
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
  if (err.message?.startsWith("Unsupported file type")) {
    return res.status(400).json({ success: false, error: err.message });
  }
  console.error("❌ Unhandled error:", err.message);
  res.status(500).json({ success: false, error: err.message });
});

// ─────────────────────────────────────────────
// START
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Xarvis AI v5.1 running on port ${PORT}`);
  console.log(`   Health: http://localhost:${PORT}/api/health`);
});
