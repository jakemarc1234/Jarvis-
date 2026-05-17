# Xarvis AI v2.0 — Production-Grade Creator Intelligence Platform

> Multi-agent AI system. ChatGPT × Notion AI × Stripe-level architecture.

---

## 🏗 Architecture Overview

```
xarvis-ai/
├── backend/
│   ├── config/          ← Central config. ALL env vars consumed here. Never import process.env elsewhere.
│   ├── controllers/     ← HTTP layer. Parse request → call service → format response. No business logic.
│   │   ├── chat.controller.js
│   │   ├── tools.controller.js
│   │   └── memory.controller.js
│   ├── middleware/      ← Cross-cutting concerns: auth, session, validation, rate limiting.
│   ├── routes/          ← Route definitions. Maps URL paths to controllers.
│   ├── services/
│   │   ├── ai-router.js         ← Smart AI routing with circuit breakers
│   │   ├── memory.js            ← Memory abstraction layer
│   │   ├── agents/index.js      ← All 4 agents (BaseAgent + specializations)
│   │   ├── prompts/index.js     ← ALL prompts externalized here
│   │   ├── providers/
│   │   │   ├── groq.js          ← Primary provider (fast)
│   │   │   ├── anthropic.js     ← Fallback provider (deep reasoning)
│   │   │   └── static-fallback.js ← Last resort (never crashes)
│   │   └── memory/
│   │       └── local-adapter.js ← In-memory (swap for Redis/Pinecone adapter)
│   ├── utils/
│   │   ├── logger.js    ← Structured logger (replace with Winston/Pino in prod)
│   │   └── errors.js    ← AppError class + global error handler
│   ├── app.js           ← Express setup, middleware chain, route mounting
│   └── server.js        ← Entry point. Starts server + handles graceful shutdown.
├── frontend/
│   ├── index.html       ← Full production SPA (vanilla JS, zero dependencies)
│   └── utils/api.js     ← All API calls abstracted here. No raw fetch() in UI components.
├── .env.example
└── package.json
```

---

## 🧠 Data Flow

```
User Request
    │
    ▼
Express Router (routes/)
    │
    ▼
Middleware Chain
  ├── helmet (security headers)
  ├── cors (origin whitelist)
  ├── sessionMiddleware (attach sessionId)
  ├── rateLimiter (per-IP + per-session)
  └── validateBody (required field check)
    │
    ▼
Controller (controllers/)
  ├── Parse & validate request body
  ├── Build context from memory
  └── Call agent.run() or agent.runStream()
    │
    ▼
Agent (services/agents/)
  ├── Fetch creator context from MemoryService
  ├── Build system prompt with context injection
  └── Call AIRouter.complete() or AIRouter.stream()
    │
    ▼
AI Router (services/ai-router.js)
  ├── Check circuit breaker state per provider
  ├── Try Groq (primary — fast, cheap)
  ├── On failure → Try Anthropic (fallback — deep reasoning)
  └── On all failures → Static fallback (never 500s)
    │
    ▼
Provider (services/providers/)
  ├── Format request for provider API
  ├── Handle streaming/non-streaming
  └── Return unified response shape
    │
    ▼
Controller
  ├── Persist exchange to MemoryService
  └── Return JSON response
```

---

## 🤖 Agent System

| Agent | ID | Prompt Focus | Use Case |
|---|---|---|---|
| Xarvis Core | `core` | All-around strategist | Chat, general advice |
| Viral Content | `viral` | Hook psychology, formats | Viral Studio tool |
| Strategy | `strategy` | Roadmaps, phases, planning | Calendar, Execution Plan |
| Creator Growth | `growth` | Monetization, revenue | Growth-specific queries |

**Adding a new agent**: Create a `new BaseAgent('Name', SYSTEM_PROMPTS.yourNewPrompt)` in `services/agents/index.js`. Register in AGENTS map. No other files need changing.

---

## 🔀 AI Router — Circuit Breaker Logic

```
Provider State Machine:
  CLOSED (normal) → failure → OPEN (skip provider)
                                    ↓
                            60s elapsed → HALF-OPEN → success → CLOSED
                                                     → failure → OPEN

Circuit opens after 3 consecutive failures.
Auto-resets after 60 seconds.
```

---

## 💾 Memory System — Adapter Pattern

```javascript
// Swap drivers via config — zero code changes elsewhere
MEMORY_DRIVER=local    // Map-based (current)
MEMORY_DRIVER=redis    // Redis (horizontal scale)
MEMORY_DRIVER=vector   // Pinecone/Weaviate (semantic memory)
```

All adapters implement the same interface:
- `getHistory(sessionId)`
- `appendMessage(sessionId, role, content)`
- `getProfile(sessionId)` / `setProfile(sessionId, profile)`
- `clearSession(sessionId)`

---

## 🚀 Setup

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start development server
npm run dev

# 4. Open frontend
open frontend/index.html
# Or serve with: npx serve frontend -p 3000
```

**Required API Keys:**
- [Groq](https://console.groq.com) — Primary provider (free tier available)
- [Anthropic](https://console.anthropic.com) — Fallback provider

---

## 📡 API Reference

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/api/chat` | `{ message }` | Standard chat |
| POST | `/api/chat/stream` | `{ message }` | SSE streaming chat |
| POST | `/api/tools/viral-content` | `{ idea, format }` | Viral package |
| POST | `/api/tools/post-idea` | `{ additionalContext? }` | Best post idea |
| POST | `/api/tools/calendar` | `{}` | 7-day calendar |
| POST | `/api/tools/analyze` | `{ content }` | Content analysis |
| POST | `/api/tools/execution-plan` | `{ goal }` | Roadmap |
| GET | `/api/memory` | — | Get profile |
| POST | `/api/memory` | `{ niche, platform, goal, tone }` | Save profile |
| DELETE | `/api/memory` | — | Clear session |

All requests require `X-Session-ID` header.

---

## 🔮 Scaling Roadmap

| Layer | Current | Next |
|---|---|---|
| Memory | In-memory Map | Redis (add adapter) |
| Semantic Memory | None | Pinecone (add adapter) |
| Auth | Session ID | JWT + user accounts |
| Providers | Groq + Claude | + OpenAI, Gemini |
| Jobs | Synchronous | Bull queue (async) |
| Deployment | Node.js | Docker + Railway/Fly.io |
| Monitoring | Console logs | Datadog / Sentry |

---

*Built for scale. Designed like a real AI startup.*
