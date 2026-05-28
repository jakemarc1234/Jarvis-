// config.js
/**
 * XARVIS AI — PRODUCTION CONFIG
 * Centralized frontend configuration
 */

const PROD_BACKEND = "https://xarvis-ai.onrender.com";
const DEV_BACKEND  = "http://localhost:3001";

const isLocalhost =
  location.hostname === "localhost" ||
  location.hostname === "127.0.0.1";

const API_BASE = isLocalhost ? DEV_BACKEND : PROD_BACKEND;

export const CONFIG = {
  API_BASE,

  ROUTES: {
    HEALTH:   "/api/health",
    CHAT:     "/api/chat",
    STREAM:   "/api/chat/stream",
    GENERATE: "/api/generate",
    AGENT:    "/api/agent/plan",
  },

  REQUEST_TIMEOUT: 35000,
  RETRY_ATTEMPTS:  2,
  RETRY_DELAY:     1500,
  MAX_HISTORY:     10,
  DEBUG:           true,
};

if (!CONFIG.API_BASE) {
  throw new Error("[Xarvis Config] Missing API_BASE");
}

console.log("[Xarvis Config Loaded]", {
  API_BASE: CONFIG.API_BASE,
  MODE: isLocalhost ? "development" : "production",
});
