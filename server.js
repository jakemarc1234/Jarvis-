const express = require("express");
const cors = require("cors");
require("dotenv").config();

const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
app.use(cors());
app.use(express.json());

// 🔑 Gemini setup
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// 🧠 VIRAL AI ENGINE (Gemini-powered)
async function getViralScore(idea) {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const prompt = `
You are a viral content expert.

Score this content idea from 0 to 100 based on VIRAL potential.

Analyze:
- Hook strength
- Emotion (anger, curiosity, motivation)
- Shareability
- Trend potential
- Retention power

Return ONLY valid JSON like this:
{
  "score": number,
  "verdict": "VIRAL | GOOD | MID | LOW",
  "reason": "short explanation"
}

Idea:
"${idea}"
`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    try {
        return JSON.parse(text);
    } catch (e) {
        return {
            score: 50,
            verdict: "MID",
            reason: "AI returned non-JSON response"
        };
    }
}

// 🚀 API ROUTE
app.post("/score", async (req, res) => {
    try {
        const { idea } = req.body;

        if (!idea) {
            return res.status(400).json({ error: "Idea is required" });
        }

        const result = await getViralScore(idea);

        res.json({
            idea,
            viralScore: result.score,
            verdict: result.verdict,
            reason: result.reason
        });

    } catch (err) {
        res.status(500).json({
            error: "Server error",
            details: err.message
        });
    }
});

// 🌐 START SERVER
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`🚀 Xarvis AI (Gemini Engine) running on port ${PORT}`);
});