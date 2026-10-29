const express = require('express')
const router = express.Router()
const { createChatCompletion } = require('../services/groq-client')
const { buildSystemPrompt, hasCrisisKeyword } = require('../utils/prompt-builder')
const { sanitizeInput, validateLanguage, validateResponseMode } = require('../services/safety-service')
const { chatLimiter } = require('../middleware/rate-limiter')

const extractJSON = (str) => {
  if (!str) return null
  try { return JSON.parse(str) } catch {}
  const match = str.match(/\{[\s\S]*\}/)
  if (match) { try { return JSON.parse(match[0]) } catch {} }
  return null
}

const MULTILINGUAL_FALLBACKS = {
  hi: "Main yahan aapke saath hu dost. Aap akele nahi hain, sab theek ho jayega. 💙 Kya aap thoda aur bata sakte hain ki kya chal raha hai?",
  hg: "Main yahan aapke saath hu dost. You are not alone, sab theek ho jayega. 💙 Thoda aur batao kya chal raha hai?",
  gu: "હું અહીં તમારી સાથે છું, મિત્ર. તમે એકલા નથી, બધું સારું થઈ જશે. 💙 શું તમે થોડું વધુ જણાવી શકો કે શું થઈ રહ્યું છે?",
  gj: "Hu ahi tari sathe chu dost. You are not alone, badhu saru thai jashe. 💙 Thodu vadhare batav shu thayu che?",
  mr: "मी इथे तुझ्या सोबत आहे, मित्र. तू एकटा नाहीस, सर्व काही ठीक होईल. 💙 काय घडत आहे ते मला थोडे अधिक सांगशील का?",
  en: "I'm right here with you, friend. You are not alone, and we will get through this together. 💙 Can you tell me a little more about what's on your mind?",
}

router.post('/', chatLimiter, async (req, res) => {
  try {
    const { messages, language, responseMode, sessionContext = [], voice_mode = false } = req.body

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages required' })
    }

    const lang = validateLanguage(language)
    const mode = validateResponseMode(responseMode)

    // Sanitize user messages
    const sanitizedMessages = messages.map(m => ({
      role: m.role,
      content: m.role === 'user' ? sanitizeInput(m.content) : m.content,
    }))

    // Client-side crisis pre-check
    const userText = sanitizedMessages.map(m => m.content).join(' ')
    const isCrisisInput = hasCrisisKeyword(userText)

    const systemPrompt = buildSystemPrompt(lang, isCrisisInput ? 'crisis' : mode, voice_mode)

    const contextMessages = sessionContext
      .slice(-6)
      .map(m => ({ role: m.role, content: m.role === 'user' ? sanitizeInput(m.content) : m.content }))

    const fullMessages = [
      { role: 'system', content: systemPrompt },
      ...contextMessages,
      ...sanitizedMessages,
    ]

    const completion = await createChatCompletion(fullMessages, {
      max_tokens: voice_mode ? 80 : 220,
      temperature: 0.7,
    })
    const raw = completion.choices[0]?.message?.content || ''
    const parsed = extractJSON(raw)

    // Check if AI flagged crisis
    const isCrisisResponse = raw.includes('CRISIS_DETECTED') || isCrisisInput

    if (parsed && parsed.response) {
      return res.json({
        response: parsed.response.replace('CRISIS_DETECTED', '').trim(),
        detectedMood: parsed.detectedMood || 'neutral',
        riskLevel: isCrisisResponse ? 'critical' : (parsed.riskLevel || 'none'),
        suggestedTool: parsed.suggestedTool === 'null' ? null : (parsed.suggestedTool || null),
        escalationRequired: isCrisisResponse || ['high', 'critical'].includes(parsed.riskLevel),
        languageVerified: true,
      })
    }

    if (raw && raw.length > 10) {
      return res.json({
        response: raw.replace('CRISIS_DETECTED', '').trim(),
        detectedMood: 'neutral',
        riskLevel: isCrisisResponse ? 'critical' : 'none',
        suggestedTool: null,
        escalationRequired: isCrisisResponse,
        languageVerified: true,
      })
    }

    res.status(500).json({ error: 'Empty response from AI' })

  } catch (err) {
    console.error('[Chat] Fallback triggered:', err.message)
    const userText = (req.body?.messages || []).map(m => m.content || '').join(' ')
    const isCrisis = hasCrisisKeyword(userText)

    if (isCrisis) {
      return res.json({
        response: "I'm really worried about you right now, and I care about you deeply. 💙 You are not alone. Please reach out to iCall (9152987821) or AASRA (9820466627) right now.",
        detectedMood: 'grief',
        riskLevel: 'critical',
        suggestedTool: null,
        escalationRequired: true,
        languageVerified: true,
      })
    }

    const langKey = req.body?.language || 'hi'
    const fallbackText = MULTILINGUAL_FALLBACKS[langKey] || MULTILINGUAL_FALLBACKS.en

    return res.json({
      response: fallbackText,
      detectedMood: 'neutral',
      riskLevel: 'none',
      suggestedTool: null,
      escalationRequired: false,
      languageVerified: true,
    })
  }
})

module.exports = router
