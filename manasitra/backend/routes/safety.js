const express = require('express')
const router = express.Router()
const { createChatCompletion } = require('../services/groq-client')
const { sanitizeInput, validateLanguage } = require('../services/safety-service')
const { hasCrisisKeyword } = require('../utils/prompt-builder')
const { chatLimiter } = require('../middleware/rate-limiter')

router.post('/classify', chatLimiter, async (req, res) => {
  try {
    const { message, language } = req.body
    const lang = validateLanguage(language)
    const sanitized = sanitizeInput(message)

    if (hasCrisisKeyword(sanitized)) {
      return res.json({
        riskLevel: 'critical',
        flags: ['crisis_keyword_detected'],
        recommendedAction: 'crisis_screen',
      })
    }

    const completion = await createChatCompletion([
      {
        role: 'system',
        content: `You are a safety classifier for a mental health support app. Classify the risk level of the following message from an Indian student. Return ONLY valid JSON: {"riskLevel": "none|low|medium|high|critical", "flags": [], "recommendedAction": "none|surface_helpline|crisis_screen"}`,
      },
      { role: 'user', content: sanitized },
    ], { max_tokens: 100, temperature: 0 })

    const raw = completion.choices[0]?.message?.content || '{}'
    let parsed
    try { parsed = JSON.parse(raw) } catch { parsed = { riskLevel: 'none', flags: [], recommendedAction: 'none' } }

    res.json(parsed)
  } catch (err) {
    console.error('Safety classify error:', err.message)
    const fallbackSanitized = sanitizeInput(req.body?.message || '')
    if (hasCrisisKeyword(fallbackSanitized)) {
      return res.json({
        riskLevel: 'critical',
        flags: ['crisis_keyword_detected_offline'],
        recommendedAction: 'crisis_screen',
      })
    }
    res.json({ riskLevel: 'none', flags: [], recommendedAction: 'none' })
  }
})

module.exports = router
