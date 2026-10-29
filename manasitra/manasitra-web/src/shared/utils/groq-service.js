import { CapacitorHttp } from '@capacitor/core'
import { FALLBACK_RESPONSES } from '@data/prompts/system-prompts'

const getApiUrl = () => {
  return import.meta.env.VITE_API_URL || 'http://43.205.235.182:8000'
}

export const sendChatMessage = async ({ messages, language, responseMode, sessionContext = [], userName }) => {
  const lastUserMessage = messages.filter(m => m.role === 'user').pop()
  const userText = lastUserMessage ? lastUserMessage.content : ''

  try {
    const response = await CapacitorHttp.post({
      url: `${getApiUrl()}/chat`,
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        user_name: userName || 'Friend',
        message: userText
      }
    })

    if (response.status !== 200) {
      throw new Error(`HTTP ${response.status}: ${JSON.stringify(response.data)}`)
    }

    const resData = typeof response.data === 'string' ? JSON.parse(response.data) : response.data
    return {
      response: resData.reply || "No response",
      detectedMood: resData.emotion || 'neutral',
      riskLevel: 'none',
      suggestedTool: null,
      escalationRequired: false,
    }
  } catch (err) {
    console.error('[Backend API Error]', err.message)
    const category = responseMode || 'general'
    const lang = language || 'en'
    const fallbacks = FALLBACK_RESPONSES[category]?.[lang] || FALLBACK_RESPONSES[category]?.en || FALLBACK_RESPONSES.general.en
    return {
      response: fallbacks[Math.floor(Math.random() * fallbacks.length)],
      detectedMood: 'neutral',
      riskLevel: 'none',
      suggestedTool: null,
      escalationRequired: false,
      isFallback: true,
    }
  }
}

export const getFallbackResponse = (detectedMood, language) => {
  const key = detectedMood || 'anxious'
  const fallbacks = FALLBACK_RESPONSES[key] || FALLBACK_RESPONSES.anxious
  return {
    response: fallbacks[language] || fallbacks.en,
    detectedMood: key,
    riskLevel: 'none',
    suggestedTool: null,
    isFallback: true,
  }
}
