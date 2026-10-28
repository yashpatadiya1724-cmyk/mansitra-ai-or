import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'
const api = axios.create({ baseURL: API_BASE_URL })

// ── Blocking chat ─────────────────────────────────────────────────────────────
export async function sendMessage(userName, message) {
  const { data } = await api.post('/chat', { user_name: userName, message })
  return data // { reply, emotion, confidence, scores, model_used }
}

// ── Streaming chat ────────────────────────────────────────────────────────────
/**
 * Calls /chat/stream and fires callbacks as tokens arrive.
 * @param {string} userName
 * @param {string} message
 * @param {(token: string) => void} onToken   - called for each streamed token
 * @param {(meta: object) => void}  onDone    - called once with { emotion, confidence, scores }
 */
export async function sendMessageStream(userName, message, onToken, onDone) {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_name: userName, message }),
  })

  if (!response.ok) throw new Error(`HTTP ${response.status}`)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const payload = JSON.parse(line.slice(6))
        if (payload.done) {
          onDone(payload)
        } else if (payload.token) {
          onToken(payload.token)
        }
      } catch {
        // ignore malformed lines
      }
    }
  }
}

// ── Mood ──────────────────────────────────────────────────────────────────────
export async function getMoodHistory(userName, limit = 20) {
  const { data } = await api.get('/mood/history', { params: { user_name: userName, limit } })
  return data
}

export async function getMoodStats(userName, limit = 30) {
  const { data } = await api.get('/mood/stats', { params: { user_name: userName, limit } })
  return data
}

// ── Ollama status ─────────────────────────────────────────────────────────────
export async function getOllamaStatus() {
  const { data } = await api.get('/ollama/status')
  return data // { ollama_running, model, ready }
}

export async function getInnovationStatus() {
  const { data } = await api.get('/innovation/status')
  return data
}


// ── Feature 2: LMPE — Mood patterns ──────────────────────────────────────────
export async function getMoodPatterns(userName) {
  const { data } = await api.get('/mood/patterns', { params: { user_name: userName } })
  return data // { patterns, insights }
}

// ── Feature 1: EAMC — Weighted memory ────────────────────────────────────────
export async function getWeightedMemory(userName, limit = 10) {
  const { data } = await api.get('/mood/memory', { params: { user_name: userName, limit } })
  return data // { weighted_memory }
}

// ── Voice ─────────────────────────────────────────────────────────────────────
export async function transcribeAudio(audioBlob) {
  const form = new FormData()
  form.append('file', audioBlob, 'recording.wav')
  const { data } = await api.post('/voice/transcribe', form)
  return data.transcription
}

// ── Journal ───────────────────────────────────────────────────────────────────
export async function addJournalEntry(userName, entryType, content) {
  const { data } = await api.post('/journal', { user_name: userName, entry_type: entryType, content })
  return data
}

export async function getJournalEntries(userName, limit = 50) {
  const { data } = await api.get('/journal', { params: { user_name: userName, limit } })
  return data // { entries }
}
