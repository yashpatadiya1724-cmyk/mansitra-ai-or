import { useState, useEffect } from 'react'
import { addJournalEntry, getJournalEntries } from '../api'

export default function JournalPage() {
  const [entries, setEntries] = useState([])
  const [content, setContent] = useState('')
  const [entryType, setEntryType] = useState('achievement')
  const [userName, setUserName] = useState('Yash') // TODO: Dynamic if needed
  const [loading, setLoading] = useState(false)
  const [fetching, setFetching] = useState(true)

  const loadEntries = async () => {
    try {
      setFetching(true)
      const data = await getJournalEntries(userName)
      setEntries(data.entries || [])
    } catch (error) {
      console.error('Failed to load journal entries', error)
    } finally {
      setFetching(false)
    }
  }

  useEffect(() => {
    loadEntries()
  }, [userName])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!content.trim()) return

    try {
      setLoading(true)
      await addJournalEntry(userName, entryType, content)
      setContent('')
      loadEntries() // Reload the list
    } catch (error) {
      console.error('Failed to save journal entry', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6 h-full flex flex-col md:flex-row gap-6">
      {/* Left side: Add Entry Form */}
      <div className="flex-1 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col">
        <h2 className="text-2xl font-bold text-white mb-2">My Journal</h2>
        <p className="text-slate-400 mb-6 text-sm">
          Log your daily achievements and difficult moments. Manasitra will use this to motivate you later.
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4 flex-1">
          <div className="flex gap-4 mb-2">
            <button
              type="button"
              onClick={() => setEntryType('achievement')}
              className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                entryType === 'achievement'
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                  : 'bg-slate-800 text-slate-400 border border-transparent hover:bg-slate-700'
              }`}
            >
              🌟 Achievement
            </button>
            <button
              type="button"
              onClick={() => setEntryType('bad_time')}
              className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                entryType === 'bad_time'
                  ? 'bg-red-500/20 text-red-400 border border-red-500/50'
                  : 'bg-slate-800 text-slate-400 border border-transparent hover:bg-slate-700'
              }`}
            >
              🌧️ Bad Time
            </button>
          </div>

          <textarea
            className="flex-1 bg-slate-950 border border-slate-800 rounded-xl p-4 text-white placeholder-slate-500 focus:outline-none focus:border-manasitra-500 focus:ring-1 focus:ring-manasitra-500 resize-none"
            placeholder={
              entryType === 'achievement'
                ? "What did you accomplish today? Big or small!"
                : "What went wrong today? How are you feeling?"
            }
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />

          <button
            type="submit"
            disabled={!content.trim() || loading}
            className="w-full bg-manasitra-600 hover:bg-manasitra-500 text-white font-medium py-3 rounded-xl shadow-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Saving...' : 'Save to Journal'}
          </button>
        </form>
      </div>

      {/* Right side: Past Entries */}
      <div className="flex-1 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl overflow-hidden flex flex-col">
        <h3 className="text-xl font-bold text-white mb-4">Past Entries</h3>
        
        <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar">
          {fetching ? (
            <div className="text-center text-slate-500 mt-10">Loading entries...</div>
          ) : entries.length === 0 ? (
            <div className="text-center text-slate-500 mt-10">
              No journal entries yet. Start writing!
            </div>
          ) : (
            entries.map((entry, idx) => (
              <div 
                key={idx} 
                className={`p-4 rounded-xl border ${
                  entry.entry_type === 'achievement' 
                    ? 'bg-emerald-500/10 border-emerald-500/20' 
                    : 'bg-red-500/10 border-red-500/20'
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className={`text-xs font-bold uppercase tracking-wider ${
                    entry.entry_type === 'achievement' ? 'text-emerald-400' : 'text-red-400'
                  }`}>
                    {entry.entry_type === 'achievement' ? '🌟 Achievement' : '🌧️ Bad Time'}
                  </span>
                  <span className="text-xs text-slate-500">
                    {new Date(entry.created_at).toLocaleDateString()}
                  </span>
                </div>
                <p className="text-sm text-slate-300 leading-relaxed">
                  {entry.content}
                </p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
