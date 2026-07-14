import { Routes, Route, NavLink } from 'react-router-dom'
import ChatPage from './pages/ChatPage'
import DashboardPage from './pages/DashboardPage'
import JournalPage from './pages/JournalPage'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Nav */}
      <nav className="bg-slate-900 border-b border-slate-800 px-6 py-3 flex items-center gap-6">
        <span className="text-manasitra-400 font-bold text-xl tracking-tight">
          🧠 Manasitra
        </span>
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `text-sm font-medium transition-colors ${
              isActive ? 'text-manasitra-400' : 'text-slate-400 hover:text-slate-200'
            }`
          }
        >
          Chat
        </NavLink>
        <NavLink
          to="/dashboard"
          className={({ isActive }) =>
            `text-sm font-medium transition-colors ${
              isActive ? 'text-manasitra-400' : 'text-slate-400 hover:text-slate-200'
            }`
          }
        >
          Mood Dashboard
        </NavLink>
        <NavLink
          to="/journal"
          className={({ isActive }) =>
            `text-sm font-medium transition-colors ${
              isActive ? 'text-manasitra-400' : 'text-slate-400 hover:text-slate-200'
            }`
          }
        >
          Journal
        </NavLink>
      </nav>

      {/* Pages */}
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/journal" element={<JournalPage />} />
        </Routes>
      </main>
    </div>
  )
}
