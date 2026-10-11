import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { BarChart3, ClipboardCheck, Cloud, CloudSun, Info, LineChart, Moon, Sun } from 'lucide-react'
import Forecast from './pages/Forecast'
import History from './pages/History'
import Results from './pages/Results'
import Review from './pages/Review'
import About from './pages/About'

const NAV = [
  { to: '/forecast', label: 'Forecast', icon: Cloud },
  { to: '/history', label: 'History', icon: BarChart3 },
  { to: '/results', label: 'Results', icon: LineChart },
  { to: '/about', label: 'About', icon: Info },
]

type Theme = 'light' | 'dark' | null

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem('theme') as Theme) ?? null
    } catch {
      return null
    }
  })
  useEffect(() => {
    const el = document.documentElement
    if (theme) el.dataset.theme = theme
    else delete el.dataset.theme
    try {
      if (theme) localStorage.setItem('theme', theme)
      else localStorage.removeItem('theme')
    } catch {
      /* storage unavailable */
    }
  }, [theme])
  const dark = theme === 'dark' || (theme === null && matchMedia('(prefers-color-scheme: dark)').matches)
  return [theme, () => setTheme(dark ? 'light' : 'dark')]
}

export default function App() {
  const [theme, toggle] = useTheme()
  const dark = theme === 'dark' || (theme === null && matchMedia('(prefers-color-scheme: dark)').matches)
  return (
    <div className="app">
      <nav className="sidebar" aria-label="Main">
        <NavLink to="/forecast" className="brand">
          <CloudSun size={40} color="var(--rain-strong)" aria-hidden />
          <div>
            <div className="brand-title">SJCET</div>
            <div className="brand-sub">Weather Forecasting</div>
          </div>
        </NavLink>
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
            <Icon size={20} aria-hidden /> {label}
          </NavLink>
        ))}
        <div className="nav-sep">Team tools</div>
        <NavLink to="/review" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
          <ClipboardCheck size={20} aria-hidden /> Review Tool
        </NavLink>
        <div className="sidebar-foot">
          <strong>St. Joseph's College of Engineering and Technology</strong>
          Palai, Kerala
          <div>
            <button className="theme-btn" onClick={toggle} aria-label="Toggle dark mode">
              {dark ? <Sun size={14} /> : <Moon size={14} />} {dark ? 'Light' : 'Dark'} mode
            </button>
          </div>
        </div>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/forecast" replace />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/history" element={<History />} />
          <Route path="/results" element={<Results />} />
          <Route path="/review" element={<Review />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/forecast" replace />} />
        </Routes>
      </main>
    </div>
  )
}
