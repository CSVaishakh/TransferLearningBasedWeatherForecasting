import { ArrowDown, ArrowUp, CloudRain, CloudSun, Sun } from 'lucide-react'
import { fmtDate, type ForecastDay } from '../api'

/** FR-F2: rain icon when rain_prob >= 0.6 and rain >= 2.5 mm. */
export function WeatherIcon({ day, size = 72 }: { day: ForecastDay; size?: number }) {
  if (day.rain_prob >= 0.6 && day.rain_mm >= 2.5)
    return <CloudRain size={size} color="var(--rain-strong)" strokeWidth={1.6} aria-label="Rain likely" />
  if (day.rain_prob >= 0.3)
    return <CloudSun size={size} color="#f2a516" strokeWidth={1.6} aria-label="Partly cloudy, some chance of rain" />
  return <Sun size={size} color="#f2a516" strokeWidth={1.6} aria-label="Mostly dry" />
}

const signed = (v: number, unit: string) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(1)}${unit}`

function TempError({ fc, obs, label }: { fc: number; obs: number | null | undefined; label: string }) {
  if (obs === null || obs === undefined)
    return <span className="err na" title={`${label} not recorded in the register`}>{label} –</span>
  const e = fc - obs
  return (
    <span className={'err ' + (Math.abs(e) <= 1.5 ? 'good' : 'bad')} title={`${label}: forecast − observed`}>
      {label} {signed(e, '°C')}
    </span>
  )
}

export default function DayCard({ day, n, issued }: { day: ForecastDay; n: number; issued: string }) {
  const o = day.observed
  const mid = Math.round((day.tmax_c + day.tmin_c) / 2)
  const v = (x: number | null | undefined, unit: string) =>
    x === null || x === undefined ? <span title="not recorded in the register">–</span> : `${x.toFixed(1)}${unit}`
  return (
    <article className="card day-card" aria-label={`Forecast for ${day.date}`}>
      <div className="day-top">
        <div>
          <div className="day-date">{fmtDate(day.date, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}</div>
          <div className="day-issued">issued {fmtDate(issued, { day: 'numeric', month: 'short' })}, 1 day ahead</div>
        </div>
        <span className="badge">Day {n}</span>
      </div>
      <div className="day-main">
        <WeatherIcon day={day} />
        <div className="day-temp num">
          {mid}°C<small>mid-range</small>
        </div>
        <div className="hilo num">
          <span className="hi"><ArrowUp size={15} aria-label="max" /> {day.tmax_c.toFixed(0)}°</span>
          <span className="lo"><ArrowDown size={15} aria-label="min" /> {day.tmin_c.toFixed(0)}°</span>
        </div>
      </div>
      <div className="day-stats">
        <span className="lbl">Rainfall</span>
        <span className="lbl">Rain probability</span>
        <span className="stat">{day.rain_mm.toFixed(1)} mm</span>
        <span className="prob">
          <span className="stat">{Math.round(day.rain_prob * 100)}%</span>
          <span className="prob-bar" role="img" aria-label={`${Math.round(day.rain_prob * 100)}% chance of 2.5 mm or more`}>
            <div style={{ width: `${day.rain_prob * 100}%` }} />
          </span>
        </span>
      </div>
      {o && (
        <div className="obs">
          <div>
            <strong>Observed (register)</strong>
            <div className="vals num">
              {v(o.tmax_c, '°C')} / {v(o.tmin_c, '°C')}
              <br />
              {v(o.rain_mm, ' mm')}
            </div>
          </div>
          <div className="err-chips num">
            <TempError fc={day.tmax_c} obs={o.tmax_c} label="max" />
            <TempError fc={day.tmin_c} obs={o.tmin_c} label="min" />
            {o.rain_mm === null ? (
              <span className="err na" title="rain not recorded (weekend/holiday readings are multi-day totals)">rain –</span>
            ) : (
              <span className="err rain" title="rain: forecast − observed">{signed(day.rain_mm - o.rain_mm, ' mm')}</span>
            )}
          </div>
        </div>
      )}
    </article>
  )
}
