import { useEffect, useMemo, useState } from 'react'
import { BrainCircuit, CalendarDays, Clock3, FileText, Info, MapPin, Radio, RefreshCw } from 'lucide-react'
import { addDays, api, fmtDate, todayIST, useApi, type Forecast as Fc, type ForecastDay } from '../api'
import DayCard from '../components/DayCard'
import ForecastChart, { type PastDay } from '../components/ForecastChart'
import NodeMap from '../components/NodeMap'
import { Empty, ErrorState, Load, Skeleton } from '../components/States'

const DEMO_DATE = '2015-07-07' // monsoon week in the test period with readings on all days; replay default
type Mode = 'live' | 'replay'

function summary(d: ForecastDay, label: string) {
  const p = Math.round(d.rain_prob * 100)
  const rain =
    d.rain_prob >= 0.6 ? `rain likely (${p}% chance of 2.5 mm or more, about ${d.rain_mm.toFixed(0)} mm).`
    : d.rain_prob >= 0.3 ? `some chance of rain (${p}%).`
    : `mostly dry (${p}% chance of rain).`
  return `${label}: ${rain} Max ${d.tmax_c.toFixed(0)} °C, min ${d.tmin_c.toFixed(0)} °C.`
}

function ModeSwitch({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  return (
    <div className="chips" role="radiogroup" aria-label="Forecast mode">
      <button className="chip" role="radio" aria-checked={mode === 'live'} aria-pressed={mode === 'live'} onClick={() => setMode('live')}>
        <Radio size={13} style={{ verticalAlign: -2 }} /> Live
      </button>
      <button className="chip" role="radio" aria-checked={mode === 'replay'} aria-pressed={mode === 'replay'} onClick={() => setMode('replay')}>
        <Clock3 size={13} style={{ verticalAlign: -2 }} /> Replay (2015–2016)
      </button>
    </div>
  )
}

/** Live: the forecast from the latest available data, showing today and the next two days. */
function Live() {
  const q = useApi(api.live, [])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [fc, setFc] = useState<Fc | null>(null)
  useEffect(() => setFc(q.data), [q.data])
  const refresh = () => {
    setBusy(true)
    setErr(null)
    api.refreshLive().then(setFc).catch((e: Error) => setErr(e.message)).finally(() => setBusy(false))
  }
  const today = todayIST()

  if (q.error && !fc)
    return q.error.startsWith('404') ? (
      <div className="card">
        <Empty>
          No live forecast yet.
          <div style={{ marginTop: 12 }}>
            <button className="btn" onClick={refresh} disabled={busy}><RefreshCw size={14} /> {busy ? 'Fetching data and running the model…' : 'Make one now'}</button>
          </div>
          {err && <p className="small" style={{ color: 'var(--error)' }}>{err}</p>}
        </Empty>
      </div>
    ) : <div className="card"><ErrorState error={q.error} retry={q.retry} /></div>
  if (!fc) return <section className="grid grid-3">{[0, 1, 2].map((i) => <Skeleton key={i} height={300} />)}</section>

  const fromToday = fc.days.filter((d) => d.date >= today)
  const cards = (fromToday.length ? fromToday : fc.days.slice(-3)).slice(0, 3)
  const stale = !fromToday.length
  const past: PastDay[] = (fc.recent ?? []).map((r) => ({ date: r.date, tmax: r.tmax_c, tmin: r.tmin_c, rain: r.rain_mm }))
  const stations = fc.inputs?.noaa_stations ?? []
  return (
    <>
      <div className="card notice" role="note">
        <Info size={18} aria-hidden />
        <div>
          <strong>Live forecast, issued from data through {fmtDate(fc.issued, { day: 'numeric', month: 'long', year: 'numeric' })}.</strong>{' '}
          NASA POWER publishes about 3 days late, so "today" is {cards[0]?.lead_days ?? '–'} days ahead of the newest data.
          The SJCET register has no readings after 2016 and {stations.filter((s) => s.days_with_data > 0).length} of {stations.length} nearby
          NOAA stations have published recent data, so this forecast comes from regional reanalysis.
          {stale && ' This forecast no longer reaches today: refresh it.'}
          <div className="notice-actions">
            <button className="btn" onClick={refresh} disabled={busy}>
              <RefreshCw size={14} className={busy ? 'spin' : undefined} /> {busy ? 'Fetching latest data…' : 'Refresh with latest data'}
            </button>
            {fc.generated && <span className="small muted">generated {new Date(fc.generated).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' })}</span>}
          </div>
          {err && <p className="small" style={{ color: 'var(--error)', margin: '6px 0 0' }}>{err}</p>}
        </div>
      </div>
      <section className="grid grid-3 mt" aria-label="Today and the next two days">
        {cards.map((d, i) => (
          <DayCard key={d.date} day={d} n={i + 1} issued={fc.issued}
            badge={d.date === today ? 'Today' : d.date === addDays(today, 1) ? 'Tomorrow' : `+${d.lead_days} days`} />
        ))}
      </section>
      <Panels past={past} next={fc.days} pastLabel="reanalysis" summaryText={cards[0] ? summary(cards[0], cards[0].date === today ? 'Today' : fmtDate(cards[0].date, { weekday: 'long' })) : '–'}
        chartTitle={<>Temperature at SJCET <small>(last 14 days of reanalysis + {fc.days.length}-day forecast)</small></>} />
    </>
  )
}

/** Replay: pick a past issue date and compare the forecast with what the register recorded. */
function Replay({ model, setModel }: { model: string | null; setModel: (m: string) => void }) {
  const models = useApi(api.models, [])
  useEffect(() => {
    if (models.data && !model) setModel(models.data.find((m) => m.default)?.id ?? models.data[0]?.id)
  }, [models.data, model, setModel])
  const dates = useApi(() => (model ? api.forecastDates(model) : Promise.resolve([] as string[])), [model])
  const available = useMemo(() => new Set(dates.data ?? []), [dates.data])
  const [date, setDate] = useState<string | null>(null)
  useEffect(() => {
    if (dates.data?.length && !date) setDate(available.has(DEMO_DATE) ? DEMO_DATE : dates.data[Math.max(0, dates.data.length - 1)])
  }, [dates.data, date, available])
  // forecasts issued in the 14 days before give the "observed" history; the one issued on `date` gives the cards
  const range = useApi(
    () => (model && date ? api.forecastRange(addDays(date, -14), date, model) : Promise.resolve([] as Fc[])),
    [model, date],
  )
  const current = range.data?.find((f) => f.issued === date) ?? null
  const past: PastDay[] = (range.data ?? []).filter((f) => f.issued < date!).map((f) => {
    const d = f.days[0]
    return { date: d.date, tmax: d.observed?.tmax_c, tmin: d.observed?.tmin_c, rain: d.observed?.rain_mm }
  })
  const first = dates.data?.[0]
  const lastDate = dates.data?.[dates.data.length - 1]

  return (
    <>
      <div className="controls" style={{ marginBottom: 20 }}>
        <div className="control">
          <CalendarDays size={26} color="var(--text-2)" aria-hidden />
          <div>
            <label htmlFor="replay-date">Forecast issued on</label>
            <input id="replay-date" type="date" value={date ?? ''} min={first} max={lastDate} onChange={(e) => e.target.value && setDate(e.target.value)} />
          </div>
        </div>
        {models.data && models.data.length > 1 && (
          <div className="control">
            <BrainCircuit size={24} color="var(--text-2)" aria-hidden />
            <div>
              <label htmlFor="model">Model</label>
              <select id="model" value={model ?? ''} onChange={(e) => setModel(e.target.value)}>
                {models.data.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </div>
          </div>
        )}
        <div className="control replay">
          <Clock3 size={26} aria-hidden />
          <div>
            <strong>Replay mode</strong>
            <span>Compared with what the register recorded</span>
          </div>
        </div>
      </div>

      {models.error ? <div className="card"><Load q={models}>{() => null}</Load></div>
        : dates.error ? <div className="card"><Load q={dates}>{() => null}</Load></div>
        : date && dates.data && !available.has(date) ? (
          <div className="card"><Empty>No forecast was issued on {fmtDate(date, { day: 'numeric', month: 'long', year: 'numeric' })}. Replay covers {first} to {lastDate}.</Empty></div>
        ) : (
          <section className="grid grid-3" aria-label="First three forecast days">
            {range.loading || !current ? [0, 1, 2].map((i) => <Skeleton key={i} height={300} />)
              : current.days.slice(0, 3).map((d, i) => <DayCard key={d.date} day={d} n={i + 1} issued={current.issued} />)}
          </section>
        )}
      <Panels past={past} next={current?.days ?? []} pastLabel="observed" loading={range.loading}
        summaryText={current ? summary(current.days[0], fmtDate(current.days[0].date, { weekday: 'long', day: 'numeric', month: 'short' })) : '–'}
        chartTitle={<>Temperature forecast vs observed <small>(last 14 days + {current?.days.length ?? 3}-day forecast)</small></>}
        modelLabel={models.data?.find((m) => m.id === model)?.label} />
    </>
  )
}

function Panels({ past, next, pastLabel, summaryText, chartTitle, loading, modelLabel }: {
  past: PastDay[]; next: ForecastDay[]; pastLabel: string; summaryText: string; chartTitle: React.ReactNode; loading?: boolean; modelLabel?: string
}) {
  const graph = useApi(api.graph, [])
  const [edges, setEdges] = useState(false)
  return (
    <>
      <section className="grid grid-main mt">
        <div className="card">
          <div className="card-head"><h2>{chartTitle}</h2></div>
          {loading ? <Skeleton height={300} /> : past.length || next.length ? <ForecastChart past={past} next={next} pastLabel={pastLabel} /> : <Empty>No data around this date.</Empty>}
        </div>
        <div className="card">
          <div className="card-head">
            <h2>Weather network map</h2>
            <label className="switch"><input type="checkbox" checked={edges} onChange={(e) => setEdges(e.target.checked)} /> Show edges</label>
          </div>
          <Load q={graph} height={360}>
            {(g) => (
              <div className="map-wrap">
                <NodeMap graph={g} showEdges={edges} />
                <div className="legend" aria-hidden>
                  <span><i className="dot" style={{ background: '#e0453a' }} /> SJCET (target)</span>
                  <span><i className="dot" style={{ background: '#2f6fe4' }} /> NOAA station</span>
                  <span><i className="dot" style={{ background: '#fff', borderColor: '#7d8a9c' }} /> Reanalysis node</span>
                </div>
              </div>
            )}
          </Load>
        </div>
      </section>
      <section className="grid grid-3 mt">
        <div className="card info-card">
          <div className="info-icon"><FileText size={24} /></div>
          <div><h3>Summary</h3><p>{summaryText}</p></div>
        </div>
        <div className="card info-card">
          <div className="info-icon"><BrainCircuit size={24} /></div>
          <div>
            <h3>Model used</h3>
            <p className="strong">{modelLabel ?? 'GNN, Region → Stations → SJCET (1→2→3)'}</p>
            <p>Pretrained on NASA POWER reanalysis and NOAA stations, fine-tuned on the SJCET register (2010–2013); forecasts 6 days ahead</p>
          </div>
        </div>
        <div className="card info-card">
          <div className="info-icon"><MapPin size={24} /></div>
          <div>
            <h3>Location</h3>
            <p className="strong">St. Joseph's College of Engineering and Technology</p>
            <p>Palai, Kerala (9.727° N, 76.727° E)</p>
          </div>
        </div>
      </section>
    </>
  )
}

export default function Forecast() {
  const [mode, setMode] = useState<Mode>(() => (new URLSearchParams(location.search).get('mode') === 'replay' ? 'replay' : 'live'))
  const [model, setModel] = useState<string | null>(null)
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Weather Forecast for SJCET</h1>
          <p>Graph neural network forecasts, pretrained on regional reanalysis and fine-tuned on the digitized SJCET register</p>
        </div>
        <ModeSwitch mode={mode} setMode={setMode} />
      </header>
      {mode === 'live' ? <Live /> : <Replay model={model} setModel={setModel} />}
    </>
  )
}
