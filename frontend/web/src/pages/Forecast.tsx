import { useEffect, useMemo, useState } from 'react'
import { BrainCircuit, CalendarDays, Clock3, FileText, MapPin } from 'lucide-react'
import { addDays, api, fmtDate, useApi, type Forecast as Fc, type ForecastDay } from '../api'
import DayCard from '../components/DayCard'
import ForecastChart from '../components/ForecastChart'
import NodeMap from '../components/NodeMap'
import { Empty, Load, Skeleton } from '../components/States'

const DEMO_DATE = '2015-07-07' // monsoon week in the test period with readings on all 3 days; default when available

function summary(d: ForecastDay) {
  const p = Math.round(d.rain_prob * 100)
  const rain =
    d.rain_prob >= 0.6 ? `Rain likely (${p}% chance of 2.5 mm or more, about ${d.rain_mm.toFixed(0)} mm).`
    : d.rain_prob >= 0.3 ? `Some chance of rain (${p}%).`
    : `Mostly dry (${p}% chance of rain).`
  return `${rain} Max ${d.tmax_c.toFixed(0)} °C, min ${d.tmin_c.toFixed(0)} °C.`
}

export default function Forecast() {
  const models = useApi(api.models, [])
  const [model, setModel] = useState<string | null>(null)
  useEffect(() => {
    if (models.data && !model) setModel(models.data.find((m) => m.default)?.id ?? models.data[0]?.id ?? null)
  }, [models.data, model])

  const dates = useApi(() => (model ? api.forecastDates(model) : Promise.resolve([] as string[])), [model])
  const available = useMemo(() => new Set(dates.data ?? []), [dates.data])
  const [date, setDate] = useState<string | null>(null)
  useEffect(() => {
    if (dates.data?.length && !date) setDate(available.has(DEMO_DATE) ? DEMO_DATE : dates.data[Math.max(0, dates.data.length - 3)])
  }, [dates.data, date, available])

  const range = useApi(
    () => (model && date ? api.forecastRange(addDays(date, -14), addDays(date, 2), model) : Promise.resolve([] as Fc[])),
    [model, date],
  )
  const graph = useApi(api.graph, [])
  const [edges, setEdges] = useState(false)

  const past = (range.data ?? []).filter((f) => f.issued < date!)
  const next = (range.data ?? []).filter((f) => f.issued >= date!)
  const modelLabel = models.data?.find((m) => m.id === model)?.label ?? 'GNN'
  const first = dates.data?.[0]
  const lastDate = dates.data?.[dates.data.length - 1]

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Weather Forecast for SJCET</h1>
          <p>Graph neural network forecasts, pretrained on regional reanalysis and fine-tuned on the digitized SJCET register</p>
        </div>
        <div className="controls">
          <div className="control">
            <CalendarDays size={26} color="var(--text-2)" aria-hidden />
            <div>
              <label htmlFor="replay-date">Replay date (forecast issued)</label>
              <input
                id="replay-date"
                type="date"
                value={date ?? ''}
                min={first}
                max={lastDate}
                onChange={(e) => e.target.value && setDate(e.target.value)}
              />
            </div>
          </div>
          {models.data && models.data.length > 1 && (
            <div className="control">
              <BrainCircuit size={24} color="var(--text-2)" aria-hidden />
              <div>
                <label htmlFor="model">Model</label>
                <select id="model" value={model ?? ''} onChange={(e) => setModel(e.target.value)}>
                  {models.data.map((m) => (
                    <option key={m.id} value={m.id}>{m.label}</option>
                  ))}
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
      </header>

      {models.error ? (
        <div className="card"><Load q={models}>{() => null}</Load></div>
      ) : dates.error ? (
        <div className="card"><Load q={dates}>{() => null}</Load></div>
      ) : date && dates.data && !available.has(date) ? (
        <div className="card">
          <Empty>
            No forecast was issued on {fmtDate(date, { day: 'numeric', month: 'long', year: 'numeric' })}. Replay forecasts cover the
            test period {first} to {lastDate}.
          </Empty>
        </div>
      ) : (
        <section className="grid grid-3" aria-label="Next three days">
          {range.loading || !range.data
            ? [0, 1, 2].map((i) => <Skeleton key={i} height={300} />)
            : next.map((f, i) => <DayCard key={f.issued} day={f.days[0]} n={i + 1} issued={f.issued} />)}
          {range.data && !range.loading && next.length < 3 && (
            <div className="card"><Empty>No forecast beyond {lastDate}: the test period ends there.</Empty></div>
          )}
        </section>
      )}

      <section className="grid grid-main mt">
        <div className="card">
          <div className="card-head">
            <h2>Temperature forecast vs observed <small>(last 14 days + next 3 days)</small></h2>
          </div>
          <Load q={range} height={300}>
            {() => (past.length || next.length ? <ForecastChart past={past} next={next} /> : <Empty>No data around this date.</Empty>)}
          </Load>
        </div>
        <div className="card">
          <div className="card-head">
            <h2>Weather network map</h2>
            <label className="switch">
              <input type="checkbox" checked={edges} onChange={(e) => setEdges(e.target.checked)} /> Show edges
            </label>
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
          <div>
            <h3>Tomorrow's summary</h3>
            <p>{next[0] ? summary(next[0].days[0]) : '–'}</p>
          </div>
        </div>
        <div className="card info-card">
          <div className="info-icon"><BrainCircuit size={24} /></div>
          <div>
            <h3>Model used</h3>
            <p className="strong">{modelLabel}</p>
            <p>Pretrained on NASA POWER reanalysis and NOAA stations, fine-tuned on the SJCET register (2010–2013)</p>
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
