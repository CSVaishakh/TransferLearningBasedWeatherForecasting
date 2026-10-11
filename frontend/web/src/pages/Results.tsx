import { Fragment, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { api, fmtDate, useApi, type Metrics, type TestRow } from '../api'
import { Load } from '../components/States'

type Target = 'tmax' | 'tmin' | 'rain'
const TARGETS: { key: Target; label: string; unit: string; metric: string; col: string }[] = [
  { key: 'tmax', label: 'Max temp', unit: '°C', metric: 'tmax_c', col: 'tmax_c' },
  { key: 'tmin', label: 'Min temp', unit: '°C', metric: 'tmin_c', col: 'tmin_c' },
  { key: 'rain', label: 'Rainfall', unit: 'mm', metric: 'rain', col: 'rain_mm' },
]
const STRATEGY_COLOURS: Record<string, string> = {
  'GNN from scratch (local only)': '#2f6fe4',
  'GNN pretrained 1→3': '#188a4b',
  'GNN pretrained 1→2→3': '#e07a1f',
  climatology: '#d6392f',
  xgboost: '#8b5cf6',
}
const RUNS: Record<string, string> = {
  's123_f1.0_s0': 'GNN pretrained 1→2→3',
  's13_f1.0_s0': 'GNN pretrained 1→3',
  'scratch_f1.0_s0': 'GNN from scratch (SJCET only)',
}
const pretty = (n: string) => n.replace('(local only)', '(SJCET only)').replace(/^\w/, (c) => c.toUpperCase())

function TargetPicker({ value, onChange }: { value: Target; onChange: (t: Target) => void }) {
  return (
    <div className="chips" role="radiogroup" aria-label="Target variable">
      {TARGETS.map((t) => (
        <button key={t.key} className="chip" role="radio" aria-checked={value === t.key} aria-pressed={value === t.key} onClick={() => onChange(t.key)}>
          {t.label}
        </button>
      ))}
    </div>
  )
}

/** FR-M1: models x metrics, best per column in bold green; baselines grouped separately. */
function MetricsTable({ m }: { m: Metrics }) {
  const cols = [
    { t: 'tmax_c', k: 'rmse', h: 'Max RMSE', u: '°C' }, { t: 'tmax_c', k: 'bias', h: 'Max bias', u: '°C' },
    { t: 'tmin_c', k: 'rmse', h: 'Min RMSE', u: '°C' }, { t: 'tmin_c', k: 'bias', h: 'Min bias', u: '°C' },
    { t: 'rain', k: 'rmse', h: 'Rain RMSE', u: 'mm' }, { t: 'rain', k: 'mae', h: 'Rain MAE', u: 'mm' },
    { t: 'rain', k: 'accuracy', h: 'Rain/no-rain', u: '' }, { t: 'tmax_c', k: 'n', h: 'Days', u: '' },
  ]
  const val = (mod: Metrics['models'][number], c: (typeof cols)[number]) => mod.metrics[c.t]?.[c.k]
  const best = cols.map((c) => {
    if (c.k === 'n') return undefined
    const vs = m.models.map((mod) => val(mod, c)).filter((v): v is number => v !== undefined)
    return c.k === 'accuracy' ? Math.max(...vs) : c.k === 'bias' ? Math.min(...vs.map(Math.abs)) : Math.min(...vs)
  })
  const groups: ['baseline' | 'gnn', string][] = [['gnn', 'Graph neural network'], ['baseline', 'Baselines']]
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {cols.map((c) => <th key={c.h} title={c.k === 'rmse' ? 'Root-mean-square error: typical size of a miss, penalising big misses' : c.k === 'bias' ? 'Average of forecast − observed (0 = no systematic error)' : c.k === 'accuracy' ? 'Share of days where "rain ≥ 2.5 mm or not" was called correctly' : undefined}>{c.h}{c.u && ` (${c.u})`}</th>)}
          </tr>
        </thead>
        <tbody>
          {groups.map(([g, label]) => (
            <Fragment key={g}>
              <tr className="group"><td colSpan={cols.length + 1}>{label}</td></tr>
              {m.models.filter((mod) => mod.group === g).map((mod) => (
                <tr key={mod.name}>
                  <td>{pretty(mod.name)}</td>
                  {cols.map((c, i) => {
                    const v = val(mod, c)
                    const isBest = v !== undefined && best[i] !== undefined && (c.k === 'bias' ? Math.abs(v) === best[i] : v === best[i])
                    return <td key={c.h} className={isBest ? 'best' : undefined}>{v === undefined ? '–' : c.k === 'n' ? v : c.k === 'accuracy' ? `${Math.round(v * 100)}%` : v.toFixed(2)}</td>
                  })}
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** FR-M2: RMSE vs fraction of SJCET training months, mean ± std over seeds. */
function Ablation({ m, target }: { m: Metrics; target: Target }) {
  const key = `${target}_rmse`
  const models = [...new Set(m.ablation.map((a) => a.model))]
  const fractions = [...new Set(m.ablation.map((a) => a.local_fraction))].sort((a, b) => a - b)
  const data = fractions.map((f) => {
    const row: Record<string, number | string | number[]> = { fraction: `${Math.round(f * 100)}%` }
    for (const mod of models) {
      const v = m.ablation.filter((a) => a.model === mod && a.local_fraction === f).map((a) => Number(a[key]))
      if (!v.length) continue
      const mu = v.reduce((s, x) => s + x, 0) / v.length
      const sd = Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / v.length)
      row[mod] = +mu.toFixed(3)
      row[`${mod}_err`] = [sd, sd]
    }
    return row
  })
  const persistence = m.models.find((x) => x.name === 'persistence')?.metrics[TARGETS.find((t) => t.key === target)!.metric]?.rmse
  const unit = TARGETS.find((t) => t.key === target)!.unit
  return (
    <div style={{ height: 320 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, left: -4, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="fraction" label={{ value: 'share of SJCET training months used', position: 'insideBottom', offset: -2, fontSize: 12 }} height={40} />
          <YAxis domain={['auto', 'auto']} unit={` ${unit}`} width={60} />
          <Tooltip formatter={(v, n) => [`${Number(v).toFixed(2)} ${unit}`, pretty(String(n))]} />
          <Legend formatter={(n) => pretty(String(n))} />
          {persistence !== undefined && <ReferenceLine y={persistence} stroke="var(--text-3)" strokeDasharray="4 4" label={{ value: `persistence ${persistence.toFixed(2)}`, position: 'insideBottomRight', fontSize: 12, fill: 'var(--text-2)' }} />}
          {models.map((mod) => (
            <Line key={mod} dataKey={mod} stroke={STRATEGY_COLOURS[mod] ?? 'var(--text-2)'} strokeWidth={mod.startsWith('GNN') ? 2.5 : 1.5}
              strokeDasharray={mod.startsWith('GNN') ? undefined : '5 4'} dot={{ r: 4 }}>
              <ErrorBar dataKey={`${mod}_err`} width={5} stroke={STRATEGY_COLOURS[mod]} />
            </Line>
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function Seasons({ m, target }: { m: Metrics; target: Target }) {
  const key = `${target}_rmse`
  const seasons = ['Winter (Jan–Feb)', 'Pre-monsoon (Mar–May)', 'SW monsoon (Jun–Sep)', 'NE monsoon (Oct–Dec)']
  const models = [...new Set(m.by_season.map((b) => b.model))]
  const data = seasons.map((s) => Object.fromEntries([['season', s.replace(/ \(.*/, '')], ...models.map((mod) => [mod, Number(m.by_season.find((b) => b.model === mod && b.season === s)?.[key])])]))
  const unit = TARGETS.find((t) => t.key === target)!.unit
  return (
    <div style={{ height: 280 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 4, left: -4, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="season" />
          <YAxis unit={` ${unit}`} width={60} />
          <Tooltip formatter={(v, n) => [`${Number(v).toFixed(2)} ${unit}`, pretty(String(n))]} />
          <Legend formatter={(n) => pretty(String(n))} />
          {models.map((mod) => <Bar key={mod} dataKey={mod} fill={STRATEGY_COLOURS[mod]} radius={[4, 4, 0, 0]} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function TestPlots({ rows, target }: { rows: TestRow[]; target: Target }) {
  const t = TARGETS.find((x) => x.key === target)!
  const pts = rows.filter((r) => r[`obs_${t.col}`] !== null).map((r) => ({ obs: r[`obs_${t.col}`] as number, pred: r[`pred_${t.col}`] as number, date: r.date }))
  const lo = Math.min(...pts.flatMap((p) => [p.obs, p.pred])), hi = Math.max(...pts.flatMap((p) => [p.obs, p.pred]))
  return (
    <div className="grid grid-2">
      <div style={{ height: 300 }} role="img" aria-label={`Predicted vs observed ${t.label}`}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 8, left: -4, bottom: 8 }}>
            <CartesianGrid stroke="var(--border)" />
            <XAxis type="number" dataKey="obs" name="Observed" unit={` ${t.unit}`} domain={[Math.floor(lo), Math.ceil(hi)]} />
            <YAxis type="number" dataKey="pred" name="Forecast" unit={` ${t.unit}`} domain={[Math.floor(lo), Math.ceil(hi)]} width={60} />
            <ZAxis range={[18, 18]} />
            <ReferenceLine segment={[{ x: Math.floor(lo), y: Math.floor(lo) }, { x: Math.ceil(hi), y: Math.ceil(hi) }]} stroke="var(--text-3)" strokeDasharray="4 4" />
            <Tooltip formatter={(v, n) => [`${Number(v).toFixed(1)} ${t.unit}`, n]} />
            <Scatter data={pts} fill="var(--accent)" fillOpacity={0.55} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div style={{ height: 300 }} role="img" aria-label={`${t.label} over the test period, forecast vs observed`}>
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 8, right: 8, left: -4, bottom: 8 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="date" tickFormatter={(d) => fmtDate(d, { month: 'short', year: '2-digit' })} minTickGap={40} />
            <YAxis domain={['auto', 'auto']} unit={` ${t.unit}`} width={60} />
            <Tooltip labelFormatter={(d) => fmtDate(String(d), { day: 'numeric', month: 'short', year: 'numeric' })} formatter={(v, n) => [v === null ? 'not recorded' : `${Number(v).toFixed(1)} ${t.unit}`, n]} />
            <Legend />
            <Line dataKey={`obs_${t.col}`} name="Observed" stroke="var(--observed)" dot={false} strokeWidth={1.2} connectNulls={false} isAnimationActive={false} />
            <Line dataKey={`pred_${t.col}`} name="Forecast" stroke={target === 'tmin' ? 'var(--temp-min)' : target === 'tmax' ? 'var(--temp-max)' : 'var(--rain-strong)'} strokeDasharray="4 3" dot={false} strokeWidth={1.2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function Results() {
  const metrics = useApi(api.metrics, [])
  const [target, setTarget] = useState<Target>('tmax')
  const [run, setRun] = useState('s123_f1.0_s0')
  const test = useApi(() => api.testPredictions(run), [run])
  const period = useMemo(() => metrics.data?.test_period.join(' to '), [metrics.data])
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Does pretraining help?</h1>
          <p>Next-day forecasts at SJCET, scored on the held-out test period {period ?? '2015–2016'}</p>
        </div>
        <TargetPicker value={target} onChange={setTarget} />
      </header>

      <section className="card">
        <div className="card-head"><h2>Error vs amount of SJCET data</h2></div>
        <p className="takeaway">
          Pretrained models learn SJCET from far less data: with 10% of its training months they already beat a model trained
          from scratch on all of it. On temperature, simple persistence ("tomorrow = today") is still hard to beat, because the
          register's readings change very little from day to day.
        </p>
        <Load q={metrics} height={320}>{(m) => <Ablation m={m} target={target} />}</Load>
      </section>

      <section className="card mt">
        <div className="card-head"><h2>All models <small>(all training months)</small></h2></div>
        <p className="takeaway">Lower RMSE is better; best per column in green. Persistence can only be scored on days that follow a reading, so it covers fewer days.</p>
        <Load q={metrics} height={260}>{(m) => <MetricsTable m={m} />}</Load>
      </section>

      <section className="card mt">
        <div className="card-head" style={{ flexWrap: 'wrap' }}>
          <h2>Forecast vs observed, test period</h2>
          <select className="inline" value={run} onChange={(e) => setRun(e.target.value)} aria-label="Model">
            {Object.entries(RUNS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          </select>
        </div>
        <p className="takeaway">Points on the dashed diagonal are perfect forecasts.</p>
        <Load q={test} height={300}>{(rows) => <TestPlots rows={rows} target={target} />}</Load>
      </section>

      <section className="card mt">
        <div className="card-head"><h2>Error by season</h2></div>
        <p className="takeaway">The monsoon seasons are the hardest to forecast for rain; temperatures vary least during the SW monsoon.</p>
        <Load q={metrics} height={280}>{(m) => <Seasons m={m} target={target} />}</Load>
      </section>
    </>
  )
}
