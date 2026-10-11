import { useMemo, useState } from 'react'
import { Download } from 'lucide-react'
import {
  Bar, BarChart, Brush, CartesianGrid, ComposedChart, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, fmtDate, useApi, type HistoryRow } from '../api'
import { Load } from '../components/States'

type Var = 'tmax_c' | 'tmin_c' | 'dry_bulb_c' | 'wet_bulb_c' | 'rh_pct' | 'rain_mm'
const VARS: { key: Var; label: string; unit: string; color: string }[] = [
  { key: 'tmax_c', label: 'Max temp', unit: '°C', color: 'var(--temp-max)' },
  { key: 'tmin_c', label: 'Min temp', unit: '°C', color: 'var(--temp-min)' },
  { key: 'dry_bulb_c', label: 'Dry bulb', unit: '°C', color: '#d9822b' },
  { key: 'wet_bulb_c', label: 'Wet bulb', unit: '°C', color: '#1f9d8b' },
  { key: 'rh_pct', label: 'Rel. humidity', unit: '%', color: '#8b5cf6' },
  { key: 'rain_mm', label: 'Rainfall', unit: 'mm', color: 'var(--rain-strong)' },
]
const FIELDS = ['tmax_c', 'tmin_c', 'rain_mm', 'dry_bulb_c', 'wet_bulb_c'] as const
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const year = (d: string) => Number(d.slice(0, 4))
const doy = (d: string) => {
  const t = Date.UTC(year(d), Number(d.slice(5, 7)) - 1, Number(d.slice(8, 10)))
  return Math.round((t - Date.UTC(year(d), 0, 1)) / 864e5)
}
const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length
function trend(xs: number[], ys: number[]) {
  const mx = mean(xs), my = mean(ys)
  const b = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0)
  return { slope: b, at: (x: number) => my + b * (x - mx) }
}

/** FR-H2: years x day-of-year, coloured by how many of the 5 fields were recorded. */
function CoverageHeatmap({ rows }: { rows: HistoryRow[] }) {
  const years = [...new Set(rows.map((r) => year(r.date)))].sort()
  const cw = 3, ch = 18, lw = 40
  const colour = (r: HistoryRow) => {
    const n = FIELDS.filter((f) => r[f] !== null).length
    if (n === 0) return null
    if (r.qc_flag === 'suspect') return 'var(--suspect)'
    return n >= 4 ? 'var(--ok)' : n >= 2 ? '#7cc49b' : '#bfe3cd'
  }
  return (
    <>
      <svg className="heatmap" viewBox={`0 0 ${lw + 366 * cw} ${years.length * (ch + 4) + 18}`} role="img"
        aria-label="Register coverage: one row per year, one column per day; gaps are days with no reading">
        {MONTHS.map((m, i) => (
          <text key={m} x={lw + doy(`2001-${String(i + 1).padStart(2, '0')}-01`) * cw} y={11} className="recharts-text" fontSize={10}>{m}</text>
        ))}
        {years.map((y, yi) => (
          <g key={y} transform={`translate(0 ${18 + yi * (ch + 4)})`}>
            <text x={0} y={ch - 5} className="recharts-text" fontSize={11}>{y}</text>
            <rect x={lw} y={0} width={366 * cw} height={ch} fill="var(--surface-2)" rx={3} />
            {rows.filter((r) => year(r.date) === y).map((r) => {
              const c = colour(r)
              return c && <rect key={r.date} x={lw + doy(r.date) * cw} y={0} width={cw} height={ch} fill={c}><title>{`${r.date}: ${r.qc_flag}`}</title></rect>
            })}
          </g>
        ))}
      </svg>
      <div className="heat-legend">
        <span><i style={{ background: 'var(--ok)' }} />4–5 fields</span>
        <span><i style={{ background: '#7cc49b' }} />2–3 fields</span>
        <span><i style={{ background: '#bfe3cd' }} />1 field</span>
        <span><i style={{ background: 'var(--suspect)' }} />suspect (QC flag)</span>
        <span><i style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }} />no reading (weekends, holidays, missing months)</span>
      </div>
    </>
  )
}

function SuspectDot(props: { cx?: number; cy?: number; payload?: HistoryRow; value?: number | null }) {
  const { cx, cy, payload, value } = props
  if (cx === undefined || cy === undefined || value === null || payload?.qc_flag !== 'suspect') return <g />
  return <circle cx={cx} cy={cy} r={3.5} fill="var(--surface)" stroke="var(--suspect)" strokeWidth={1.5} />
}

function downloadCsv(rows: HistoryRow[], name: string) {
  const cols = Object.keys(rows[0]) as (keyof HistoryRow)[]
  const esc = (v: unknown) => (v === null || v === undefined ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

function HistoryView({ rows }: { rows: HistoryRow[] }) {
  const years = useMemo(() => [...new Set(rows.map((r) => year(r.date)))].sort(), [rows])
  const [from, setFrom] = useState(years[0])
  const [to, setTo] = useState(years[years.length - 1])
  const [vars, setVars] = useState<Var[]>(['tmax_c', 'tmin_c'])
  const sel = useMemo(() => rows.filter((r) => year(r.date) >= from && year(r.date) <= to), [rows, from, to])

  // monthly totals (the gauge accumulates between readings, so the sum of readings is the month's total);
  // months with no reading at all are left out rather than counted as 0
  const monthly = useMemo(() => {
    const tot = new Map<string, number>()
    for (const r of rows) if (r.rain_mm !== null) tot.set(r.date.slice(0, 7), (tot.get(r.date.slice(0, 7)) ?? 0) + r.rain_mm)
    return MONTHS.map((m, i) => {
      const v = [...tot].filter(([k]) => Number(k.slice(5)) === i + 1).map(([, x]) => x)
      const mu = v.length ? mean(v) : 0
      return { month: m, mean: Math.round(mu), range: v.length ? [mu - Math.min(...v), Math.max(...v) - mu] : [0, 0], n: v.length }
    })
  }, [rows])

  const annual = useMemo(() => {
    const full = years.filter((y) => rows.some((r) => r.date.startsWith(`${y}-01`)))  // complete years only
    const out = full.map((y) => {
      const yr = rows.filter((r) => year(r.date) === y)
      const pick = (k: 'tmax_c' | 'tmin_c') => yr.map((r) => r[k]).filter((x): x is number => x !== null && true)
      return {
        year: y,
        rain: Math.round(yr.reduce((s, r) => s + (r.rain_mm ?? 0), 0)),
        tmax: +mean(pick('tmax_c')).toFixed(2),
        tmin: +mean(pick('tmin_c')).toFixed(2),
      }
    })
    const xs = out.map((o) => o.year)
    const tr = (k: 'rain' | 'tmax' | 'tmin') => trend(xs, out.map((o) => o[k]))
    const t = { rain: tr('rain'), tmax: tr('tmax'), tmin: tr('tmin') }
    return {
      data: out.map((o) => ({ ...o, rainTrend: +t.rain.at(o.year).toFixed(0), tmaxTrend: +t.tmax.at(o.year).toFixed(2), tminTrend: +t.tmin.at(o.year).toFixed(2) })),
      slope: { tmax: t.tmax.slope * 10, tmin: t.tmin.slope * 10, rain: t.rain.slope },
    }
  }, [rows, years])

  const observed = rows.filter((r) => FIELDS.some((f) => r[f] !== null))
  const suspect = observed.filter((r) => r.qc_flag === 'suspect').length
  const tempVars = vars.filter((v) => v !== 'rain_mm')

  return (
    <>
      <section className="card">
        <div className="card-head">
          <h2>Register coverage <small>{rows[0].date} to {rows[rows.length - 1].date}</small></h2>
          <span className="muted small num">{observed.length} days with readings · {suspect} flagged suspect</span>
        </div>
        <p className="takeaway">Readings were taken on working days only. November and December 2012 have blank register pages.</p>
        <CoverageHeatmap rows={rows} />
      </section>

      <section className="card mt">
        <div className="card-head" style={{ flexWrap: 'wrap' }}>
          <h2>Daily readings</h2>
          <div className="controls" style={{ alignItems: 'center' }}>
            <label className="small muted">From <select className="inline" value={from} onChange={(e) => setFrom(+e.target.value)}>{years.map((y) => <option key={y}>{y}</option>)}</select></label>
            <label className="small muted">to <select className="inline" value={to} onChange={(e) => setTo(+e.target.value)}>{years.filter((y) => y >= from).map((y) => <option key={y}>{y}</option>)}</select></label>
            <button className="btn" onClick={() => downloadCsv(sel, `sjcet_${from}-${to}.csv`)}><Download size={15} /> CSV</button>
          </div>
        </div>
        <div className="chips" role="group" aria-label="Variables to plot" style={{ marginBottom: 12 }}>
          {VARS.map((v) => (
            <button key={v.key} className="chip" aria-pressed={vars.includes(v.key)}
              onClick={() => setVars((s) => (s.includes(v.key) ? s.filter((x) => x !== v.key) : [...s, v.key]))}>
              {v.label} ({v.unit})
            </button>
          ))}
        </div>
        <p className="takeaway">Hollow amber points are values flagged suspect during QC. Gaps are days with no reading. Values are shown against the date written in the register (08:30 reading).</p>
        <div style={{ height: 340 }}>
          <ResponsiveContainer>
            <ComposedChart data={sel} margin={{ top: 6, right: 4, left: -8, bottom: 0 }}>
              <CartesianGrid stroke="var(--border)" vertical={false} />
              <XAxis dataKey="date" tickFormatter={(d) => fmtDate(d, { month: 'short', year: '2-digit' })} minTickGap={40} />
              {tempVars.length > 0 && <YAxis yAxisId="t" domain={['auto', 'auto']} width={44} />}
              {vars.includes('rain_mm') && <YAxis yAxisId="r" orientation="right" domain={[0, 'auto']} unit=" mm" width={60} />}
              <Tooltip
                labelFormatter={(d) => fmtDate(String(d), { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}
                formatter={(v, n, p) => [v === null ? 'not recorded' : `${v}${VARS.find((x) => x.label === n)?.unit === 'mm' ? ' mm' : VARS.find((x) => x.label === n)?.unit === '%' ? ' %' : ' °C'}${(p.payload as HistoryRow).qc_flag === 'suspect' ? ' (suspect)' : ''}`, n]}
              />
              <Legend />
              {vars.includes('rain_mm') && <Bar yAxisId="r" dataKey="rain_mm" name="Rainfall" fill="var(--rain)" isAnimationActive={false} />}
              {tempVars.map((k) => {
                const v = VARS.find((x) => x.key === k)!
                return <Line key={k} yAxisId="t" dataKey={k} name={v.label} stroke={v.color} dot={<SuspectDot />} strokeWidth={1.5} connectNulls={false} isAnimationActive={false} />
              })}
              <Brush dataKey="date" height={24} stroke="var(--accent)" tickFormatter={(d) => fmtDate(d, { month: 'short', year: '2-digit' })} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="grid grid-2 mt">
        <div className="card">
          <div className="card-head"><h2>Monthly rainfall</h2></div>
          <p className="takeaway">Mean recorded total per month (bars) with the driest and wettest year (whiskers). The SW monsoon (Jun–Sep) and NE monsoon (Oct–Nov) peaks stand out.</p>
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={monthly} margin={{ top: 6, right: 4, left: -4, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="month" />
                <YAxis unit=" mm" width={60} />
                <Tooltip formatter={(v, _n, p) => [`${v} mm (mean of ${(p.payload as { n: number }).n} years)`, 'Monthly total']} />
                <Bar dataKey="mean" name="Mean monthly total" fill="var(--rain)" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  <ErrorBar dataKey="range" width={6} stroke="var(--text-2)" strokeWidth={1.5} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card">
          <div className="card-head"><h2>Annual means</h2></div>
          <p className="takeaway num">
            Trend: max {annual.slope.tmax >= 0 ? '+' : ''}{annual.slope.tmax.toFixed(1)} °C/decade, min {annual.slope.tmin >= 0 ? '+' : ''}{annual.slope.tmin.toFixed(1)} °C/decade.
            Only {annual.data.length} complete years, so treat trends as indicative.
          </p>
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={annual.data} margin={{ top: 6, right: 4, left: -8, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="year" />
                <YAxis domain={['auto', 'auto']} unit="°" width={44} />
                <Tooltip formatter={(v, n) => [`${v} °C`, n]} />
                <Legend />
                <Line dataKey="tmax" name="Mean max" stroke="var(--temp-max)" strokeWidth={2.5} isAnimationActive={false} />
                <Line dataKey="tmin" name="Mean min" stroke="var(--temp-min)" strokeWidth={2.5} isAnimationActive={false} />
                <Line dataKey="tmaxTrend" name="Max trend" stroke="var(--temp-max)" strokeDasharray="5 4" dot={false} legendType="none" isAnimationActive={false} />
                <Line dataKey="tminTrend" name="Min trend" stroke="var(--temp-min)" strokeDasharray="5 4" dot={false} legendType="none" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="card mt">
        <div className="card-head"><h2>Annual rainfall</h2></div>
        <p className="takeaway">Recorded total per complete year, with a linear trend. 2012 is low partly because its November and December pages are blank.</p>
        <div style={{ height: 240 }}>
          <ResponsiveContainer>
            <ComposedChart data={annual.data} margin={{ top: 6, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--border)" vertical={false} />
              <XAxis dataKey="year" />
              <YAxis unit=" mm" width={70} />
              <Tooltip formatter={(v, n) => [`${v} mm`, n]} />
              <Bar dataKey="rain" name="Annual total" fill="var(--rain)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              <Line dataKey="rainTrend" name="Trend" stroke="var(--text-2)" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>
    </>
  )
}

export default function History() {
  const q = useApi(api.history, [])
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Register History</h1>
          <p>The SJCET weather register, digitized from the handwritten pages (July 2010 – December 2016)</p>
        </div>
      </header>
      <Load q={q} height={400}>{(rows) => (rows.length ? <HistoryView rows={rows} /> : <div className="card state">No register data yet.</div>)}</Load>
    </>
  )
}
