import { Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtDate, type ForecastDay } from '../api'

export interface PastDay {
  date: string
  tmax: number | null | undefined
  tmin: number | null | undefined
  rain: number | null | undefined
}

interface Point {
  date: string
  tmaxObs?: number | null
  tminObs?: number | null
  tmaxFc?: number | null
  tminFc?: number | null
  rainObs?: number | null
  rainFc?: number | null
}

/** FR-F5: recent days (solid) + forecast days (dashed). Gaps stay gaps.
 *  `pastLabel` names what the solid part is: register observations (replay) or reanalysis (live). */
export default function ForecastChart({ past, next, pastLabel = 'observed' }: { past: PastDay[]; next: ForecastDay[]; pastLabel?: string }) {
  const pts: Point[] = past.map((p) => ({ date: p.date, tmaxObs: p.tmax, tminObs: p.tmin, rainObs: p.rain }))
  const last = pts[pts.length - 1]
  if (last) Object.assign(last, { tmaxFc: last.tmaxObs, tminFc: last.tminObs }) // join the dashed line to the last solid point
  for (const d of next) {
    if (last && d.date <= last.date) continue
    pts.push({ date: d.date, tmaxFc: d.tmax_c, tminFc: d.tmin_c, rainFc: d.rain_mm })
  }
  const split = last?.date
  const tick = (d: string) => fmtDate(d, { day: 'numeric', month: 'short' })
  return (
    <div style={{ height: 300 }} role="img" aria-label={`Temperature and rainfall: recent days (${pastLabel}) and forecast`}>
      <ResponsiveContainer>
        <ComposedChart data={pts} margin={{ top: 8, right: 4, left: -8, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="date" tickFormatter={tick} interval="preserveStartEnd" minTickGap={18} />
          <YAxis yAxisId="t" domain={[15, 40]} ticks={[15, 20, 25, 30, 35, 40]} unit="°" width={44} />
          <YAxis yAxisId="r" orientation="right" domain={[0, (max: number) => Math.max(40, Math.ceil(max / 10) * 10)]} unit=" mm" width={56} />
          <Tooltip
            labelFormatter={(d) => fmtDate(String(d), { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })}
            formatter={(v, n) => [v === null || v === undefined ? 'not recorded' : `${Number(v).toFixed(1)}${String(n).includes('Rain') ? ' mm' : ' °C'}`, n]}
          />
          <Legend wrapperStyle={{ fontSize: 13 }} />
          {split && <ReferenceLine yAxisId="t" x={split} stroke="var(--text-3)" strokeDasharray="2 4" label={{ value: 'Forecast →', position: 'insideTopRight', fill: 'var(--text-2)', fontSize: 12 }} />}
          <Bar yAxisId="r" dataKey="rainObs" name={`Rain ${pastLabel}`} fill="var(--rain)" opacity={0.55} radius={[3, 3, 0, 0]} />
          <Bar yAxisId="r" dataKey="rainFc" name="Rain forecast" fill="var(--rain)" opacity={0.3} stroke="var(--rain-strong)" strokeDasharray="3 2" radius={[3, 3, 0, 0]} />
          <Line yAxisId="t" dataKey="tmaxObs" name={`Max ${pastLabel}`} stroke="var(--temp-max)" strokeWidth={2.5} dot={{ r: 3 }} connectNulls={false} />
          <Line yAxisId="t" dataKey="tminObs" name={`Min ${pastLabel}`} stroke="var(--temp-min)" strokeWidth={2.5} dot={{ r: 3 }} connectNulls={false} />
          <Line yAxisId="t" dataKey="tmaxFc" name="Max forecast" stroke="var(--temp-max)" strokeWidth={2.5} strokeDasharray="6 4" dot={{ r: 3, fill: 'var(--surface)' }} connectNulls={false} />
          <Line yAxisId="t" dataKey="tminFc" name="Min forecast" stroke="var(--temp-min)" strokeWidth={2.5} strokeDasharray="6 4" dot={{ r: 3, fill: 'var(--surface)' }} connectNulls={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
