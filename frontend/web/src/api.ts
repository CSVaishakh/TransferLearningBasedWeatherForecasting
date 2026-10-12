// Typed wrappers for the FastAPI endpoints (shapes follow docs/data-contracts.md).
import { useCallback, useEffect, useState } from 'react'

export interface Observed {
  tmax_c: number | null
  tmin_c: number | null
  rain_mm: number | null
  tmean_c: number | null
  dewpoint_c: number | null
  rh_pct: number | null
}

export interface ForecastDay {
  date: string
  lead_days?: number
  tmax_c: number
  tmin_c: number
  tmean_c: number
  dewpoint_c: number
  rh_pct: number
  rain_mm: number
  rain_prob: number
  rain_expected: boolean
  observed?: Observed
}

export interface RecentDay {
  date: string
  tmax_c: number | null
  tmin_c: number | null
  rain_mm: number | null
}

export interface Forecast {
  version: number
  mode?: 'live'
  generated?: string
  data_through?: string
  recent?: RecentDay[]
  inputs?: { reanalysis_nodes: number; noaa_stations: { id: string; name: string; days_with_data: number }[]; sjcet_register: string }
  issued: string
  station: { id: string; name: string; lat: number; lon: number }
  model: { name: string; run_id: string }
  days: ForecastDay[]
  notes?: string
}

export interface ModelInfo {
  id: string
  label: string
  default: boolean
}

export interface GraphNode {
  id: number
  kind: 'power' | 'gsod' | 'sjcet' | 'era5'
  region: string
  lat: number
  lon: number
  name?: string
}

export interface Graph {
  nodes: GraphNode[]
  edges: [number, number][]
}

export interface HistoryRow {
  date: string
  rain_mm: number | null
  dry_bulb_c: number | null
  wet_bulb_c: number | null
  tmin_c: number | null
  tmax_c: number | null
  rh_pct: number | null
  obs_time: string | null
  source_page: string | null
  qc_flag: 'ok' | 'suspect' | 'corrected' | null
  qc_note: string | null
}

export type MetricSet = Record<string, number>

export interface Metrics {
  version: number
  horizon?: number
  test_period: [string, string]
  local_station: { id: string; name: string }
  models: { name: string; group: 'baseline' | 'gnn'; metrics: Record<string, MetricSet>; by_lead?: Record<string, MetricSet>[]; by_lead_hidden?: Record<string, MetricSet>[] }[]
  ablation: ({ model: string; local_fraction: number; seed: number; lead?: number } & Record<string, number | string>)[]
  by_season: ({ model: string; season: string } & Record<string, number | string>)[]
}

export interface TestRow {
  date: string
  rain_prob: number | null
  [key: string]: string | number | null
}

export interface ReviewStatus {
  available: boolean
  cells: number
  queue?: number
  reason?: string
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let detail = r.statusText
    try {
      detail = (await r.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(`${r.status}: ${detail}`)
  }
  return r.json() as Promise<T>
}

export const api = {
  models: () => get<ModelInfo[]>('/api/models'),
  forecastDates: (model: string) => get<string[]>(`/api/forecast/dates?model=${model}`),
  forecastRange: (from: string, to: string, model: string) =>
    get<Forecast[]>(`/api/forecast/range?from=${from}&to=${to}&model=${model}`),
  graph: () => get<Graph>('/api/graph'),
  metrics: () => get<Metrics>('/api/metrics'),
  history: () => get<HistoryRow[]>('/api/history'),
  testPredictions: (model: string, lead = 1) => get<TestRow[]>(`/api/results/test?model=${model}&lead=${lead}`),
  forecast: (date: string, model: string) => get<Forecast>(`/api/forecast?date=${date}&model=${model}`),
  live: () => get<Forecast>('/api/forecast/live'),
  refreshLive: () => get<Forecast>('/api/forecast/live/refresh', { method: 'POST' }),
  reviewStatus: () => get<ReviewStatus>('/api/review/status'),
}

export interface Loadable<T> {
  data: T | null
  error: string | null
  loading: boolean
  retry: () => void
}

/** Fetch on mount and whenever `deps` change; exposes loading/error/retry for the UI states. */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[]): Loadable<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(fn, deps)
  useEffect(() => {
    let live = true
    setLoading(true)
    setError(null)
    run()
      .then((d) => live && setData(d))
      .catch((e: Error) => live && setError(e.message))
      .finally(() => live && setLoading(false))
    return () => {
      live = false
    }
  }, [run, tick])
  return { data, error, loading, retry: () => setTick((t) => t + 1) }
}

// ---- small date helpers (dates are ISO strings, handled in UTC to avoid TZ shifts)
export const addDays = (iso: string, n: number) => {
  const d = new Date(iso + 'T00:00:00Z')
  d.setUTCDate(d.getUTCDate() + n)
  return d.toISOString().slice(0, 10)
}

/** Today's date in India (the station's time zone), as YYYY-MM-DD. */
export const todayIST = () => new Date(Date.now() + 5.5 * 3600e3).toISOString().slice(0, 10)

export const fmtDate = (iso: string, opts: Intl.DateTimeFormatOptions) =>
  new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', { timeZone: 'UTC', ...opts })
