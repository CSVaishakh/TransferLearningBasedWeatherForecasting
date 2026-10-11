import { RefreshCw } from 'lucide-react'
import type { Loadable } from '../api'

export function Skeleton({ height = 120 }: { height?: number }) {
  return <div className="skeleton" style={{ height }} aria-busy="true" aria-label="Loading" />
}

export function ErrorState({ error, retry }: { error: string; retry: () => void }) {
  const offline = /fetch|NetworkError|502|504/i.test(error)
  return (
    <div className="state error" role="alert">
      <div>{offline ? 'Could not reach the API. Is it running on port 8000?' : 'Something went wrong loading this.'}</div>
      <button className="btn" style={{ marginTop: 12 }} onClick={retry}>
        <RefreshCw size={14} /> Retry
      </button>
      <details>
        <summary>Details</summary>
        {error}
      </details>
    </div>
  )
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="state">{children}</div>
}

/** Render loading / error states for a Loadable, or the children once data is in. */
export function Load<T>({ q, height, children }: { q: Loadable<T>; height?: number; children: (d: T) => React.ReactNode }) {
  if (q.error) return <ErrorState error={q.error} retry={q.retry} />
  if (q.loading || q.data === null) return <Skeleton height={height} />
  return <>{children(q.data)}</>
}
