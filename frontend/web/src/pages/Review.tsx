import { ClipboardCheck } from 'lucide-react'
import { api, useApi } from '../api'
import { Load } from '../components/States'

export default function Review() {
  const q = useApi(api.reviewStatus, [])
  return (
    <>
      <header className="page-head">
        <div>
          <h1>Review Tool</h1>
          <p>Check and correct OCR'd register cells (team tool)</p>
        </div>
      </header>
      <section className="card">
        <Load q={q} height={160}>
          {(s) =>
            s.available ? (
              <div className="state">
                {s.queue} of {s.cells} cells need review. The keyboard review interface is not built yet.
              </div>
            ) : (
              <div className="state">
                <ClipboardCheck size={40} color="var(--text-3)" aria-hidden />
                <p><strong>Nothing to review yet.</strong></p>
                <p className="muted">{s.reason}</p>
                <p className="muted small">
                  The queue opens when the OCR pipeline writes <code>data/sjcet/cells.csv</code>, for example once the
                  2017+ phone photos are processed.
                </p>
              </div>
            )
          }
        </Load>
      </section>
    </>
  )
}
