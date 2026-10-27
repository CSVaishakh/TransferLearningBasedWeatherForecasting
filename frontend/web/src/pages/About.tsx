import { ArrowRight } from 'lucide-react'

const STEPS = [
  { t: 'Register', d: 'Handwritten daily readings at SJCET (rain, dry/wet bulb, min/max), 2010–2016 scanned' },
  { t: 'Digitize', d: 'Every page transcribed, then checked against monthly rain totals and physical rules' },
  { t: 'Pretrain', d: 'Graph neural network learns regional weather from 36 years of reanalysis and NOAA stations' },
  { t: 'Fine-tune', d: 'The pretrained network is adapted to the SJCET register (2010–2013)' },
  { t: 'Forecast', d: 'Max/min temperature and rain at SJCET up to 6 days ahead: scored on 2015–2016, and live from the latest data' },
]

const TEAM = ['C S Vaishakh (SJC23AD025)', 'Elisha Anto (SJC23AD032)', 'Jeffin James (SJC23AD037)', 'Rahul Sarang (SJC23AD061)']

export default function About() {
  return (
    <>
      <header className="page-head">
        <div>
          <h1>About the project</h1>
          <p>Transfer-learning based weather forecasting from historical paper registers</p>
        </div>
      </header>

      <section className="card">
        <div className="card-head"><h2>The idea</h2></div>
        <p className="muted" style={{ marginTop: 0 }}>
          SJCET has kept a paper weather register for years, but a few years of one station is far too little data to train a
          forecasting model on its own. This project digitizes the register, pretrains a graph neural network on plentiful
          regional data, and then fine-tunes it on the college's own readings. The Results page measures how much that
          pretraining helps.
        </p>
        <div className="pipeline" aria-label="Pipeline">
          {STEPS.map((s, i) => (
            <div key={s.t} style={{ display: 'contents' }}>
              {i > 0 && <ArrowRight className="arrow" size={20} aria-hidden />}
              <div className="step">
                <strong>{i + 1}. {s.t}</strong>
                <span>{s.d}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="grid grid-2 mt">
        <div className="card">
          <div className="card-head"><h2>Model</h2></div>
          <ul className="muted" style={{ margin: 0, paddingLeft: 18 }}>
            <li>1,062 reanalysis grid nodes (1° over India, 0.5° over Kerala), 6 NOAA stations and the SJCET register, linked to their 8 nearest neighbours</li>
            <li>Each node sees its last 7 days of 9 variables; 4 rounds of attention-based message passing; one output block per lead day (1–6 days ahead)</li>
            <li>Stage 1: reanalysis only · Stage 2: + NOAA stations · Stage 3: SJCET</li>
            <li>Trained 1990–2013, validated on 2014, tested on 2015–2016</li>
          </ul>
        </div>
        <div className="card">
          <div className="card-head"><h2>Data sources</h2></div>
          <ul className="muted" style={{ margin: 0, paddingLeft: 18 }}>
            <li><strong>SJCET weather register</strong>, St. Joseph's College of Engineering and Technology, Palai</li>
            <li><strong>NASA POWER</strong> daily reanalysis (MERRA-2 based), NASA Langley Research Center</li>
            <li><strong>NOAA GSOD</strong> Global Surface Summary of the Day, NOAA National Centers for Environmental Information</li>
            <li>Map tiles © OpenStreetMap contributors</li>
          </ul>
        </div>
      </section>

      <section className="card mt">
        <div className="card-head"><h2>Team</h2></div>
        <ul className="muted" style={{ margin: 0, paddingLeft: 18 }}>
          {TEAM.map((m) => <li key={m}>{m}</li>)}
        </ul>
        <p className="muted" style={{ marginBottom: 0 }}>
          Guided by <strong>Mrs. Neethu Tom</strong>, Assistant Professor, Department of Artificial Intelligence and Data
          Science, St. Joseph's College of Engineering and Technology, Palai.
        </p>
      </section>
    </>
  )
}
