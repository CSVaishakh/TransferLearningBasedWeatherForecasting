import { useMemo } from 'react'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer } from 'react-leaflet'
import type { Graph, GraphNode } from '../api'

const SJCET: [number, number] = [9.727, 76.727]
// literal colours: the canvas renderer cannot resolve CSS variables
const C = { sjcet: '#e0453a', station: '#2f6fe4', grid: '#7d8a9c' }

const KIND_LABEL: Record<string, string> = {
  sjcet: 'SJCET register (target)',
  gsod: 'NOAA GSOD station',
  power: 'Reanalysis grid node (NASA POWER)',
  era5: 'Reanalysis grid node (ERA5)',
}

function style(n: GraphNode) {
  if (n.kind === 'sjcet') return { radius: 11, color: '#fff', weight: 3, fillColor: C.sjcet, fillOpacity: 1 }
  if (n.kind === 'gsod') return { radius: 8, color: '#fff', weight: 2, fillColor: C.station, fillOpacity: 1 }
  return { radius: n.region === 'fine' ? 4 : 3, color: C.grid, weight: 1, fillColor: '#ffffff', fillOpacity: 0.9 }
}

/** FR-F4: graph nodes around SJCET, coloured by kind; optional edges; click for details. */
export default function NodeMap({ graph, showEdges }: { graph: Graph; showEdges: boolean }) {
  const byId = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph])
  // stations last so they draw on top of the grid
  const nodes = useMemo(() => [...graph.nodes].sort((a, b) => Number(a.kind !== 'power') - Number(b.kind !== 'power')), [graph])
  const edges = useMemo(
    () =>
      showEdges
        ? graph.edges
            .filter(([s, t]) => s < t) // each undirected edge once
            .map(([s, t]) => [byId.get(s)!, byId.get(t)!])
            .filter(([a, b]) => Math.abs(a.lat - SJCET[0]) < 6 && Math.abs(a.lon - SJCET[1]) < 6 && b)
        : [],
    [graph, byId, showEdges],
  )
  return (
    <MapContainer center={SJCET} zoom={8} preferCanvas scrollWheelZoom={false} aria-label="Map of graph nodes around SJCET">
      <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {edges.map(([a, b], i) => (
        <Polyline
          key={i}
          positions={[[a.lat, a.lon], [b.lat, b.lon]]}
          pathOptions={{ color: a.kind === 'sjcet' || b.kind === 'sjcet' ? C.sjcet : C.grid, weight: a.kind === 'sjcet' || b.kind === 'sjcet' ? 2 : 1, opacity: 0.6, dashArray: '3 4' }}
        />
      ))}
      {nodes.map((n) => (
        <CircleMarker key={n.id} center={[n.lat, n.lon]} pathOptions={style(n)} radius={style(n).radius}>
          <Popup>
            <strong>{n.name ?? `Node ${n.id}`}</strong>
            <br />
            {KIND_LABEL[n.kind] ?? n.kind}
            {n.kind === 'power' && ` · ${n.region === 'fine' ? '0.5°' : n.region === 'sjcet' ? 'at SJCET' : '1°'} grid`}
            <br />
            Lat {n.lat.toFixed(3)}, Lon {n.lon.toFixed(3)}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  )
}
