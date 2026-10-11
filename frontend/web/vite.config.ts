import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api is served by the FastAPI app in ../api (uv run uvicorn main:app --port 8000)
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
