import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5001',
      '/shap_explanations': 'http://localhost:5001',
      '/results': 'http://localhost:5001'
    }
  }
})
