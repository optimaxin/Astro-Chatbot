import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    // Same-origin in dev too (the Node server on :10000; for wrangler dev change this to :8787), so the admin cookie and CSP behave as in production.
    proxy: { '/api': 'http://localhost:10000' },
  },
  preview: { port: 5180 },
});
