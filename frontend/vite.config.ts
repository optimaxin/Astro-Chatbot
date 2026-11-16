import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    // Same-origin in dev too, so the admin cookie and CSP behave as in production.
    proxy: { '/api': 'http://localhost:8787' },
  },
  preview: { port: 5180 },
});
