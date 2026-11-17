import { existsSync } from 'node:fs';
import type { Server } from 'node:http';
import { join } from 'node:path';
import { getRequestListener } from '@hono/node-server';
import express, { type NextFunction, type Request, type Response } from 'express';
import { app as api, CSP, RETRY_HEADER } from '../app';
import type { Sql } from '../db';
import type { Env } from '../env';

export interface BuiltServer {
  /** The Express application (mount it in tests with `http.createServer(app)`). */
  app: express.Express;
  /** Background work still running (generations finish and are logged even after the browser leaves). */
  pending: () => number;
  /** Waits for in-flight background work, then closes the database pool. */
  drain: (timeoutMs?: number) => Promise<void>;
}

export interface ServerOptions {
  /** Folder holding the built frontend (frontend/dist). */
  staticDir: string;
}

const MAX_BODY_BYTES = 2_000_000;
const BODY_METHODS = new Set(['POST', 'PUT', 'PATCH']);

function securityHeaders(req: Request, res: Response) {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Content-Security-Policy', CSP);
  if (String(req.headers['x-forwarded-proto'] ?? '').split(',')[0]?.trim() === 'https') {
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  }
}

/**
 * One Express server for the whole product: the built React app and the /api routes. The API itself is the
 * Hono application in ../app.ts (the same code that runs on a Cloudflare Worker); Express hosts it, serves the
 * static frontend, and answers the platform health check without touching the database.
 */
export function buildServer(env: Env, opts: ServerOptions): BuiltServer {
  const background = new Set<Promise<unknown>>();
  const ctx = {
    waitUntil(p: Promise<unknown>) {
      const tracked: Promise<unknown> = Promise.resolve(p).catch(() => undefined).finally(() => background.delete(tracked));
      background.add(tracked);
    },
    passThroughOnException() {},
  };
  // Hono's request/response classes must not replace the globals: provider calls use the platform's own fetch.
  const handleApi = getRequestListener(async (request) => {
    // Buffer the (size-limited) body once so a request can be replayed if its pooled connection turns out to be dead.
    const body = request.body ? await request.arrayBuffer() : undefined;
    const attempt = () => api.fetch(new Request(request.url, { method: request.method, headers: request.headers, body }), env, ctx as never);
    let res = await attempt();
    if (res.headers.has(RETRY_HEADER)) {
      await new Promise((r) => setTimeout(r, 150));
      res = await attempt();
    }
    if (res.headers.has(RETRY_HEADER)) {
      res = new Response(res.body, res); // headers of a Response may be immutable; copy before editing
      res.headers.delete(RETRY_HEADER);
    }
    return res;
  }, { overrideGlobalObjects: false });

  const app = express();
  app.disable('x-powered-by');
  app.disable('etag');
  app.set('trust proxy', true);

  // Liveness for the platform: always cheap, never depends on the database. (/api/health is the deep check.)
  app.get('/healthz', (_req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.json({ status: 'ok' });
  });

  // Refuse oversized or unsized request bodies before any of it is read into memory.
  app.use('/api', (req, res, next) => {
    if (!BODY_METHODS.has(req.method)) return next();
    const declared = req.headers['content-length'];
    if (declared === undefined) return void res.status(411).json({ detail: 'Content-Length is required.' });
    if (Number(declared) > MAX_BODY_BYTES) return void res.status(413).json({ detail: 'Request body too large.' });
    next();
  });
  app.all(/^\/api(?:\/|$)/, (req, res, next) => {
    // Express strips nothing here (no mount path), so Hono sees the original URL.
    handleApi(req, res).catch(next);
  });

  const indexFile = join(opts.staticDir, 'index.html');
  if (!existsSync(indexFile)) {
    console.warn(`Frontend not built: ${indexFile} is missing (run "npm run build"). The API still works.`);
  }
  app.use(express.static(opts.staticDir, {
    index: false,
    redirect: false,
    setHeaders(res, path) {
      res.setHeader('X-Content-Type-Options', 'nosniff');
      res.setHeader('X-Frame-Options', 'DENY');
      res.setHeader('Referrer-Policy', 'no-referrer');
      res.setHeader('Content-Security-Policy', CSP);
      // Vite fingerprints everything under /assets, so it can be cached for good.
      res.setHeader('Cache-Control', /[\\/]assets[\\/]/.test(path) ? 'public, max-age=31536000, immutable' : 'no-cache');
    },
  }));
  // Single-page app: any other page-like URL (/, /admin, ...) is the shell. Missing files stay 404.
  app.use((req, res, next) => {
    if ((req.method !== 'GET' && req.method !== 'HEAD') || /\.[A-Za-z0-9]{1,8}$/.test(req.path)) return next();
    if (!existsSync(indexFile)) return void res.status(503).type('text/plain').send('The playground frontend has not been built.');
    securityHeaders(req, res);
    res.setHeader('Cache-Control', 'no-cache');
    res.sendFile(indexFile);
  });
  app.use((req, res) => {
    securityHeaders(req, res);
    res.status(404).type('text/plain').send('Not found');
  });
  app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    console.error('request failed:', err instanceof Error ? err.name : 'unknown');
    if (!res.headersSent) res.status(500).json({ detail: 'Internal error.' });
    else res.end();
  });

  return {
    app,
    pending: () => background.size,
    async drain(timeoutMs = 25_000) {
      let timer: NodeJS.Timeout | undefined;
      const timeout = new Promise<void>((resolve) => { timer = setTimeout(resolve, timeoutMs); });
      await Promise.race([Promise.all([...background]).then(() => undefined), timeout]);
      clearTimeout(timer);
      await (env.SQL_POOL as Sql | undefined)?.end({ timeout: 5 }).catch(() => undefined);
    },
  };
}

/** Tune the HTTP server for a platform proxy: its idle connections outlive Node's 5 s default. */
export function tuneForProxy(server: Server) {
  server.keepAliveTimeout = 120_000;
  server.headersTimeout = 125_000;
  server.requestTimeout = 300_000;
}
