// Checks a deployed playground end to end without spending any model tokens.
//
//   npm run smoke -- https://ai-playground.onrender.com
//   ADMIN_PASSWORD='...' npm run smoke -- https://ai-playground.onrender.com     (also tests admin login)
//
// It starts one tester session named "Deployment check" (it shows up in the admin's tester list).
const base = (process.argv[2] ?? '').replace(/\/+$/, '');
if (!/^https?:\/\//.test(base)) {
  console.error('Usage: npm run smoke -- https://<your-service>.onrender.com');
  process.exit(2);
}

const jar = new Map();
let failed = 0;
const ok = (cond, label, extra = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}${extra ? '  ' + extra : ''}`);
  if (!cond) failed++;
};

async function call(path, init = {}) {
  const headers = { ...init.headers };
  if (jar.size) headers.Cookie = [...jar].map(([k, v]) => `${k}=${v}`).join('; ');
  if (init.body) headers['Content-Type'] = 'application/json';
  const t0 = Date.now();
  const res = await fetch(base + path, { ...init, headers, redirect: 'manual' });
  for (const sc of res.headers.getSetCookie()) {
    const [name, value = ''] = sc.split(';')[0].split('=');
    if (value === '' || /max-age=0/i.test(sc)) jar.delete(name); else jar.set(name, value);
  }
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch { /* not JSON */ }
  return { status: res.status, text, json, headers: res.headers, ms: Date.now() - t0 };
}

// A free Render service may be asleep: the first request can take up to a minute.
console.log(`Checking ${base} (a sleeping free service can take up to a minute to answer)...`);
let live;
for (let i = 0; i < 12; i++) {
  try {
    live = await call('/healthz');
    if (live.status === 200) break;
  } catch { /* still waking */ }
  await new Promise((r) => setTimeout(r, 5000));
}
ok(live?.status === 200, 'service is up (/healthz)', live ? `${live.ms} ms` : '');
if (live?.status !== 200) process.exit(1);

const health = await call('/api/health');
ok(health.status === 200 && health.json?.database === 'ok', 'database reachable (/api/health)', JSON.stringify(health.json));

const shell = await call('/');
ok(shell.status === 200 && shell.text.includes('id="root"'), 'React app served at /');
ok(Boolean(shell.headers.get('content-security-policy')), 'security headers present');
const admin = await call('/admin');
ok(admin.status === 200 && admin.text.includes('id="root"'), 'React app served at /admin');

const before = await call('/api/playground/models');
ok(before.status === 401, 'tester API refuses requests without a session');

const start = await call('/api/session', { method: 'POST', body: JSON.stringify({ name: 'Deployment check' }) });
ok(start.status === 201 && Boolean(start.json?.session_id), 'tester session starts', start.json?.session_id ?? start.text.slice(0, 120));
const cookie = start.headers.getSetCookie().join(';');
ok(/HttpOnly/i.test(cookie), 'session cookie is HttpOnly');
if (base.startsWith('https://')) ok(/Secure/i.test(cookie), 'session cookie is Secure');

const models = await call('/api/playground/models');
ok(models.status === 200, 'model catalogue loads');
if (models.json) {
  const chatbot = models.json.modes?.find((m) => m.id === 'kundli_chatbot');
  ok(Boolean(chatbot?.available), 'Kundli chatbot mode available (AI_CORE_URL is set)');
  for (const p of models.json.providers ?? []) console.log(`INFO  ${p.name}: ${p.configured ? 'configured' : 'no API key'}`);
}

if (process.env.ADMIN_PASSWORD) {
  const login = await call('/api/admin/login', { method: 'POST', body: JSON.stringify({ password: process.env.ADMIN_PASSWORD }) });
  ok(login.status === 200 && Boolean(login.json?.csrf_token), 'admin login works', `${login.ms} ms`);
  if (login.status === 200) {
    const analytics = await call('/api/admin/analytics');
    ok(analytics.status === 200, 'admin analytics load');
    await call('/api/admin/logout', { method: 'POST', headers: { 'X-CSRF-Token': login.json.csrf_token } });
    ok((await call('/api/admin/me')).status === 401, 'admin logout ends the session');
  }
} else {
  console.log('INFO  admin login not tested (set ADMIN_PASSWORD to include it)');
}

console.log(failed ? `\n${failed} check(s) failed.` : '\nAll checks passed.');
process.exit(failed ? 1 : 0);
