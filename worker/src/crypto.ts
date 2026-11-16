import { argon2idAsync } from '@noble/hashes/argon2.js';

const enc = new TextEncoder();

function toBase64Url(bytes: Uint8Array): string {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64(b64: string): Uint8Array<ArrayBuffer> {
  const std = b64.replace(/-/g, '+').replace(/_/g, '/');
  const bin = atob(std + '='.repeat((4 - (std.length % 4)) % 4));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

const hex = (buf: ArrayBuffer) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');

export function randomToken(bytes = 32): string {
  return toBase64Url(crypto.getRandomValues(new Uint8Array(bytes)));
}

export async function sha256Hex(text: string): Promise<string> {
  return hex(await crypto.subtle.digest('SHA-256', enc.encode(text)));
}

export async function hmacHex(secret: string, message: string): Promise<string> {
  const key = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return hex(await crypto.subtle.sign('HMAC', key, enc.encode(message)));
}

/** Constant-time string comparison (both sides are hashed to equal length first). */
export async function safeEqual(a: string, b: string): Promise<boolean> {
  const [ha, hb] = await Promise.all([sha256Hex(a), sha256Hex(b)]);
  let diff = 0;
  for (let i = 0; i < ha.length; i++) diff |= ha.charCodeAt(i) ^ hb.charCodeAt(i);
  return diff === 0;
}

export type PasswordCheck = 'ok' | 'mismatch' | 'invalid-hash';

/**
 * Verifies a password against ADMIN_PASSWORD_HASH. Accepts an Argon2id PHC string
 * (`$argon2id$v=19$m=19456,t=2,p=1$<salt>$<hash>`, e.g. from `npm run hash-password`
 * or argon2-cffi) or `pbkdf2$sha256$<iterations>$<salt>$<hash>` for plans whose CPU
 * budget is too small for Argon2.
 */
export async function verifyPassword(password: string, stored: string): Promise<PasswordCheck> {
  try {
    if (stored.startsWith('$argon2id$')) {
      const parts = stored.split('$'); // ['', 'argon2id', 'v=19', 'm=..,t=..,p=..', salt, hash]
      const params = Object.fromEntries((parts[3] ?? '').split(',').map((kv) => kv.split('=')));
      const m = Number(params.m), t = Number(params.t), p = Number(params.p);
      const salt = fromBase64(parts[4] ?? ''), want = fromBase64(parts[5] ?? '');
      if (![m, t, p].every(Number.isInteger) || m < 8 || m > 262144 || t < 1 || t > 20 || p < 1 || !want.length) {
        return 'invalid-hash';
      }
      const got = await argon2idAsync(password, salt, { m, t, p, dkLen: want.length });
      return (await safeEqual(toBase64Url(got), toBase64Url(want))) ? 'ok' : 'mismatch';
    }
    if (stored.startsWith('pbkdf2$')) {
      const [, algo, iter, saltB64, hashB64] = stored.split('$');
      const iterations = Number(iter);
      if (algo !== 'sha256' || !Number.isInteger(iterations) || iterations < 1000 || iterations > 100_000) {
        return 'invalid-hash';
      }
      const want = fromBase64(hashB64 ?? '');
      const key = await crypto.subtle.importKey('raw', enc.encode(password), 'PBKDF2', false, ['deriveBits']);
      const bits = await crypto.subtle.deriveBits(
        { name: 'PBKDF2', hash: 'SHA-256', salt: fromBase64(saltB64 ?? ''), iterations }, key, want.length * 8);
      return (await safeEqual(toBase64Url(new Uint8Array(bits)), toBase64Url(want))) ? 'ok' : 'mismatch';
    }
  } catch {
    return 'invalid-hash';
  }
  return 'invalid-hash';
}
