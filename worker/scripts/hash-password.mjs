// Prints a hash for the ADMIN_PASSWORD_HASH secret. The password is read without echo and never stored.
//
//   npm run hash-password               Argon2id (recommended; needs a Workers plan with a normal CPU budget)
//   npm run hash-password -- --pbkdf2   PBKDF2-SHA256, for the Workers Free plan's 10 ms CPU limit
//   echo -n 'my password' | npm run hash-password --silent     (non-interactive)
import { pbkdf2Sync, randomBytes } from 'node:crypto';
import { argon2idAsync } from '@noble/hashes/argon2.js';

const b64 = (buf) => Buffer.from(buf).toString('base64').replace(/=+$/, '');

function readHidden(prompt) {
  return new Promise((resolve) => {
    if (!process.stdin.isTTY) {
      let data = '';
      process.stdin.on('data', (c) => (data += c)).on('end', () => resolve(data.replace(/\r?\n$/, '')));
      return;
    }
    process.stdout.write(prompt);
    let value = '';
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.setEncoding('utf8');
    const onData = (ch) => {
      for (const c of ch) {
        if (c === '\r' || c === '\n') {
          process.stdin.setRawMode(false);
          process.stdin.pause();
          process.stdin.off('data', onData);
          process.stdout.write('\n');
          return resolve(value);
        }
        if (c === '\u0003') process.exit(130);
        if (c === '\u007f' || c === '\b') value = value.slice(0, -1);
        else value += c;
      }
    };
    process.stdin.on('data', onData);
  });
}

const password = await readHidden('Admin password (12+ characters): ');
if (password.length < 12) {
  console.error('Use at least 12 characters.');
  process.exit(1);
}
if (process.stdin.isTTY && (await readHidden('Repeat: ')) !== password) {
  console.error('Passwords do not match.');
  process.exit(1);
}

const salt = randomBytes(16);
if (process.argv.includes('--pbkdf2')) {
  const hash = pbkdf2Sync(password, salt, 100_000, 32, 'sha256');
  console.log(`pbkdf2$sha256$100000$${b64(salt)}$${b64(hash)}`);
} else {
  const hash = await argon2idAsync(password, salt, { m: 19456, t: 2, p: 1, dkLen: 32 });
  console.log(`$argon2id$v=19$m=19456,t=2,p=1$${b64(salt)}$${b64(hash)}`);
}
