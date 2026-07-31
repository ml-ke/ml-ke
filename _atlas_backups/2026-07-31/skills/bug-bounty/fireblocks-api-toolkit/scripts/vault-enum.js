#!/usr/bin/env node
/**
 * Enumerate Fireblocks vault accounts by sequential integer IDs.
 * Reads API key from /dev/shm/fb-apikey.txt and secret from fireblocks_secret.key
 */
const fs = require('fs');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const https = require('https');

const API_KEY=*** 'utf8').trim();
const SECRET=fs.rea...ey','utf8');
const BASE = 'https://sandbox-api.fireblocks.io';

function sign(path) {
  const url = BASE + path;
  const parsed = new URL(url);
  const uri = parsed.pathname + parsed.search;
  const now = Math.floor(Date.now()/1000);
  const bh = crypto.createHash('sha256').update('').digest().toString('hex');
  const p = {uri, nonce:crypto.randomUUID(), iat:now, exp:now+55, sub:API_KEY, bodyHash:bh};
  return jwt.sign(p, SECRET, {algorithm:'RS256'});
}

function apiGet(path) {
  return new Promise((resolve) => {
    const token = sign(path);
    const u = new URL(BASE + path);
    const opts = {
      hostname: u.hostname, path: u.pathname + u.search,
      headers: {'X-API-Key': API_KEY, 'Authorization':'Bearer '+token}
    };
    const r = https.request(opts, res => {
      let d=''; res.on('data',c=>d+=c); res.on('end',()=>resolve({s:res.statusCode, b:d}));
    });
    r.on('error', e => resolve({s:'ERR', b:e.message}));
    r.end();
  });
}

async function main() {
  console.log('Enumerating vault accounts...');
  for (let i = 0; i < 50; i++) {
    const r = await apiGet(`/v1/vault/accounts/${i}`);
    if (r.s === 200) {
      const a = JSON.parse(r.b);
      const assets = (a.assets || []).map(x => x.id).join(', ');
      console.log(`FOUND vault ${i}: "${a.name}" [${assets || 'empty'}]`);
    } else if (r.s === 400) {
      // Vault doesn't exist — stop sequential scan
      console.log(`No more vaults after ID ${i-1}`);
      break;
    }
  }
}
main();
