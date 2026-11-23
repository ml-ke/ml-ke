#!/usr/bin/env node
// Probe Fireblocks sandbox API — reads creds from /dev/shm/fb-apikey.txt and fireblocks_secret_I.key
const fs=require('fs');const j=require('jsonwebtoken');const c=require('crypto');const h=require('https');
const K=fs.readFileSync('/dev/shm/fb-apikey.txt','utf8').trim();
const S=fs.readFileSync('fireblocks_secret.key','utf8');
function tok(m,p,b){const n=Math.floor(Date.now()/1000);const bs=b?JSON.stringify(b):'';const bh=c.createHash('sha256').update(bs).digest().toString('hex');return j.sign({uri:p,nonce:c.randomUUID(),iat:n,exp:n+55,sub:K,bodyHash:bh},S,{algorithm:'RS256'});}
function api(m,p,b){return new Promise(r=>{const t=tok(m,p,b);const bs=b?JSON.stringify(b):'';const o={hostname:'sandbox-api.fireblocks.io',path:p,method:m,headers:{'X-API-Key':K,'Authorization':'Bearer '+t,'Content-Type':'application/json','Content-Length':Buffer.byteLength(bs)}};const x=h.request(o,s=>{let d='';s.on('data',c=>d+=c);s.on('end',()=>r({s:s.statusCode,b:d}));});if(b)x.write(bs);x.end();});}
async function run(){
 const e=[['GET','/v1/vault/accounts_paged'],['GET','/v1/transactions'],['GET','/v1/supported_assets'],['POST','/v1/vault/accounts',{name:'probe-'+Date.now()}],['POST','/v1/internal_wallets',{name:'probe',assetId:'ETH_TEST5',address:'0x0'}]];
 for(const x of e){const r=await api(x[0],x[1],x[2]);console.log(r.s,x[0],x[1],r.b.substring(0,120));}
}
run();
