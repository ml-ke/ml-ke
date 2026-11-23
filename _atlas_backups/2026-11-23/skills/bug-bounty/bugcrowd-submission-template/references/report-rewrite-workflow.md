# Report Rewrite Workflow

How to take a submitted report and improve it for resubmission.

## Step 1 — Find the Original Report

Read the submitted report. Identify weak areas:
- **VRT category**: Exact Bugcrowd path or free-form text?
- **Severity**: Matches VRT baseline or inflated?
- **Impact claims**: All verified against live server?
- **Version/scope claims**: Accurate? Check ALL versions.
- **PoC**: Does it prove what the description claims?
- **Business impact**: Framed in terms of who gets harmed?

## Step 2 — Verify Every Claim

Run each impact claim through a live test. If response contradicts claim → DELETE the claim.

**Surviving (bodyHash):** SDK crash, server 401, ALL 64 versions affected.
**Failed (bodyHash):** "JWT replay" — server rejects with nonce-reused error.

## Step 3 — Fix VRT Category

Before: Free-form text. After: Exact Bugcrowd path. Load `bugcrowd-vrt` skill.

## Step 4 — Fix Severity

Before: P2 (inflated). After: P3 (VRT baseline). Honest = baseline unless verified escalation.

## Step 5 — Fix Scope Claim

Before: "v19.1.0". After: "ALL 64 npm versions". Test multiple published versions.

## Step 6 — Cross-SDK Comparison

Show this SDK is the ONLY broken one. Table of 6 correct SDKs vs 1 broken.

## Step 7 — Business Impact Framing

Before: Technical ("writes fail"). After: Business ("10+ dependents, exchanges can't manage assets").

## Step 8 — Verify PoC Matches Description

Run PoC first. Capture exact output. Description matches what it actually shows.
