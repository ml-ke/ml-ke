# UI-Only Bypass Pattern — Hidden Features via API

## The Pattern

Some applications implement client-side visibility controls (flags like `hiddenOnUI`, `visible`, `enabled`) that only affect what the browser/user interface displays. These flags are **not** enforced server-side — the API returns full data for all resources regardless of the flag value.

## Why This Works

- Frontend developers add visibility flags for UX purposes (e.g., hiding secret vaults from the main dashboard, hiding archived content from default views)
- They forget that the API doesn't filter by that flag — every endpoint returns everything
- The flag is honestly named: `hiddenOnUI` literally says "hidden on the UI" — nothing about API access

## How to Find It

1. **Look for boolean flags in API responses** — `hiddenOnUI`, `visible`, `archived`, `deleted`, `disabled`, `draft`
2. **Check if the API filters by that flag** — Does `GET /v1/resources` return items WHERE `hiddenOnUI = false`? Or does it return ALL items and let the frontend filter?
3. **Compare UI vs API** — Does the browser show 5 items but the API returns 10? That's a bypass.
4. **Check POST/PUT endpoints** — Can you create items with `hiddenOnUI: true` to hide them from other users?

## Real-World Examples

### Fireblocks Vault Enumeration
- API returns all vault accounts including those with `hiddenOnUI: true`
- Vault IDs are sequential integers (0, 1, 2, 3...)
- No rate limiting on GET enumeration (30/30 rapid requests succeeded)
- Hidden vaults are fully readable: names, balances, asset holdings
- Attack chain: Enumerate IDs → Discover hidden vaults → Read balances + metadata

### Other Potential Patterns
- Admin panels with `draft: true` articles visible via API
- Chat systems with `archived: true` conversations appearing in `/api/conversations`
- File storage with `deleted: false` filter only client-side

## Chain Impact Questions

| Finding | Without Chain | With Chain (hidden vaults) |
|---------|--------------|---------------------------|
| Sequential IDs | P4 — theoretical enumeration | P3 — hidden resources discovered |
| Hidden vault exposure | P4 — "they named it hiddenOnUI" | P3-P2 — full workspace mapping |
| Balance read | P5 — public by design | P4-P3 — ties vault identity to value |
| Cross-user access | (separate finding) | P1-P2 — data belonging to another user/tenant |
