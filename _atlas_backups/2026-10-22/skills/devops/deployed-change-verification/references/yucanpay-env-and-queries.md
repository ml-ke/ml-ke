# YucanPay / hodaripay — environment map, schema gotchas, account queries

Project-specific depth for `~/ProG/hodaripay` (Flutter → Dart Frog API → Neon →
ChoiceBank, deployed on Railway + Cloudflare Workers).

## Environment map

| Layer | Production | Testing (staging) |
|---|---|---|
| API | `https://api-production-f1b85.up.railway.app` (`/health` → `ok`) | `https://api-staging-testing.up.railway.app` |
| Web | `https://yucan.co.ke` (bundle `/main.dart.js`) | `https://hodaripay-web-staging.hodaripay.workers.dev` |
| Neon branch | `production` — pooler host `ep-hidden-wildflower-…` | `testing` — `ep-solitary-boat-…` |
| Railway | project `bf850671-ac0a-4a93-80c0-67d0992c8f6e`, env `production`, service `api` | env `testing`, service `api-staging` |
| Android | R2 `https://downloads.yucan.co.ke/android/yucanpay.apk` | — |

Workflow names for `gh run list --workflow=`: `deploy-staging.yml`,
`promote.yml`, `release-apk-production.yml`. `promote.yml` skips its migrate and
API-deploy jobs when the commit touches no `backend/**` / `packages/shared/**` /
infra path, so an app-only release leaves the API on the previous backend
commit — by design (ADR-0022), not a failure.

## Answering "is the fix in prod" for this repo

```bash
cd ~/ProG/hodaripay
git rev-parse origin/main
gh run list --workflow=deploy-staging.yml --limit 3 --json headSha,conclusion,createdAt
gh run list --workflow=promote.yml       --limit 3 --json headSha,conclusion,createdAt
curl -s https://yucan.co.ke/main.dart.js | grep -c "<string the fix added>"
curl -sI https://downloads.yucan.co.ke/android/yucanpay.apk | grep -iE "HTTP/|last-modified|content-length"
railway deployment list --project bf850671-ac0a-4a93-80c0-67d0992c8f6e --environment production --service api
```

## Table/column gotchas

- `users`: the active flag is **`status`** (value `active`) — there is **no
  `is_active`**. Merchants are identified by `phone` (often no `email`). Staff
  and admin rows live in the same table (roles `admin`, `support`,
  `content_editor`, `finance`). PIN state: `pin_updated_at`,
  `pin_failed_attempts`, `pin_locked_until`; `last_login_at` shows live use.
- `audit_logs(id, user_id, action, entity_type, entity_id, before, after, ip,
  created_at)` — `user_id` is the **actor**, `entity_id` the target, and
  `entity_id` is **varchar**: `join users u on u.id = a.entity_id::uuid`.
- `notifications(id, user_id, kind, title, body, data, dedupe_key, read_at,
  created_at)` — the dedupe unique index is **partial**
  (`WHERE dedupe_key IS NOT NULL`), so any `ON CONFLICT` must repeat that
  predicate or Postgres raises 42P10.
- `merchant_lifecycle_events(id, merchant_profile_id, actor_user_id,
  from_status, to_status, reason, action, created_at)` — keyed by **profile**,
  not user.
- `wallets(user_id, merchant_profile_id, individual_account_id, account_id,
  account_type, currency, status, balance, ledger_base_balance, balance_source,
  short_code, last_synced_at)`.
- `individual_accounts(..., status, account_type, onboarding_request_id,
  choice_user_id, account_id, kyc_status, rejection_reasons, otp_confirmed,
  smile_status, smile_job_id, upgrade_request_id, upgrade_status,
  upgrade_rejection_reasons, profile_check*)`.
- `transactions(user_id, merchant_profile_id, wallet_id, tx_id, direction,
  type, channel, amount, fee_amount, status, status_label, counterparty_*,
  otp_required, note, category, created_at)`; `webhook_events(... processed,
  handler_error, sla_status, retry_count ...)`.
- **No migrations-tracking table.** To confirm a migration landed, probe for
  what it created — e.g. `select indexname from pg_indexes where tablename =
  'notifications'`, or a column in `information_schema.columns`.

## Reconstructing what happened to a user's account

```sql
-- 1. find them
select id, full_name, phone, role, status, pin_updated_at, last_login_at
from users where full_name ilike '%<name>%';

-- 2. who touched the account, and when
select a.created_at, a.action, p.full_name as actor, p.role
from audit_logs a left join users p on p.id = a.user_id
where a.entity_id = '<user-id>' order by a.created_at desc;

-- 3. personal account + wallet state
select status, account_type, kyc_status, smile_status, upgrade_status,
       upgrade_request_id, account_id, updated_at
from individual_accounts where user_id = '<user-id>';
select account_type, status, balance, balance_source, short_code, last_synced_at
from wallets where user_id = '<user-id>';

-- 4. money + comms (a notification row right after a collection proves the
--    notification path works in the deployed build)
select created_at, direction, type, channel, amount, status, status_label
from transactions where user_id = '<user-id>' order by created_at desc limit 10;
select created_at, kind, title, left(coalesce(body,''),120), read_at
from notifications where user_id = '<user-id>' order by created_at desc limit 15;
```

Balance sanity: `sum(in) − sum(out)` over the user's transactions must equal
`wallets.balance`; `balance_source = 'choice'` with a recent `last_synced_at`
means the ChoiceBank reconciliation ran (`getAccountDetails`, not the endpoints
that 500 in sandbox).

Support resets are visible as `audit_logs.action` (`reset_user_pin` etc.) with
the staff user as actor — that is how you prove "we did work on account X" when
nothing in the repo or chat history mentions it.
