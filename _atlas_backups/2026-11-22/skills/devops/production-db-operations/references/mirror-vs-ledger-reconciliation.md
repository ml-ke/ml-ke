# Mirror vs system of record

A local row that mirrors an external provider (banking-as-a-service ledger, payment gateway,
identity vendor) is a cache with business meaning. Two questions recur, and both are answered
by querying the provider rather than the mirror:

1. Does this person/account exist at the provider at all?
2. Does our row say the same thing the provider says?

## 1. Existence test

Most providers key a lookup on the same identifier the request gave you (mobile, email, account
number). Use the repo's own client so signing/auth is right; export credentials from the
platform's env store for the run, then `unset` them.

Read the provider's "not found" code as a **fact**, not a failure:

- `not found` for every format of the identifier (9-digit, `0`-prefixed, country-code
  prefixed) ⇒ no customer exists. Nothing was unlinked; do not repair anything.
- `00000` + an account/onboarding id for the neighbouring identifier ⇒ the lookup works and
  the customer is a *different* row. That pair of results is the whole report.
- Lookups are usually single-key: a profile/KYC query needs the provider's request id, and the
  business/merchant equivalent often rejects a mobile outright. If the key-shaped lookup is
  unsupported, say so rather than concluding "no account".

Then cross-check inside the mirror: every table that carries an owner FK, counted for the
target (see `postgres-evidence-recipes.md` §2). Zero linkage rows + provider `not found` is a
closed case.

## 2. Field-by-field diff

Pull the provider record and the mirror row, then compare in a table you print once — never
field by field in prose. Classify every column as one of:

| Class | Behaviour | Example |
|---|---|---|
| Identity | must match exactly | provider user id, account id, request id |
| Descriptive | mirrors text, tolerant of formatting | name parts, DOB, gender, ID type/number, address, KRA/tax id, email, mobile |
| Derived status | our enum vs provider code, mapped deliberately | `account_opened` ↔ `3`/`7`; `current` ↔ `C001` |
| Handler-only | written **only** by the event/webhook path | verification verdicts + their result codes |
| Live | fetch from provider each read | balance, short code |

Two rules that come out of the table:

- **Never store the provider's raw code in a column the app branches on.** Map to the app's own
  vocabulary at write time; a leaked `C00x`/numeric code breaks every branch that compares it.
- **Handler-only fields are the stale ones.** A poll that persists status/ids will happily
  refresh everything except the verdict codes, so a fully provisioned account can still carry
  `verification = 0, result_code = NULL` while the provider reports verified. Confirm with the
  provider's live value and grep the event table for that identifier before believing the
  mirror.

## 3. Backfill loop

```
read before  ->  single UPDATE on the derived columns keyed by PK  ->  insert audit row
             ->  read after  ->  re-run the diff  ->  repeat until empty
```

- Audit row: action name + `before`/`after` JSON + the provider source noted in `after`
  (which endpoint answered), so the next agent can tell an ops backfill from an app write.
- Only backfill values the provider actually returned. Formatting differences (leading zero,
  country code) are not mismatches to "fix" unless the read path normalises for you.
- If the mirror is missing whole rows rather than fields, stop: that is a linking decision
  (see below), not a backfill.

## 4. Linking and duplicates

To link an existing provider account to a user, mirror what the application writes on a
successful onboarding — the same tables, the same enum values, the same foreign-key pairing —
and let the app's own read/sync paths take over afterwards. Two guards:

- the provider-side **user id must be the real one**, never a locally generated stand-in, or
  later events cannot resolve back to the row;
- account/ledger ids are usually UNIQUE in the mirror, so check for an existing row and update
  it instead of inserting a second one.

When one human has two rows, the aliases decide which one is canonical. Order:

1. Show which row owns each unique alias (email/username) and which owns the live provider
   account.
2. Move the alias to the canonical row (free it from the duplicate first — it is UNIQUE).
3. Reassign dependent records that would otherwise be deleted by a cascade (support tickets,
   notifications, consent acceptances).
4. Only then delete the duplicate, listing what disappears with it.

Get sign-off for steps 2–4: a merged identity changes how someone logs in, so it is a product
decision as much as an ops one. If sign-off is not given, the minimum safe move is to free the
alias and leave the duplicate inert — and say that it remains reachable by its own phone.
