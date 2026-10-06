---
name: hodaripay-flutter-web-ux
description: >-
  Platform parity + web-safe UI/UX playbook for the YucanPay Flutter app.
  Load before writing or editing ANY app UI or business-process flow: web and
  APK are both first-class, and web must work at phone (<600), tablet
  (600-1023) and desktop (>=1024) widths. Covers why compute() is synchronous
  on web and how to keep image/upload work non-blocking (AsyncImageEngine,
  header probes, the custom crop editor), the Paybill-only collection rules
  (no inbound Till), onboarding/OTP status UX, page-level button + inherited-
  widget footguns, and the widget-test seams that keep these flows testable.
  Triggers: onboarding, upload/crop/rotate, camera/gallery, KYC photos, wallet/
  receive/Paybill, animations/loading, responsive layout, widget tests for the
  app.
---
# YucanPay Flutter — Web + APK platform & UX playbook

Every UI/UX/business-process change must behave identically on **APK, mobile
web, tablet web and desktop web**. Verified gotchas that cost real hours; apply
them before touching `apps/hodaripay`.

## 1. Web is JS — `compute()` is synchronous, heavy work freezes the UI

- `compute()` on web (`_isolates_web.dart`) runs the callback **on the main
  thread**: `await null; return callback(message)`. There are no isolates.
- Anything `package:image` (decode/orient/resize/encode) or pure-Dart-heavy
  blocks the browser UI while it runs. Never rely on `compute` to keep web
  smooth.
- Do heavy image work through `AsyncImageEngine`
  (`apps/hodaripay/lib/core/utils/async_image_engine.dart`):
  `prepareImageUploadAsync`, `rotateImageBytesAsync`, `cropImageAsync` — real
  isolate on native, bounded synchronous + frame-yield on web. For web-only
  non-blocking decode/encode, prefer browser-native codecs; the fast path in
  `prepareImageUpload` (`probeImageDimensions` header scan, no decode) passes
  compliant JPEGs through untouched so most real photos never re-encode.
- The crop/rotate editor is purpose-built (`_CropSurface` in
  `camera_crop_picker.dart`) — it does NOT use `crop_your_image` (that
  package's internal `compute()` decode/crop froze web). Dimensions come from
  a header probe; the crop runs through `cropImageAsync`.
- **Animation rule**: any async step that may block (web) must paint a visible
  animation FIRST, then yield a frame, then do the work. The crop dialog
  defers mounting, plays "Preparing/Cropping…", then encodes.

## 2. Platform matrix — test every width, keep business logic identical

- Breakpoints: `isMobile` <600, `isTablet` 600–1023, `isDesktop` ≥1024
  (`core/utils/responsive.dart`; gutter 16 mobile / 32 desktop). Test on a
  NARROW phone web width (360–412px), not just desktop and APK.
- Business process must not fork by platform: onboarding OTP/review states,
  Paybill-only collection, history "Verify payment", per-wallet Paybill cards —
  the same data, labels and actions everywhere. Only chrome (nav rails vs bars,
  table vs cards) adapts to width.

## 3. Collection rules (ChoiceBank)

- **Inbound = M-Pesa Paybill + account number ONLY — there is no inbound
  Till.** STK push or Paybill+account; Till/Paybill are OUTBOUND B2B rails.
  Never offer a Till as a receiving option and never fabricate one.
- Per-wallet Paybill account number == the wallet's Choice `account_id`. The
  shortcode is the env `PAYBILL_SHORTCODE` (Railway: prod `444174`, staging
  `4101847`) — the app must never hardcode it; render only what
  `GET /merchant/receive/channels` returns (Home "Wallet balances", Wallet card,
  Receive tab).
- Crop-frame aspect: fixed for shaped documents (`DocumentAspect`: national/
  alien ID-1 1.586, passport ID-3 1.420, selfie 1:1); business documents
  (permits, certs, CR12, KRA certs…) are free-resizable (`aspect == null`).

## 4. Page-level footguns

- **Never read an inherited widget (`AppStateScope.of`, `Theme.of`) synchronously
  in `initState`** — it throws in debug and History was silently empty until the
  load was deferred. First load goes in
  `WidgetsBinding.instance.addPostFrameCallback`.
- **Page-level Material buttons** (`FilledButton`/`TextButton`/
  `OutlinedButton`) inherit the theme's `minimumSize: Size.fromHeight(48)`
  (an infinite `minWidth` from `Size.fromHeight`) and overflow in Rows /
  unbounded shells. At page level prefer `InkWell` actions or bound the size
  (`FilledButton.styleFrom(minimumSize: Size(0, 48))`).
- Every shell branch needs a visible nav destination; utility pages are
  sub-routes, not extra branches (selectedIndex out of range crashes).

## 5. Widget-test seams

- `AsyncImageEngine.isolatesEnabled = false` in `setUp` for tests that drive the
  crop flow (isolate responses never arrive in the fake-async zone).
- `HodariPayApi.realtimeEnabled = false` disables SSE + the 60s fallback +
  SessionActivityGuard timers (else "A Timer is still pending").
- `pumpAndSettle` hangs on repeating animations — pump fixed durations.
- Give `Image.memory` an `errorBuilder` so undecodable test bytes never surface.
- Unit tests that exercise the isolate path use plain `test()`, not
  `testWidgets`.

## 6. Where the rest lives

- Backend/Choice gotchas: `hodaripay-backend-debugging`, `hodaripay-choicebank`.
- Capture/onboarding image specs and Paybill details: `docs/choicebank.md`,
  `docs/choicebank-onboarding.md`, `docs/flutter-app-gotchas.md`.
