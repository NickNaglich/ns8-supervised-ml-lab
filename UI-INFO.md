# UI-INFO.md

Phased plan to weave the project narrative and usage hints into the Streamlit UI without adding heavy text.

## Phase INFO-0 — Extract key narrative anchors
- [x] Capture concise bullets from the conversation summary:
  - NS8 lab = engineered-feature ML experiments (entropy/symmetry/autocorr/FFT), not raw grids.
  - Three tasks with intent: A (view invariance limits), B (leakage honesty test via group splits), C (scale sanity check).
  - Synthetic data used to design failure modes; transparency > benchmarks.
  - UI purpose: browse artifacts, not chase scores; read-only by default.

## Phase INFO-1 — Surface intent in Overview
- [x] Add short “What you’re seeing” cards to Overview:
  - Project intent: honest, inspectable ML on NS8 (features, not pixels).
  - Task blurbs: A invariance, B leakage test, C scale sanity.
  - Usage hint: browse artifacts; run controls are gated.
- [x] Keep copy lean (1–2 sentences per card) with links to relevant pages.

## Phase INFO-2 — Contextual tooltips/labels
- [x] Results Browser: add help text for artifact badges (“hover to see missing”) and a note that Task B collapse under strict splits is expected.
- [x] Run Details: small caption reminding that features are engineered summaries (no raw grids).
- [x] Compare: note that differing dataset signatures imply different data.
- [x] Dataset Explorer: one-liner on why grids are reduced to features; optional link to README section.

## Phase INFO-3 — Page-level quick tips
- [x] Add a short tip bar per page (Results/Details/Compare/Dataset) with 1–2 bullets pulled from the narrative, hidden behind an expander on desktop, open on mobile/tablet.
- [x] Ensure tips are device-aware (concise on mobile).

## Phase INFO-4 — Read-more links (lightweight)
- [x] Add “Learn more” links to README sections (Tasks, Results, Method) from Overview and Docs tabs (overview card link to README/Docs).
- [x] Keep in-UI text minimal; defer depth to README/model cards.

## Phase INFO-5 — Final pass & consistency
- [ ] Verify copy is concise and aligned with the narrative (transparency over scores).
- [ ] Ensure no heavy text blocks; prefer captions, tooltips, and short expanders.
- [ ] Check device layouts still clean after tips/tooltips.
