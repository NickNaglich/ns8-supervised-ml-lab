# WB-OPT.md

Phase plan to optimize the Streamlit UI for desktop, tablet, phablet, and mobile with responsive layouts, higher information density on large screens, and friendlier controls on small screens.

## WB-OPT0 — Audit & Targets
- Status: in progress; captures pending (no screenshots captured in this step).
- [ ] Capture current UI screenshots at common breakpoints (desktop ≥1280, tablet 1024, phablet ~900, mobile 480–768).
- [x] List key components per page and must-stay-visible items:
  - Results Browser: filters, primary metric column, task/model, artifact status; leaderboard at top.
  - Run Details: metrics JSON, config, CV summary/plot, figures (confusion, ROC/PR), run path/signature.
  - Compare: metric delta card, config diff, signatures, figures side-by-side.
  - Dataset Explorer: head table, target balance, histograms, pair plot, grid renderer.
  - Run Controls: mode status, config picker, dry-run preview, action buttons (tune/baseline/tests), log output.
  - Health: versions, paths, fingerprint inputs, session state, danger-zone clear.
- [x] Note current pain points to address in later phases:
  - Desktop: extra whitespace when filters are single-column; could use grid layout.
  - Tables: can be wide; need better column pruning on small screens and pagination on mobile.
  - Filters: stack vertically on small screens; need accordion/sticky behavior for mobile.
  - Figures: thumbnails can overflow on small screens; need size caps/lazy-load.
  - Run Controls: long buttons/log output can push content; need collapsible logs on small devices.

## WB-OPT1 — Responsive Layout System
- Status: in progress; initial layout helper/device toggle and padding/font adjustments wired.
- [x] Define breakpoints and layout rules (xs <640, sm 640–960, md 960–1280, lg >1280) in a small style helper.
- [x] Set global spacing/density tokens (tight, normal, roomy/compact) and apply based on breakpoint (padding/font via CSS).
- [x] Ensure tables use `use_container_width=True`; font sizing applied via layout profile.

## WB-OPT2 — Desktop (lg) Density Boost
- Status: in progress; desktop defaults added for expansion/thumbnails.
- [x] Use multi-column grids (e.g., 2–3 columns) for filters/leaderboards to reduce empty space (filters already split; thumbnails grid for desktop).
- [x] Show expanded tables by default with visible metrics/figures columns (desktop expands inline by default).
- [x] Keep figure thumbnails inline where space allows; enable side-by-side panels (grid thumbnails on desktop).

## WB-OPT3 — Tablet/Phablet (md)
- Status: in progress; tablet thumb grid added.
- [x] Switch to two-column where comfortable (filters left, content right) via layout helper; tablet uses columns for filters.
- [x] Collapse less-critical badges into tooltips; prioritize primary metric, task, model (artifacts column hover hint).
- [x] Ensure charts/figures fit without horizontal scroll; cap image widths for tablet.

## WB-OPT4 — Mobile (xs/sm)
- Status: in progress; filter accordion and card-mode figures toggle added.
- [x] Collapse filters into accordions; default to key filters only (Filters expander on mobile).
- [x] Stack content vertically; show compact cards instead of wide tables (primary metric, task, model, artifacts badge).
- [x] Hide/accordion secondary charts (thumbnails, pair plots) with “Show more” toggles (mobile card mode has optional figures toggle).

## WB-OPT5 — Controls & UX Polish
- Status: started; implementation pending.
- [x] Reduce control padding/margins on small screens; enlarge tap targets for mobile (density profile adjusts padding/font).
- [x] Persist filter states across pages; remember last dataset path and run selection (session_state for filters/dataset path).
- [x] Add quick “back to top” or “jump to filters” anchors for long pages.
- [x] Provide a density toggle (Compact/Comfort) for desktop/tablet users (sidebar density selector).

## WB-OPT6 — Performance & Media Handling
- Status: in progress; mobile pagination and figure gating added.
- [x] Lazy-load heavy figures/thumbnails; gate large images behind expanders on small screens (card-mode figures behind an expander).
- [x] Limit table/card rows on mobile with “Load more” / pagination; keep full on desktop.
- [x] Debounce expensive filters and cache dataset samples per path (dataset caching already present; filter apply button added).

## WB-OPT7 — Validation
- [ ] Re-screenshot after changes at all breakpoints; verify no horizontal scroll, readable fonts, and usable controls.
- [ ] Spot-check Lighthouse/CLS-like UX: avoid layout shifts when toggling filters.
- [ ] User walkthrough: complete key flows (browse, details, compare, dataset) on each device class.
