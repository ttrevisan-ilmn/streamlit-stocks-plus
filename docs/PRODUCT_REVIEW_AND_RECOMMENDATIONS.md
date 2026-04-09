# Product Review and Recommendations

This review focuses on practical improvements across **UI/UX**, **engineering quality/tech debt**, and **feature enhancements** for the Streamlit Stocks Plus app.

## 1) What is working well

- Clear modular value proposition (Market Health, Sector Rotation, Stock Analysis, Congress, Options, Screener, etc.) in one terminal-style interface.
- Existing caching strategy (`@st.cache_data`) already reduces repeated API pressure in key modules.
- Core regression tests exist for high-risk integration points (options flow key fields, Congress API parsing, SEAF shape), which is a strong foundation.

## 2) UI/UX recommendations

### A. Simplify first-run experience

**Current friction**
- Very dense page with 10 tabs and long content in several tabs can feel overwhelming for first-time users.

**Recommendation**
- Add a **“Quick Start” mode** in sidebar with 3 presets:
  1. Macro view (Market Health + Intermarket)
  2. Single-ticker deep dive (Stock Analysis + Power Gauge + Stage + CANSLIM)
  3. Options trader view (Options Flow + Screener)
- Persist selected preset in session state so users return to their preferred workflow.

### B. Improve information hierarchy in tab content

**Current friction**
- Some sections mix headline signals and detailed diagnostics without clear visual hierarchy.

**Recommendation**
- For each tab, enforce a pattern: **Top signal card → 2-3 supporting KPIs → expandable diagnostics**.
- Move secondary explanatory text into tooltips or expanders to reduce scroll fatigue.

### C. Reduce custom HTML/CSS fragmentation

**Current friction**
- Many inline `st.markdown(..., unsafe_allow_html=True)` blocks create inconsistent spacing, borders, and color semantics.

**Recommendation**
- Create a shared UI helper module (e.g., `ui_components.py`) for reusable cards:
  - signal badge
  - KPI tile
  - warning/info banners
- Keep one centralized theme palette (success/warn/error/neutral) to improve visual consistency.

### D. Improve watchlist usability

**Current friction**
- Watchlist supports add/remove but lacks sorting, notes, and quick actions.

**Recommendation**
- Add small controls:
  - pin favorite tickers
  - drag/sort or priority ranking
  - one-click “analyze all watchlist” queue with progress

### E. Improve loading experience

**Current friction**
- Several expensive operations run with spinners only; users can’t estimate duration.

**Recommendation**
- Standardize progress feedback:
  - estimated step counter (e.g., 1/3 Power Gauge, 2/3 Stage, 3/3 CANSLIM)
  - cached timestamp display (“data as of HH:MM”).

## 3) Tech debt and reliability recommendations

### A. Break up `streamlit_app.py` into feature entry modules (high priority)

**Problem**
- `streamlit_app.py` currently mixes layout, data calls, indicator calculations, state management, persistence, and rendering.

**Recommendation**
- Split into:
  - `app.py` (orchestrator + routing)
  - `views/` per tab
  - `services/` for external data access
  - `ui/` shared rendering helpers
- This lowers blast radius and makes testing + onboarding much easier.

### B. Replace broad `except:` with explicit exception handling (high priority)

**Problem**
- Multiple modules use bare exceptions, suppressing root causes and making production debugging harder.

**Recommendation**
- Replace broad catches with explicit exception types (`ValueError`, `KeyError`, `requests.RequestException`, etc.) and add structured logging.

### C. Decouple UI from data-fetch functions (medium-high priority)

**Problem**
- Some data-layer functions directly call Streamlit UI primitives (`st.progress`, `st.error`, etc.), reducing reusability and testability.

**Recommendation**
- Return status objects from service functions and let view layer render UI.
- Keep service layer framework-agnostic wherever possible.

### D. Strengthen dependency management and reproducibility (medium priority)

**Problem**
- `requirements.txt` uses unpinned dependencies, which can cause regressions over time.

**Recommendation**
- Pin versions (or at least major/minor ranges) and add a lock workflow.
- Add CI matrix for Python versions you intend to support.

### E. Add observability and guardrails for external APIs (medium priority)

**Problem**
- API reliability/rate limits can degrade UX unexpectedly.

**Recommendation**
- Introduce a lightweight “provider health” panel and retry/backoff wrappers.
- Track per-provider error rates, latency, and cache hit ratio.

## 4) Feature enhancement opportunities

### A. Portfolio-level workflow

- Add simple portfolio input (weights + cost basis), then aggregate:
  - exposure by sector/factor
  - aggregate signal score (weighted Power Gauge / risk state)
  - downside scenario quick stress test

### B. Alerting and automation

- Add optional alerts when:
  - A6 flips state (BUY/NEUTRAL/CASH)
  - ticker enters Stage 2/4
  - options sentiment crosses threshold
- Start with local/email/webhook support.

### C. Explainability layer

- For each strategy verdict, add “why this score changed” deltas vs prior run.
- Show top contributing factors and confidence/coverage quality.

### D. Data quality confidence indicator

- Add confidence score per panel based on:
  - stale data age
  - missing field percentage
  - provider fallback usage

### E. Screener enhancements

- Add saved filters and custom strategy builder (logical conditions).
- Add quick backtest snapshot (e.g., next 20-day forward return histogram for historical signals).

## 5) Suggested implementation roadmap

### Phase 1 (1-2 weeks)
- Modularize app shell + move reusable UI components.
- Replace bare `except:` in high-traffic modules.
- Pin dependency versions.

### Phase 2 (2-4 weeks)
- Service/view separation for data fetch + rendering.
- Add provider health telemetry and better error surfaces.
- Introduce quick-start workflow presets.

### Phase 3 (4+ weeks)
- Portfolio mode + alerts.
- Strategy explainability and screener custom builder.

## 6) Success metrics to track

- Time-to-first-insight (seconds from load to first actionable signal).
- API error rate by provider.
- Cache hit rate.
- Weekly active users and session duration.
- Retention of users who create watchlists.

## 7) Immediate low-effort wins

1. Replace 5-10 highest-risk bare exception blocks with explicit handling and logging.
2. Add dependency pinning + CI test run on PR.
3. Add a compact “Top Signals Today” summary strip above tabs.
4. Add data freshness timestamps to each tab header.
