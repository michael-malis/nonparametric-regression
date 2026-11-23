# Verification

Checks the regression engine, lecture scenarios, and offline browser behavior.

- `unit/` contains the Vitest suite.
- `numerical/` exports TypeScript outputs and compares them with NumPy.
- `browser/` uses Playwright to exercise the offline release in Chromium.
