# Verification

Run these commands from the project root after `npm ci`. Install NumPy, Pillow and Playwright in your Python environment and run `python -m playwright install chromium`.

| Command | Coverage |
| --- | --- |
| `npm test` | Estimators, geometry, profiles, color scale, lecture sequencing, and performance checks |
| `npm run typecheck` | TypeScript application code and unit tests |
| `npm run build` | Type checking and generation of the offline HTML |
| `npm run verify:numpy` | Independent NumPy checks of estimator grids and lecture scenarios |
| `npm run e2e` | Offline Chromium, WebGL2 and 2D fallback, layouts, interaction, and all eight lecture scenes |
| `npm run verify:all` | Unit tests, build, numerical checks, and browser checks |

Build before running `npm run e2e` on its own. The numerical reports go to `.artifacts/numerical/`. The browser suite writes its report, screenshots, and video to `.artifacts/browser/`.

To repeat the browser checks against a published release, set `DEMO_TEST_URL` to its full Pages address, including the trailing slash, and run `npm run e2e`. The live checks use a fresh Chromium session and store their output in `.artifacts/browser-public/`.

To inspect the lecture captions, run `npx tsx tools/print_captions.ts`. After a browser run, `python -I tools/palette_sim.py` uses NumPy and Pillow to generate optional palette simulations in `.artifacts/palette/`.
