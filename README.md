# Nonparametric Regression Demo

An interactive lecture demo for comparing 1-nearest-neighbor, k-nearest-neighbor, and Nadaraya-Watson regression on two predictors.

[Open the interactive demo](https://michael-malis.github.io/nonparametric-regression/)

- Follow eight guided scenes in 2D and 3D, with pause, replay, and keyboard navigation.
- Drag the query cross or a site in the map, edit observed responses, and vary the neighbor count or kernel bandwidth in Explore.
- Inspect prediction weights, exact neighbor regions, path profiles, and a separate interactive Voronoi diagram.

## Run locally

After building, open `dist/index.html` for the offline demo. The build embeds its scripts, styles, and fonts in one HTML file. 3D needs WebGL2. On Windows, double-click `START_DEMO.cmd` to open the demo in full screen.

To build from source, use Node.js 22.12+ or a newer supported release:

```sh
npm ci
npm run build
```

Run `npm run dev` for the Vite development server.

## Verify

`npm test` runs the unit suite. `npm run verify:all` also builds the offline demo, checks its numerical outputs against NumPy, and runs the Playwright browser suite. Install the Python dependencies with `python -m pip install numpy pillow playwright`, then run `python -m playwright install chromium`. Generated screenshots, video, and test output are written to `.artifacts/`.

See [verification commands](VERIFICATION.md), the [project map](FILETREE.md), and the [lecture script in Russian](DEMO_SCRIPT.md).

## Publish on GitHub Pages

In **Settings → Pages → Build and deployment**, select **GitHub Actions**. A push to `main` runs the **Publish demo** workflow. To redeploy the current revision, choose **Actions → Publish demo → Run workflow**.

The workflow builds from the lockfile, tests the offline demo and uploads `dist/`. After deployment, it compares the public HTML with the tested build and repeats the browser suite at the Pages URL. Browser evidence is available in the workflow artifacts for 14 days. Vite uses a relative base, so the same build also works under other repository names.

## Third-party materials

The demo generates its data, diagrams and surfaces locally. It includes Three.js and KaTeX under the MIT license, and the KaTeX math fonts and Atkinson Hyperlegible under the SIL Open Font License. The production build includes the accompanying [license notices](public/licenses/). Distribute those notices with the HTML.

Developed by Michael Malis
