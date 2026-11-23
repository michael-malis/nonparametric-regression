// Print every experiment caption at every stop (the lecturer's text, from live state).
//   npx tsx tools/print_captions.ts
(globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame = () => 0;
const { Store } = await import('../src/state/store');
const { Runner } = await import('../src/experiments/runner');
const runs: [string, string][] = [['segment', '1nn'], ['segment', 'knn'], ['segment', 'nw'], ['response', '1nn'], ['response', 'knn'], ['response', 'nw'], ['hsweep', 'nw'], ['ksteps', 'knn']];
for (const [id, m] of runs) {
  const store = new Store();
  store.set({ method: m as never });
  store.flush();
  const r = new Runner(store, () => true);
  r.start(id as never);
  store.flush();
  console.log(`\n== ${id} / ${m}`);
  for (;;) {
    const i = r.info(store.get())!;
    console.log(`[${i.step}] ${i.caption}`);
    if (!r.next()) break;
    store.flush();
  }
}
