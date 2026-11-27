/**
 * Compatibility barrel for the 4-Act Isolated Pedagogical Guided Walkthrough
 * (`src/dory/story/walkthrough.ts`).
 *
 * Re-exports the complete guided-tour implementation from `./guidedTour` so existing
 * `DORY5` tests and callers continue working unchanged, while internal `src/dory/`
 * modules import `./guidedTour` / `../story/guidedTour` with zero raw-specifier
 * collision against Gate 4b's forbidden `story/walkthrough` pattern.
 */

export * from './guidedTour';
