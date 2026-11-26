# Architecture Guide — Nemotron 3 Nano Visualization

This guide documents the software architecture, WebGL2 rendering pipeline, mathematical cache ownership invariants, runtime performance architecture, and intentional v1.0 boundaries of **Nemotron 3 Nano 30B-A3B — Architecture Visualization**.

---

## 1. Module Dependency Architecture

The codebase is organized as a strict unidirectional dependency graph with **zero third-party browser runtime dependencies**:

```mermaid
flowchart LR
  Model["src/model/\n(spec, memory, state, rcPlayback)"]
  Math["src/math/\n(pure Float32 sims, hybridFlow,\ncacheContracts, incrementalDecode)"]
  Scene["src/scene/\n(spine, 5 cutaways, atlases,\naccumulation, sheetInspector)"]
  Render["src/render/\n(WebGL2 device, 5 passes,\nlinePacking, camera, pick)"]
  Story["src/story/\n(7 chapters / 22-beat 3-Act tour)"]
  UI["src/ui/\n(controls, overlay, memoryChart, CSS)"]
  Main["src/main.ts\n(composition root & RAF lifecycle)"]

  Model --> Math
  Model --> Scene
  Math --> Scene
  Model --> Render
  Scene --> Render
  Model --> Story
  Model --> UI
  Math --> UI
  Scene --> UI
  Story --> UI
  Model --> Main
  Math --> Main
  Scene --> Main
  Render --> Main
  Story --> Main
  UI --> Main
```

### Layer Responsibilities & Boundaries

| Directory | Responsibility | Strict Boundary |
| --- | --- | --- |
| `src/model/` | Pinned checkpoint metadata (`spec.ts`), logical 1→1M memory math (`memory.ts`), scene/UI state contracts (`state.ts`), and playback/session clocks (`rcPlayback.ts`, `mambaPlayback.ts`, `moePlayback.ts`). | Pure TypeScript; no WebGL or DOM access. |
| `src/math/` | Deterministic miniature Float32 simulators (`gqaSim.ts`, `mambaSim.ts`, `moeSim.ts`), unified 52-layer forward pass (`hybridFlow.ts`), AR decode cache contracts (`cacheContracts.ts`), incremental decoding & branching (`incrementalDecode.ts`), and formula formatting (`formulaFormat.ts`). | **Strictly pure CPU math:** zero DOM, WebGL, or `src/scene/` imports (depends only on `src/model/spec.ts`). Executable identically in Node (Vitest & fixture export) and the browser. |
| `src/scene/` | Pure geometry builders (`spine.ts`, `cutaways.ts`, `embeddingCutaway.ts`, `outputCutaway.ts`, `gqaCutaway.ts`, `gqaSpatial.ts`, `mambaCutaway.ts`, `mambaSpatial.ts`, `moeCutaway.ts`, `moeSpatial.ts`), RG32F value atlas packers (`atlasPacker.ts`, `*Atlas.ts`), focused-cell accumulation (`accumulation.ts`), and sheet inspection (`sheetInspector.ts`). | Emits packed `Float32Array` instance buffers, atlas texels, and CPU ray-pick hitboxes without calling WebGL APIs directly. |
| `src/render/` | Native WebGL2 engine (`device.ts`, `camera.ts`, `pick.ts`, `solids.ts`, `lines.ts`, `linePacking.ts`, `sheets.ts`, `sheetFrame.ts`, `ribbons.ts`). | Owns GPU shaders, VAOs/VBOs, FBOs, and projection/picking math; unaware of domain copy or walkthrough beats. |
| `src/ui/` | Accessible HTML/SVG UI chrome (`controls.ts`, `overlay.ts`, `memoryChart.ts`, `theme.css`) implementing WAI-ARIA tablists, modal focus traps, screen-space elbow callouts, floating cell popovers, and the 1→1M SVG memory scaling chart. | Pure view/controller components driven by state snapshots and callbacks from `src/main.ts`. |
| `src/story/` | Canonical 7-chapter / 22-beat, 3-Act guided walkthrough (`walkthrough.ts`) sharing the 8-token prefill prefix (`TOUR_PREFILL_TOKEN_IDS = [0..7]`) across Act I, Act II, and Act III (`8P + 1S = 9`). | Pure declarative beat metadata & dynamic Act III narrative helpers. |
| `src/main.ts` | Single browser composition root: initializes `RenderDevice`, all 5 GPU passes, `HybridModelParameters`, `DualDecodeSessionStore`, ResizeObserver, pointer/keyboard handlers, WebGL context-loss handlers, and the frame loop. | Sole owner of `requestAnimationFrame` scheduling and live DOM wiring. |

---

## 2. WebGL2 5-Pass Rendering Pipeline & Runtime Performance

### `RenderDevice` (`src/render/device.ts`)

- Creates a native WebGL2 context (`alpha: false`, `depth: true`) backed by an offscreen `RGBA8` + `DEPTH_COMPONENT24` framebuffer at up to **4× MSAA** (probing `getInternalformatParameter` for the highest mutually supported sample count `<= 4`, falling back cleanly to single-sample when MSAA is unavailable).
- Resolves multisampled color via a single-sample resolve FBO and blits cleanly to the canvas default backbuffer.
- Tracks an `fboAllocFailed` non-retry latch and explicitly destroys partially allocated FBO/renderbuffer handles on null, incomplete, or exception paths (reset on resize, `destroyFramebuffer()`, or `markLost()`).

### Five GPU Render Passes

All 5 passes are created at boot and recreated atomically on `webglcontextrestored`:

1. **`SolidsPass` (`src/render/solids.ts`)** — Instanced unit cuboids (`48 B`/instance) rendering the 52-layer Tower, structural housings, and bookend plates with directional shading, procedural face-local hatching, and BF16-gold / FP8-frosted styling (`POLYGON_OFFSET_FILL` enabled to prevent hairline z-fighting).
2. **`LinesPass` (`src/render/lines.ts`) + `linePacking.ts` (`src/render/linePacking.ts`)** — Depth-tested, screen-space antialiased hairlines packed in the authoritative **18-float (`72 B`)** per-instance layout (`LINE_INSTANCE_FLOATS = 18`, `LINE_INSTANCE_ATTRIBS`):
   - `iPointA` (`[0..2]`), `iPointB` (`[3..5]`), `iCssWidth` (`[6]`), `iStyle` (`[7]`: neutral, active, guide, BF16 gold), `iAlpha` (`[8]`), `iOwnerId` (`[9]`), `iFace0`/`iFace1` (`[10..11]`), `iOwnerCenter` (`[12..14]`), `iOwnerExtent` (`[15..17]`).
   - Uses homogeneous Liang-Barsky frustum clipping and perspective front-face edge culling (`face0 = face1 = -1` disables backface culling for free guide lines).
   - Shared across `spine.ts` and all 8 cutaway builders via `pushFreeGuideLine`, `pushBoxFrameXZ`, and `pushDashedFrameXZ`, preventing stride drift across builders.
3. **`SheetsPass` (`src/render/sheets.ts`)** — Arbitrarily oriented 3D thin tensor matrix sheets (`20`-float / `80 B` main instance + optional `6`-float orthonormal basis `iAxisU`/`iAxisN` + optional `4`-float progressive reveal frontier `iReveal`), staying within `13` vertex attribute locations (`<=` WebGL2 guaranteed minimum `16`).
   - Renders crisp adaptive `fwidth()` cell-grid lines (adapted from `bbycroft/llm-viz`) and samples live scalar values from a single **`RG32F` Float32 texture atlas** (`R` = Float32 tensor value, `G` = causal `-∞` mask flag) consistently on both `+N` and `-N` faces.
   - **Content-Keyed Atlas Caching (`PR-12`)**: `computeValueAtlasContentKey()` hashes view mode, bay ID, execution mode, `executionId`, token count, precision, active head/expert/token selectors, stepper step + discrete term bucket, reduced-motion state, and atlas dimensions. Pointer hover moves within the same head/expert/token skip GPU `texImage2D` re-uploads completely.
4. **`RibbonsPass` (`src/render/ribbons.ts`)** — Instanced translucent 3D dataflow bars and tapered directional arrowheads (`48 B`/instance, `depthMask(false)`, straight alpha blending).
5. **`FlatFlowRibbonsPass` (`src/render/ribbons.ts`)** — Dynamic 3D cubic Bezier and polyline flow ribbons with optional cross-fins (`crossRibbon`) and flush double-sided cell/span highlights (`addCellHighlightOnFrame`) rendered along oriented `SheetFrame3D` planes, with pulse animation gated strictly to active playback.

### WebGL Context-Loss Recovery (`PR-10`)

- **`webglcontextlost`**: Calls `ev.preventDefault()`, cancels any active camera fly, pending hover pick, and `requestAnimationFrame` handle; pauses all stepper, Mamba temporal, MoE cycling, and Tower spine-wave clocks; clears `lastUploadedAtlasKey` and `lastOverlayRenderKey`; calls `device.markLost(true)`; and displays an accessible fallback banner while keeping the sidebar controls, guided walkthrough text, and interactive `1 → 1M` memory chart fully operational.
- **`webglcontextrestored`**: Constructs fresh instances of all 5 GPU passes (`SolidsPass`, `SheetsPass`, `LinesPass`, `RibbonsPass`, `FlatFlowRibbonsPass`) into local variables before swapping references, resets cache keys, clears `device.markLost(false)`, hides the fallback banner, marks geometry dirty, and renders synchronously. If pass reconstruction fails, `device.markLost(true)` remains set and an accessible recovery-failed notice stays visible.

### Idle-RAF, Hover-Coalescing & Spine-Throttle Architecture (`PR-11`–`PR-14`)

- **Zero Idle RAF Spin (`PR-11`)**: `shouldAnimateCutawayPulse()` and `shouldContinueRafLoop()` permit continuous `requestAnimationFrame` loops **only** when a camera fly animation, an unpaused Tower spine replay, or an active cutaway playback loop is advancing frames (and WebGL is live and `reducedMotion` is false). Static inspection, pinned cells, hover, and paused steppers render strictly on demand.
- **Hover-Coalescing & Cached Insets (`PR-12`)**: `pointermove` CPU ray-picks are coalesced to at most one pick per animation frame via `pendingHoverRafId`. Cutaway viewport insets (`cachedCutawayInsets`) are cached outside per-frame hot paths, and DOM/SVG overlay sync short-circuits via `lastOverlayRenderKey`.
- **Lightweight Stepper Progress Sync (`PR-13`)**: Between discrete operand-term transitions during 60 fps Math Stepper playback, `controls.syncStepperProgressOnly()` updates only the scrub range input instead of rebuilding the sidebar DOM.
- **Discrete Spine-Wave Geometry Throttling (`PR-14`)**: `computeSpineWaveGeometryKey()` maps continuous replay progress `∈ [0, 1]` onto discrete stage/bay keys (`embedding`, `bay:1..52`, `final_norm`, `output`, `idle`), reducing `buildMacroSpine()` geometry rebuilds over a `5.5 s` (`330`-frame) instructional replay from `330` rebuilds down to `~55` discrete transitions while `syncSpineWaveHud()` smoothly updates the lightweight HUD.

---

## 3. Mathematical Execution, Cache Ownership & Memory Bounds

### Model Parameter Ownership & Weight LRU (`src/math/hybridFlow.ts`)

- `HybridModelParameters` separates immutable model weights from per-run executions (`HybridExecutionResult`) and incremental steps (`HybridIncrementalStepResult`).
- Every parameter instance carries a canonical 64-bit `parameterFingerprint` computed over seed, full layer pattern, `d`, `vocabSize`, IEEE-754 float64 bits of `rmsNormEps` and `residualOutputScale`, and vocabulary strings (`dc0568e222e22a4e` for `createDefaultHybridModel()`).
- Deterministic bay weights (`getBayWeights(bayId)`) are retained in a bounded LRU cache defaulting to **`DEFAULT_MAX_RETAINED_WEIGHT_BAYS = 52`** (all 52 bays + embedding/output = `299,192` Float32 values = **`1,196,768 B` ≈ `1.14 MiB`**).
- Shared read-only `Float32Array` parameter views are reused across executions, decode steps, and layer traces — eliminating per-step PRNG thrash and preventing duplicate weight buffer retention across multi-branch AR sessions.

### Immutable-by-Contract `HybridDecodeCacheState` (`src/math/cacheContracts.ts`, `src/math/incrementalDecode.ts`)

Every cached AR decode snapshot enforces strict ownership and provenance contracts:

1. **Private Non-Aliased Backing Buffers (Copy-on-Step)**:
   - Every `HybridDecodeCacheState` and nested `GqaLayerKvCache` / `MambaLayerBoundaryState` owns private, non-aliased `ArrayBuffer`s. Stepping or forking from `prevState` never mutates any typed array inside `prevState`.
   - Exact default miniature snapshot accounting (`L = 52`, `23 M`, `6 *`, `23 E`, `maxTokens = 16`):
     - Mamba-2 fixed state (`convState` `[128, 4]` + `ssmState` `[16, 4, 4]`): `23 × 3,072 B = 70,656 B` (**`69.00 KiB`**)
     - GQA KV allocated (`K` + `V` `[8, 16]`): `6 × 1,024 B = 6,144 B` (**`6.00 KiB`**; populated = `384 B/token` = `0.375 KiB/token`)
     - MoE sequence cache: **`0 B`** (stateless across tokens)
     - **Total allocated default snapshot:** **`76,800 B = 75.00 KiB`** exactly (excluding JS objects and educational traces).
2. **Digest Verification (`stateDigest` + Length-Prefixed Labels)**:
   - `computeDecodeStateDigest()` computes a deterministic 64-bit dual hash over `parameterFingerprint`, `processedTokens`, `maxTokens`, `prefixTokenIds`, **length-prefixed `prefixTokenLabels`** (preventing concatenation collisions such as `["ab","c"]` vs `["a","bc"]`), and every populated Float32 element of all 23 Mamba and 6 GQA caches (excluding trailing unpopulated KV slots `k >= seqLen`).
   - `validateDecodeCacheState()` verifies complete layer coverage (`23 M`, `6 *`, `23 E`), exact tensor dimensions, non-aliased backing buffers, finite populated values, matching `parameterFingerprint`, and exact `stateDigest` equality — immediately rejecting in-place tampering or cross-model state reuse.
3. **Isolated Lazy-Reconstruction Snapshots**:
   - By default, each step retains full simulator traces eagerly for primary cutaway bays (`DEFAULT_RETAINED_TRACE_BAYS = [1, 2, 6, 42, 43, 52]`, reported as `traceOrigin: 'computed_this_step'`).
   - Requesting any other bay `1..52` via `getLayerTrace(bayId)` deterministically reconstructs that bay (`traceOrigin: 'reconstructed_from_checkpoint'`) from private cloned boundary `Map`s and detached `residualStreams` checkpoints captured inside internal closures. Even if a caller mutates exposed `residualStreams`, `X_embed`, or `ReadonlyMap` containers at runtime, lazy reconstruction remains uncorrupted.
4. **Bounded Decode Branch History (`HybridDecodeSession`)**:
   - Defaults to **`maxBranches = 16`** (`DEFAULT_SESSION_MAX_BRANCHES`) and **`maxUniqueSteps = 128`** (`DEFAULT_SESSION_MAX_UNIQUE_STEPS`), with `pruneInactiveStepTraces: true` dropping reconstructible eager traces from inactive historical steps while deduplicating shared prefix steps across surviving branches.
   - `deleteDecodeSessionBranch()` and `pruneDecodeSessionBranches()` are strictly **active-branch-safe**: they refuse to delete `activeBranchId` (or the last remaining branch), never truncate `activeBranchId`, prune only the oldest inactive branches, and never destroy shared ancestor steps still referenced by surviving branches.
   - `DualDecodeSessionStore` (`src/model/rcPlayback.ts`) completely isolates the canonical Guided Tour AR session (`tour`, seeded from `TOUR_PREFILL_TOKEN_IDS = [0..7]`) from the user's `freeExplore` AR session (`[0, 1, 2, 3]` default prompt + user branches).

---

## 4. Intentional Nano v1.0 Boundaries & Shipped Topology A TwoTower Extension

### Intentionally Kept Simple for Nano v1.0

The following trade-offs are deliberate Nano `v1.0.0` scope boundaries rather than accidental omissions:

- **Single Composition Root (`src/main.ts`)**: `src/main.ts` (~4.2k LOC) directly coordinates state, GPU passes, camera choreography, and DOM bindings without a client framework or event-bus abstraction. Rather than refactoring Nano's locked entry point into a multi-architecture router, the TwoTower visualizer ships as an isolated Topology A MPA entry (`twotower/index.html` → `src/twotower/app/main.ts`).
- **Miniature CPU Simulation Scale**: `src/math/` intentionally targets fixed miniature dimensions (`d = 8`, `V_mini = 32`, `maxTokens = 16` for Nano; extended to `V_mini = 34`, `kvCapacity = 24` by TwoTower) with deterministic synthetic Float32/Float64 weights for instant sub-millisecond interactive inspection and bit-exact incremental equivalence, without WebGPU compute shaders, WASM tokenizers, or real checkpoint streaming.
- **Metadata-Only FP8 Selective Mode**: `[ BF16 | FP8 Selective ]` drives exact production memory accounting (`src/model/memory.ts`) and 3D visual boundary highlights (`src/model/spec.ts`) without simulating hardware FP8 E4M3 quantization noise inside the miniature Float32 forward pass.
- **Nano-Specific Cutaway Builders**: The 5 cutaway builders encode Nemotron 3 Nano's specific `M`/`E`/`*` + NoPE bookend topologies rather than a generic graph-layout DSL.

### How the Nano v1.0 Seams Power the Shipped TwoTower Entry (`/twotower/`)

The v1.0 math, cache, and WebGL2 rendering seams now directly power the shipped **Nemotron-Labs-TwoTower 30B-A3B** Topology A entry (`twotower/index.html` → `/twotower/` and `/llm-viz/twotower/`), reusing `31` shared Nano primitives (`src/render/`, `src/math/`, `src/model/`, `src/scene/`) while `scripts/verify_bundle_isolation.mjs` enforces zero TwoTower leakage back into Nano's `/` bundle:

| Shipped TwoTower Capability | Reused Nano v1.0 Seam | How It Connects |
| --- | --- | --- |
| **Layer-Aligned Read-Only KV Transfer (`6` GQA bays)** | `GqaLayerKvCache` (`cacheContracts.ts`) & `createExternalDecodeCacheState()` (`incrementalDecode.ts`) | Transfers defensively cloned, non-aliased Context prefix `KV[0..P-1]` (diffusion) or `KV[:-1]` (Mock-AR) keyed by `bayId`, with explicit `maxTokens >= seqLen` capacity validation and `conditioningDigest` binding. |
| **Externally Seeded Mamba-2 States (`23` Mamba bays)** | `MambaLayerBoundaryState` & `ExternalDecodeCacheSeed` (`cacheContracts.ts`, `mambaSim.ts`) | Seeds Denoiser Mamba-2 bays from immutable Context `(convState, ssmState)` snapshots (`S-1` for diffusion, `mamba_s2` for Mock-AR) validated across TypeScript and PyTorch `2.15.0` fixtures. |
| **Bidirectional Block Attention `[B × (P+B)]`** | `gqaSim.ts` mask stage & `SheetsPass` `RG32F` `G`-channel mask bit | Extends `gqaSim.ts` with rectangular block-bidirectional attention over inherited prefix KV + active `B`-token block while reusing Nano's `SheetsPass` shader unchanged. |
| **Split PixArt-α `adaLN-single` Conditioning** | `HybridModelConfig`, `parameterFingerprint`, & `applyRmsNorm` (`hybridFlow.ts`) | Hooks Denoiser-only `t = nMasked / B` modulation (`modulate → RMSNorm` on `M`/`*` vs. `RMSNorm → modulate` on `E`, unmodulated `norm_f`) cleanly into `src/twotower/` while keeping `src/math/` pure. |
| **Iterative Diffusion Refinement & Branching** | Immutable copy-on-step lineage (`HybridDecodeSession`, `stateDigest`, lazy trace reconstruction) | Generalized in `src/twotower/session.ts` into the 4-step block diffusion lifecycle (`Init` → `Refine` → `Commit Conf` → `Commit → Context`), Persistent Nested Timeline, and isolated 4-Act / 28-Beat Guided Tour. |

For the full TwoTower dual-tower architecture, 3-tier epistemic sourcing matrix, dual `GB`/`GiB` checkpoint reconciliation, 3-regime context-length scaling contract, 4-way memory accounting boundary, and Topology A isolation contract, see [**`docs/TWOTOWER_ARCHITECTURE.md`**](TWOTOWER_ARCHITECTURE.md) and [**`docs/TWOTOWER_USER_GUIDE.md`**](TWOTOWER_USER_GUIDE.md).
