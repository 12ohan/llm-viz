# Nemotron 3 Nano · engineering plan

Build a static Vite + TypeScript explainer with native WebGL2, an orthographic camera and a DOM/SVG annotation layer. Ship only schematic geometry, checkpoint metadata and deterministic examples. Begin with the full-spine view. The first review should already explain which bays accumulate token history.

## 1. World coordinate system and spatial proportions

Checkpoint basis: BF16 revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`. Hidden width 2,688, pre-RMSNorm, NoPE, vocabulary 131,072, untied output. Use one shared expert of intermediate width 3,712 as implemented in this checkpoint. The report lists two in Table 1. [Pinned config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) · [Pinned implementation](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/modeling_nemotron_h.py)

Use a right-handed world. **+X** runs from bay 01 to 52, **+Y** rises above the residual stream, **−Z** runs from the newest visible front of the drawing into cached depth. Number stored positions from front to back in append order; token 1 stays at the front. Dimensions below are full X × Y × Z extents in drawing units. Coordinates denote centers unless marked as intervals. World proportions express function; tensor callouts carry literal model dimensions.

| Macro element | Dimensions and placement |
| --- | --- |
| Bay lattice | Pitch 1.00. Bay i has center x = i − 26.5 and boundaries x ± 0.50. Shared ground line y = 0, front datum z = 0.50. Leave the gaps between solids visible. |
| Residual and bookends | Strip 56 × 0.30 × 0.78 at (0,0,0). Embedding plate 1.20 × 4.0 × 1.20 at (−27.4,2,0). Final RMSNorm wafer 0.24 × 2.6 × 1.0 at (26.45,1.3,0). Output plate 1.2 × 4.4 × 1.2 at (27.4,2.2,0). |
| M, open rectangular return | At local x = ±0.31, two uprights 0.16 × 4.8 × 0.58 centered at y = 2.4, z = 0. Crossbar 0.78 × 0.28 × 0.58 at y = 4.66. Hatched register insert 0.46 × 2.50 × 0.40 at (0,1.55,0). Leave the upper aperture open. Conv1D band 0.78 × 0.25 × 0.14 at (0,0.30,0.34). |
| E, low stepped cabinet | Body 0.86 × 2.70 × 0.84 at (0,1.35,0), grid on the front. Cap 0.64 × 0.35 × 0.60 at (0,2.875,0). The top sits well below the M crossbar. |
| Attention, tall fork | Foot 0.86 × 0.40 × 0.92 at (0,0.20,0). Two tines 0.20 × 6.40 × 0.58 at x = ±0.31, y = 3.60, z = 0. Crosspiece 0.82 × 0.22 × 0.62 at y = 4.50. Silhouette tops are 3.05 / 4.80 / 6.80 for E / M / attention. |
| K/V rails and slices | Four rail centers at local x = −0.33, −0.15, +0.15, +0.33 (K1,V1,K2,V2), y = 4.50. Each fixed guide is 0.045 × 0.06 × 9.60 from z = −0.30 to −9.90. Slice j = 1…T has center z = −0.30 − 0.60(j−1) − 0.24 and size 0.12 × 0.22 × 0.48, at y = 4.63. Pitch 0.60, clear gap 0.12. T ranges 1–16, default 8. Occupied envelope grows from 0.48 to 9.48. Keep the fit bounds fixed at capacity. |
| Ruler and brackets | Put the depth ruler at x = 28.5, y = 4.5 with ticks at the back of each slice. Label 1, 8, 16 and the current T when distinct. Project span endpoints from bay boundaries at y = −0.30, z = 0.55. Draw the four Figure 2 brackets immediately below the projected rail, using the layout rule in §4. |

At full fit use the full content width on 768 px screens. Put controls and narration below the diagram there; at 16:9 use a lower caption band. At 768 px each pitch should span at least 10 CSS px and a front-facing attention tine should rise at least 65 CSS px from its base in projection. The tall fork, open register, low cabinet and rearward rails supply visual depth even with one token. Keep the 1–16 ruler in token units in both precision modes.

**Cutaway coordinates.** Isolate the selected bay on a detail stage. Its local origin maps to (bay x,0,0) with unit scale and the same axes; hide neighboring macro geometry on this stage. Keep the macro selection on a separate miniature spine. All three stages share a residual strip 20 × 0.22 × 0.42 at (0,0,1.5), entry and addition joints of side 0.42 at (±9,0,1.5), and a pre-norm plate 0.28 × 1.20 × 1 at (−8,1.5,1.5). Route wires on orthogonal polylines between ports at face centers. Use X then Z then Y for doglegs unless an explicit rail is given. End each incoming tip at the destination surface.

| Detail stage | Anchors, extents and synthetic step |
| --- | --- |
| M · bay 01 prototype | Projection at (−6,2,0), size 1 × 2 × 1. Four convolution slots at (−4+0.55j,2,0), j = 0…3, each 0.45 × 1 × 1. Register at (0,3,−2), size 3 × 3 × 2. Close a loop around its front via (−2,3,0), (−2,5,0), (2,5,0), (2,3,0) and the register’s two side ports. Advance slot marks and register phase once per token. Carry the projection gate on z = 1 to gated RMSNorm at (4,2,0) then output at (7,2,0), each 0.6 × 1.6 × 1. Label 64 × 64 × 128 state entries and 4,096 → 2,688 output. |
| E · bay 02 prototype | Router at (−7,2,−1), size 0.6 × 1.6 × 1. Cell r,c at (−5.25+0.70c,1.2,−1−0.66r), r = 0…7, c = 0…15. Each 0.54 × 0.55 × 0.50. ID = 16r+c+1. Shared slab at (0,1.2,0.55), size 1.08 × 0.55 × 0.50, connects directly from normalized input. Routed choices for even T are 3,18,42,67,91,124 and for odd T are 6,21,48,70,99,117. Lift six by 0.35. Fan selected runs above the tray at y = 2.3+0.14k for selection rank k = 0…5. Collect at (6.4,2,−2); combine shared output at (7.8,2,0). Label routed and shared widths 1,856 and 3,712. |
| GQA · bay 06 prototype | For g = 0,1 use group center x = −4 or +4. Put a comb spine 6.8 × 0.25 × 0.30 at (x,2,0.30). Tooth j = 0…15 is 0.22 × 1.0 × 0.76 at (x−3+0.40j,2.50,0); Q ID = 16g+j+1. K and V centers at x−0.55 and x+0.55, y = 2; use macro slice z positions with slice size 0.72 × 0.28 × 0.48. Put Q/K/V input wafers at (−7,1,1), (−7,2,1), (−7,3,1), each 0.55 × 0.45 × 0.5. For the selected tooth trace Q → K dot products → causal softmax → weighted V read; bracket all teeth into the output plate at (7.5,2,0.6), size 0.55 × 1.4 × 0.8. Q and output width 4,096, K and V width 256 each. |

## 2. Palette, surface patterns and typography

| Token | Value and use |
| --- | --- |
| Canvas / top / front / right | Warm paper **#F6F3EB**; matte **#EDE9DF**, **#DDD8CB**, **#CBC5B8**. Opposite faces use their axis mate. Use the listed display-space sRGB values consistently in CSS and the canvas. |
| Ink / hairline / leader | Text **#403D38**. Edges **rgba(69,65,58,0.60)**. Leaders same RGB at 0.55. |
| Active / FP8 | Mineral oxide **#A04732**, active strokes at 1.0 opacity. FP8 stone **#BFB7A8** mixed 35% into the base face; add pattern 3. Retained BF16 parts use their base face. State-register overlays always report FP32 storage. |
| Text | System UI sans for controls and narration, 14/20 px, weight 400; headings 18/24, 600. Local system monospace for numbers, glyphs and measurements, 12/16, 500, tabular figures. Bracket captions 11/14. Use identical metrics for overlay layout and rendering. |

Use highp fragment arithmetic with face-local world coordinates u,v measured along the two in-plane axes. Orient +u toward increasing X on front/top, +v toward increasing Y on front/right. On top +v points toward −Z; on right +u points toward −Z. For a repeating coordinate q in cycles, let d = |fract(q+0.5)−0.5| and a = max(fwidth(q), 0.0001). Line coverage is 1−smoothstep(max(0,h−0.5a),h+0.5a,d), multiplied by 1−smoothstep(0.25,0.50,a) to fade below sampling resolution. h is half-width in cycles. Mix toward ink with the specified opacity times coverage. No textures.

| ID | Face treatment |
| --- | --- |
| 0 | Matte face color only. |
| 1 | Register front/top. q = (u+v)/(√2 × 0.32)+phase, h = 0.09, opacity 0.30. Step phase to (T mod 4)/4; pin the surface outline. |
| 2 | Cabinet front and tray-cell tops. Evaluate u/0.24 and v/0.24 with h = 0.055. Use max of the two coverages at opacity 0.24. |
| 3 | FP8 overlay. q = (u−2v)/(√5 × 0.55), h = 0.07, opacity 0.18. Apply after family pattern so precision and block identity remain independent. At macro fit rely on the FP8 face tone once hatching fades. |

## 3. WebGL2 rendering and camera contract

**Frame and buffers.** Request WebGL2 with depth, antialiasing, opaque canvas. Use a scene framebuffer with RGBA8 color and DEPTH_COMPONENT24 depth at drawing-buffer resolution; use 4× MSAA if both formats support it, else best common sample count ≤4 (single sample fallback). Resolve color to a single-sample target if needed and blit to canvas. Use the same multisample coverage for solids, lines and arrows. Clear color to paper, depth to 1. All layouts below are tightly packed float32, byte offsets shown, instance divisor 1. Use integer-valued float attributes for enums and pass IDs flat. Shared uniforms include view-projection, framebuffer width/height, DPR, phase and palette.

| Pass | Input and drawing contract |
| --- | --- |
| 1 · opaque solids | Static 24-vertex unit cube centered at origin, 36 triangle indices, axis normal per face. Instance stride 48 B: center xyz @0, extent xyz @12, family pattern @24, FP8 flag @28, phase @32, bay ID @36, face-pattern bitmask @40, tone selector @44. CCW faces, back-face culling, LESS, depth writes on, blending off. Enable POLYGON_OFFSET_FILL with factor 0, units +1; disable immediately after. Axis-aligned instances suffice. |
| 2 · visible hairlines | Four quad corners (endpoint 0/1, side −1/+1), two triangles per segment. Stride 72 B: endpoint A xyz @0, B xyz @12, CSS width @24, style @28, alpha @32, owner ID @36, two adjacent face IDs @40/@44 (−1 for a free wire), owner center xyz @48 and full extent xyz @60. Face IDs 0…5 = +X,−X,+Y,−Y,+Z,−Z; face-mask bit i = 1 << i. Emit only twelve true cuboid edges. Suppress an edge when both owner faces face away. LEQUAL, depthMask(false), no face culling, straight-alpha SRC_ALPHA / ONE_MINUS_SRC_ALPHA. Draw neutral runs first, active last. |
| 3 · directional tips, detail only | Draw instanced opaque, two-sided world triangles; tip and two base corners xyz @0/@12/@24, style @36, stride 40 B. Depth test LEQUAL, depth writes on. For segment direction d, tip P, base center P−0.32d, corners base ±0.12 normalize(cross(d,eyeDirection)). If cross length <0.10, place the tip on the preceding visible dogleg. Trim stroke at the base. Restore masks and blend/cull state before next clear. |

**Line math.** Set dpr = min(devicePixelRatio,2). Backing size = rounded CSS size × DPR (round the product), minimum 1. Given clip endpoints c, map each to framebuffer pixels p = (c.xy/c.w × 0.5 + 0.5) × (W,H), origin bottom-left. Clip crossing segments to the view volume before expansion. Skip projected lengths <0.25 framebuffer pixels. Set n = (−Δy,Δx)/length(Δp), half = CSSwidth × dpr / 2, feather = 0.5 framebuffer pixels. Offset each endpoint by side × n × (half+feather) **in pixels**, then map back with clip.xy = (2p/(W,H)−1) × clip.w. Vary signed transverse pixel distance s from −(half+feather) to +(half+feather). Coverage = 1−smoothstep(max(0,half−feather),half+feather,abs(s)); discard zero coverage. Default width 1.0 CSS px; active width 1.6. Feather multiplies opacity. Leave depth writes disabled for the entire line pass.

**Depth budget.** Tight orthographic near/far below; window depth is linear. Pull line centers forward by at most one depth24 step: subtract 2 × clip.w / (2²⁴−1) from clip.z. With the solid offset this leaves roughly two steps of tolerance. Keep (far−near) ≤ 160 units for these stages, so the combined axial tolerance stays below 0.000020 units. Use the front-face edge filter even with this bias. For cuboid strokes on inclined faces, evaluate line depth at the covered sample’s screen position on its adjacent visible face plane (ray/plane intersection, or equivalent plane gradient); use the nearer in-bounds incident face, falling back to centerline depth on the outside silhouette. This avoids internal edge dropouts from expansion on slopes. Never increase global bias to fix a missing line. Inspect all slices at both DPRs against opaque foreground faces.

**Camera.** Column vectors, WebGL NDC z ∈ [−1,+1]. Eye direction from target e = (sin a cos b, sin b, cos a cos b), up reference (0,1,0). Macro a = 24°, b = 27°; detail defaults 22°, 28°. Right R = normalize(up × e), up U = e × R. For all eight corners p of the world AABB calculate dot(p,R), dot(p,U), dot(p,e) and the min/max in each axis. Use the three interval midpoints in that basis for target C. For scene viewport aspect A, half-height H₀ = max(rangeU/(2 × 0.76), rangeR/(2 × 0.88 × A)); orthographic bounds ±H₀ vertically and ±AH₀ horizontally. Reserve controls/bracket caption rows outside that viewport. Macro fit includes bookends, the ruler and full 16-token capacity. Detail fit includes its full assembly and arrows. Store independent fitted cameras. At zoom z use H = H₀/z, z ∈ [1,8]. Eye = target + 80e. From corner distances along the forward axis set near = max(0.01,minDistance−2), far = maxDistance+2. Refit on resize; preserve zoom and projected pan in units of H. Pan only in the R,U plane.

**Frame lifecycle.** Input mutates CPU state and requests one pending animation frame. Its single draw function snapshots state and camera, uploads dirty instances, draws three passes, then projects and places labels before returning. Only playback or camera easing schedules another frame. ResizeObserver on the CSS-sized host measures its content box, reallocates attachments, updates projection and invokes that same draw function synchronously before paint. Cancel redundant pending frames and resume animation if needed. Ignore zero-sized hosts and same-size notifications; canvas and overlay are absolutely positioned so backing-size changes cannot resize the host. Watch resolution media-query changes for monitor/DPR switches. On webglcontextlost prevent default, cancel frames and retain CPU state. On restoration recreate programs, VAOs, buffers and attachments, restore GL state and redraw the selected view. If WebGL2 or restoration fails, retain the HTML spine legend, readout and an accessible status message.

## 4. Picking, projected labels and interaction

**Ray.** Subtract canvas client rect origin from client coordinates. For CSS dimensions w,h and local point x,y, use ndcX = 2x/w−1 and ndcY = 1−2y/h. Multiply (ndcX,ndcY,−1,1) and (ndcX,ndcY,+1,1) by invViewProj, divide each xyz by its own w. Origin = near point; direction = normalize(far−near). Intersect slabs axis by axis. A parallel axis (|direction| < 10⁻⁸) misses only if the origin lies outside the interval. Accept when exit ≥ max(enter,0); choose smallest nonnegative entry, with lower ID winning equal entries within 10⁻⁶.

**Macro targets.** Test actual solids first and return the nearest hit’s bay, including rails and slices. Fill the inter-bay gaps with a common thin picking fascia at z = 0.50…0.52, y = −0.15…6.8; each box spans its entire pitch in X. Apply fascia only if geometry misses. This gives even the gap over a low E cabinet a stable owner. Rear rails use their owner’s ID even at T=1. Clear hover on pointer leave. In detail use real component AABBs, including the current raised position of each expert.

**Gestures.** Pointer-down captures the pointer and records its CSS position. Switch irreversibly to drag at Euclidean displacement ≥5 px; up below threshold selects. A single click pins the bay, Enter or an “Inspect bay” button opens its cutaway from M2 onward. Drag pans by −Δx·2H/h along R and +Δy·2H/h along U. For wheel, normalize line units to 16 px and page units to h, then multiply zoom by exp(−0.0015 × deltaPixels); pinch multiplies by current/start finger separation. Anchor at the cursor or pinch midpoint: intersect its ray with the plane through target normal to e before and after scaling and translate target by oldHit−newHit (old and new midpoint for a moving pinch). Clamp zoom 1…8. Two-finger input cancels any pending selection. Limit touch-action suppression to canvas. Left/right step within 01–52, starting at 01 if empty; Esc clears selection/hover and returns to overview, Reset additionally restores fit, T=8 and BF16. Native sliders and buttons retain keyboard handling when focused.

**Overlay layout, in CSS pixels.** The SVG and absolute DOM layer share the canvas content-box origin. Use the exact draw snapshot; screen anchor = ((ndc.x+1)w/2,(1−ndc.y)h/2). Cache text metrics after fonts settle. In the same draw call, project all anchors, solve placements, then batch SVG path and DOM transform writes. No independent overlay RAF, CSS transform transition or delayed framework effect. Reserve 56 px above the drawing for label tiers and 72 px below for bracket captions; include that origin offset in both projection and picking when using the inner viewport. For layout, canvas/inner viewport and both overlay coordinate transforms must agree.

1. At zoom 1 reserve only **01, 06, 13, 20, 27, 34, 43, 52**. Project their bay tops, use 26 × 16 boxes, center x clamped to [15,w−15]. Place in ascending x then ID. Try top-band rows nearest the drawing first, at center y = −9, −27, −45 relative to the inner viewport. A row fits if box plus 4 px horizontal clearance is disjoint from every occupied box. If all collide, choose the first row with a free interval and shift horizontally the least distance there, ties to the left. Leaders go from projected anchor to (anchorX,rowY+8) then the nearest bottom edge of the box. Store occupied rectangles, so mandatory numbers never disappear at full fit. Offscreen anchors at zoom >1 are culled.
2. At zoom test projected bay pitch against 30 CSS px. Above threshold, offer remaining visible IDs in ascending order, accepting their first free tier at their natural x. Skip collisions; never displace a mandatory ID. Hover/focus gets one transient tag at the bay, clamped inside the drawing below the top tiers. A pinned bay’s ID stays in the inspector. Respect pointer-events:none on decorative labels; expose the same values through the keyboard bay inspector.
3. Draw connected bracket segments 8 px below their projected boundaries along the residual baseline with 4 px end ticks. Expand the caption beyond a short span using a leader to its midpoint. Place four captions in span order in 14 px rows starting 18 px below the lowest projected bracket endpoint. Use measured caption width + 8 px and the same tier/shift rule as the macro numbers, up to four rows. Reserve this space inside the overall stage. When zoom crops a span, clip its line; show its caption only if its midpoint remains visible. Span endpoints belong to the bracket layer; the eight static numbers remain the only persistent bay labels.

## 5. Model interfaces and state schema

The declarations below are a documentation contract. Keep CPU model/state independent of WebGL. Derive memory, projected coordinates, instance arrays and effective component precision each frame from the same inputs.

```ts
type PrecisionMode = 'bf16' | 'fp8_selective'
type Kind = 'M' | 'E' | '*'
type Vec3 = readonly [number, number, number]
type Bracket = readonly [first: number, last: number, motif: string, repeats: number]
interface BayDescriptor {
  id: number                         // one-based
  kind: Kind
  feeder: boolean
  sensitive: boolean
  bracket: 0 | 1 | 2 | 3
}
interface NemotronSpec {
  revision: string
  pattern: string
  bays: readonly BayDescriptor[]
  brackets: readonly Bracket[]
  hidden: 2688
  mamba: { heads: 64; channels: 64; state: 128; groups: 8; kernel: 4 }
  moe: { routed: 128; active: 6; shared: 1; width: 1856; sharedWidth: 3712 }
  gqa: { q: 32; kv: 2; headDim: 128 }
}
interface MemoryReadout {
  tokens: number
  precision: PrecisionMode
  fixedMambaBytes: number
  kvBytesPerToken: number
  referenceBytesPerToken: number
  kvBytes: number
  referenceBytes: number
  hybridTotalBytes: number
}
interface CameraPose {
  target: Vec3
  azimuthDeg: number
  elevationDeg: number
  zoom: number
}
interface SceneState {
  view: 'macro' | 'detail'
  precision: PrecisionMode
  tokens: number
  hoveredBay: number | null
  selectedBay: number | null
  selectedPart: string | null
  camera: CameraPose
  savedMacroCamera: CameraPose
  reducedMotion: boolean
  beat: number | null
  elapsedMs: number
  playing: boolean
}
type Cue = readonly [atMs: number, action: 'sweep' | 'trace' | 'tokens' | 'head' | 'focus', value: string | number]
interface BeatDefinition {
  id: 1 | 2 | 3 | 4 | 5 | 6
  view: SceneState['view']
  bay: number | null
  focus: string                       // named scene anchor or assembly AABB
  azimuthDeg: number
  elevationDeg: number
  durationMs: number
  cues: readonly Cue[]
  takeaway: (mode: PrecisionMode) => readonly [string, string]
}
declare function memoryFor(spec: NemotronSpec, tokens: number, mode: PrecisionMode): MemoryReadout
```

Build one descriptor per character in `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`. Feeder means M immediately followed by attention. Sensitive means attention or feeder. Brackets (inclusive) are **01–35 [M·E·M·E·M·*·E] ×5**, **36–41 [M·E] ×3**, **42–44 [M·*·E] ×1**, **45–52 [M·E] ×4**. The model exposes integer counts; apply padding, separator dots and unit rounding in UI only.

Default to macro fit with no selection or beat. Precision control reads **[ BF16 | FP8 Selective ]**. A toggle preserves T, selection, geometry, camera and playback position. In selective mode tag the six attention bodies and preceding six M bodies BF16; tag the remaining 40 bodies FP8. Override all 23 Conv1D strips to BF16, all SSM state inserts to FP32 and the six KV cache contents to FP8. Distinguish projected operating precision from stored memory dtype in inspector text. Embedding, output and address guides remain neutral schematic elements.

For one sequence, fixed bytes F = 23 × (64 × 64 × 128 × 4 + (4,096 + 2 × 8 × 128) × 4 × 2). Per-token KV bytes at L attention layers = L × 2 × 2 × 128 × bytesPerScalar; use L = 6 or 52 and byte widths 2 or 1. Show fixed Mamba, actual KV and reference KV side by side, with the latter two slopes. Unit functions divide by 1,024 or 1,048,576 only at display time. Model math accepts nonnegative integer T independently of the 1–16 scene limit.

## 6. Phased build contracts, tests and risks

Treat each gate as a hard stop for subsequent milestone work until its checks pass. Review M1 as a standalone screen. File paths below describe the future clean app root; leave the existing reference checkout intact.

| Milestone | File-by-file ownership | Exit gate |
| --- | --- | --- |
| **1 · macro first** | `package.json` names Vite dev/build/preview and Vitest test scripts; lockfile pins dependencies. `tsconfig.json` sets strict DOM/ES modules. `vite.config.ts` reads configurable base. `index.html` hosts canvas and accessible controls. `src/main.ts` wires input and redraw. `model/spec.ts`, `model/memory.ts`, `model/state.ts` under src own §5 metadata, byte math and state respectively. `render/camera.ts`, `render/pick.ts`, `render/device.ts`, `render/solids.ts`, `render/lines.ts` own matrices, intersections, lifecycle and the first two passes. Keep shader strings beside their pass. `scene/spine.ts` emits macro instances, bounds, anchors and targets. `ui/overlay.ts`, `ui/controls.ts`, `ui/theme.css` own labels/brackets, gestures/readouts and tokens. `tests/spec.test.ts`, `tests/memory.test.ts`, `tests/pick.test.ts` own the fixtures below. | Vitest and production build pass. Inspect at 1366×768 and 768×600, DPR 1 and 2. Both bookends, 52 silhouettes, four brackets, eight static numbers and the 1–16 ruler are readable. Pick 42, zoom, pan, reset and scrub 8 → 9 in each mode. Exactly six cache fronts advance and the fixed value holds. Capture the two viewport sizes; review them before opening M2. |
| **2 · three cutaways** | `scene/cutaways/mamba.ts`, `moe.ts`, `gqa.ts` emit §1 detail layouts, logical ports and component hits. `render/arrows.ts` adds pass 3. `ui/inspector.ts` provides dimension and precision text, including the routing note. Extend `main.ts` to enter/exit detail and reuse existing render passes. `tests/cutaways.test.ts` checks instance counts, stable IDs, hit ownership and shared-branch width. | Open 01, 02, 06 and a second bay of each type. Register footprint holds across T; six routed IDs plus one shared branch light up; Q1/Q16 share pair 1 while Q17/Q32 use pair 2. Conv strip and SSM stay distinct under FP8. Return to the same macro camera, selection and T. |
| **3 · guided walkthrough** | `story/beats.ts` holds the six shots and two-sentence takeaways from the brief. `story/timeline.ts` evaluates absolute elapsed time into scene state. `ui/storyControls.ts` owns Previous, Next, Pause, Replay and source-linked beat 6 marginalia. `tests/timeline.test.ts` checks stepping, replay determinism, precision and reduced motion. | All six beats work in either precision. Enter any beat directly and compare the final state to sequential playback. Pausing freezes camera and simulation at the same timestamp. Reduced motion snaps between poses and leaves each takeaway readable. |
| **4 · Pages** | `.github/workflows/pages.yml` builds and publishes the chosen app root. Update Vite base handling only if either deployment fixture fails. `tests/pages.spec.ts` and `playwright.config.ts` exercise production assets, keyboard navigation, resize and context restoration in a browser. | Test both root and repository subpath, direct load and refresh, with zero failed asset requests. Verify context loss/restoration mid-selection and reduced-motion playback. Deploy the passing build from the default branch; smoke-test the published Pages URL. |

**M1 Vitest fixtures**

| Test | Exact expected result |
| --- | --- |
| Pattern and roles | Length 52; counts M/E/* = 23/23/6; attention IDs [6,13,20,27,34,43]; feeder IDs [5,12,19,26,33,42]. Twelve sensitive, 40 eligible. Check all 23 Conv1D overrides and six FP8 cache overrides independently. |
| Report brackets | Expand motifs MEMEM*E ×5, ME ×3, M*E ×1, ME ×4; concatenation equals the checkpoint string. Inclusive spans [1,35], [36,41], [42,44], [45,52] partition 01–52 once. Lengths 35+6+3+8 = 52. |
| Fixed and growing bytes | Both modes F = **49,364,992 B**. At T=8,9 BF16 actual KV = **49,152 / 55,296 B**, reference = **425,984 / 479,232 B**. FP8 actual KV = **24,576 / 27,648 B**, reference = **212,992 / 239,616 B**. Slopes are actual/reference **6,144/53,248 B**, then **3,072/26,624 B**. |
| Crossover with reference | Solve F+actualKV(T) = referenceKV(T). Exact equality at **1,048 BF16** and **2,096 FP8** tokens. For each mode test equality, hybrid total greater one token earlier, smaller one token later. Separately test the first T with actual KV ≥ F: **8,035 BF16**, **16,070 FP8**, and the preceding T below F. |
| Projection round trip | At 768×600 host, default fit, T=8, project bay 42’s register front at **(15.5,1.55,0.20)** into client CSS coordinates using a nonzero host origin (37,53) and its overlay inset. Cast the §4 ray back. Assert nearest visible bay = 42, ray-to-probe distance <10⁻⁴ world units, and unchanged hit at DPR 2. Repeat after 2× cursor zoom. Project a second screen point and assert both ray directions agree to 10⁻⁶ while origins differ. |

**Shot timing for M3.** Use 3,200 ms per beat, including a 650 ms camera move with smoothstep easing t²(3−2t). Start each beat from its declared baseline T=8, preserving precision. Replay reconstructs state from time zero. Bind the six targets in order: (1) fit whole spine, sweep 01–52 over 650–2,650 ms; (2) fit embedding/01 inlet/add joint, trace row then split at 1,000/1,800 ms; (3) fit 01 register and slots, step to T=9 at 1,600 ms; (4) fit full 02 tray at elevation 62°, show T=8 then T=9 at 1,800 ms; (5) fit both 06 combs and complete tracks, select Q1/Q16/Q17 at 700/1,400/2,100 ms; (6) fit macro, step T to 9 at 1,600 ms, trace the six writes in layer order at 100 ms intervals, settle focus on the output at 2,500 ms. During the six writes, use illustrative highlights; commit all cache lengths and byte counters together. Beat 6 margin cites the independent 3.3× throughput measurement (FP8, single H200, 8K/16K, versus Qwen3-30B-A3B-Thinking-2507) and 86.3% RULER-100 at 1M. The MoE inspector explains sigmoid routing, selection bias, squared ReLU, bias-free linear projections and zero dropout. Pretraining used aux-loss-free bias adjustment at 10⁻³ alongside load-balancing loss coefficient 10⁻⁴; RLVR froze router weights and continued expert bias updates. [NVIDIA technical report, Figure 1, Table 3 and §§2.1, 2.4, 3.2.5, 4.2](https://arxiv.org/html/2512.20848v1)

| Beat | Two-sentence takeaway for the caption band |
| --- | --- |
| 1 | A token traverses all 52 layers in order. The pattern contains 23 recurrent updates, 23 expert mixtures and six attention lookups. |
| 2 | The embedding gives this token 2,688 features. Each layer reads a normalized representation and adds its contribution back to the residual stream. |
| 3 | Mamba-2 carries information forward in a recurrent state and a short convolution window. Each new token updates both within a fixed footprint. |
| 4 | The router selects six of 128 experts for each token. Their weighted outputs combine with the wider shared expert that every token uses. |
| 5 | Sixteen query heads read the same key and value history while computing their own attention weights. The other sixteen read the second pair, giving all 32 heads access to context through two KV heads. |
| 6 | Each additional token adds **{6 KiB of BF16 / 3 KiB of FP8}** key and value data across the model, while recurrent storage stays fixed. After the final layer, the output head maps the normalized residual to vocabulary scores. |

**Release contract.** Use npm clean install against the lockfile, typecheck, Vitest, then Vite production build. Build-time base defaults to `/`; for project Pages pass `/<repo>/` derived from the repository name, including both slashes. Use Vite-resolved assets or BASE_URL for runtime URLs. Run browser smoke tests against dist mounted at each literal base. Pages source = GitHub Actions. On default-branch push or manual dispatch, build using checkout and setup-node, upload dist with upload-pages-artifact, then a dependent deploy-pages job in the github-pages environment with pages:write and id-token:write. Build needs contents:read. Serialize deployments in a pages concurrency group. Keep credentials out of client build inputs. [Vite Pages guide](https://vite.dev/guide/static-deploy.html#github-pages) · [GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

### Technical Risks & Edge Cases to Watch During Implementation

- **Thin faces and edge ownership.** A global bias can expose rear edges; depth-writing transparent fragments can erase neighboring strokes. Check T=16 obliquely at DPR 1 and 2, especially a rear track behind a front slice. Check the MSAA fallback, line endpoints, nearly end-on segments and coplanar joins. Deduplicate coincident edges and trim endpoints at joints to avoid doubled ink.
- **Sampling.** Face patterns can moiré before lines appear individually. Apply derivative fading before compositing and keep geometry readable with all patterns faded. Test fractional DPR, browser zoom, antialiasing off and display changes.
- **Projection and resize.** A stale inverse matrix, mismatched CSS/backing sizes or an overlay inset applied twice shifts the ray and leaders. Exercise scrolling, pinch midpoint drift, resize during easing, hidden/zero-size tabs and context loss. Read CSS sizes outside per-label write loops.
- **Crowding.** Short bracket spans 36–41 and 42–44 need separate caption rows. Check 768 px in both orientations and 375 px for label collisions; at phone widths retain the eight static IDs and allow bracket rows to grow below the stage. Keep controls reachable when text is enlarged. Return keyboard focus to the originating bay after closing detail.
- **Silhouette and picking.** Overlapping rails can cover nearby bays in projection. Frontmost geometry wins; keyboard stepping remains available. Inspect 42 beside the taller 43 and verify active ink cannot show through the adjacent solid.
- **Precision semantics.** The cache can be FP8 while its attention body stays BF16. FP32 register storage and BF16 convolution coexist within one M bay. Keep component-level overrides and byte tests independent. Omit scale metadata, batching, padding, allocator reserves and weights from logical cache totals. Preserve the same rail geometry in both modes.
- **Story, browser and deployment.** Cancel stale gesture captures on pointercancel and stop playback on manual exploration. Clamp elapsed time after a background-tab pause. Check Safari and Chromium WebGL2, pointer-device changes, denied GPU contexts and restoration. Repository subpath refresh must resolve every shader, style and script through the configured base.
