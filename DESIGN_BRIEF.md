# NVIDIA Nemotron 3 Nano 30B-A3B — architectural explainer

**Verification flag.** Table 1 of NVIDIA’s report lists two shared experts. The released BF16 checkpoint specifies one shared expert of width 3,712, exactly twice the routed width of 1,856. Draw one double-width shared branch per MoE layer. The supplied 52-character layer order matches the checkpoint. Use NoPE despite the rotary fields in the config. Qualify the parameter caption as “31.6B total · 3.2B active per token excluding embeddings · 3.6B active including embeddings.” [Report, §2.1 and Table 1](https://arxiv.org/html/2512.20848v1) · [Checkpoint config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json)

## 1. Macro 52-Layer Layout

Lay the model out like a long architectural section. Left to right is **layer order**. Into the page is **cached token position**. Height separates the residual path from the operations above it. Start in a shallow orthographic three-quarter view with both ends visible. A continuous graphite strip carries the 2,688-channel residual stream from embedding to final RMSNorm and the untied output head. Those bookends sit outside the numbered spine.

Place one equally spaced bay per character. Preserve each glyph’s silhouette at every zoom level. **M** has a square return path, **E** a tiled face, and **\*** a fork with two paired tracks extending behind it. In the full-spine view, print exactly eight static bay numbers: **01, 06, 13, 20, 27, 34, 43, 52**. Keep all eight readable by staggering their leaders on narrow screens. Reveal other numbers individually on hover or focus and progressively on zoom when their projected labels fit. Keep the wider gap before 43 and the nine bays after it visible.

`MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`

Directly underneath, bracket the four spans from Figure 2: **01–35** `[M·E·M·E·M·*·E] ×5`, **36–41** `[M·E] ×3`, **42–44** `[M·*·E] ×1`, **45–52** `[M·E] ×4`. Align end ticks to bay boundaries. [Figure 2](https://arxiv.org/html/2512.20848v1#S2)

**23 Mamba-2 · 23 MoE · 6 attention.** Only bays **06, 13, 20, 27, 34, 43** acquire K/V history. Scrubbing the token count advances a common depth ruler across those six separate caches. At each M bay a fixed-size register updates in place. Each E bay has zero token-history storage. For one sequence in BF16, each added token contributes **6 × 2 (K and V) × 2 KV heads × 128 channels × 2 bytes = 6 KiB** of logical KV payload. An otherwise identical 52-attention-layer spine would add 52 KiB. Keep the unit “per token” visible at every zoom.

Set a quiet, tabular-figure readout under the depth ruler. At eight tokens it reads **Mamba state 47.1 MiB fixed** · **KV 48 KiB (+6 KiB/token)** · **52-layer GQA reference 416 KiB (+52 KiB/token)**. Scrub to nine and only the latter two values advance, to 54 and 468 KiB. Use graphite text with a steady position and keep each unit attached to its value. The reference uses the same two KV heads, head width and BF16 precision.

Use warm white paper, matte neutral faces, graphite hairlines and a single oxide-red selection stroke. Reserve solid arrowheads for dataflow and fine elbow leaders for screen-space callouts. Anchor each label to its part and suppress minor labels before they overlap. Use instanced cuboids and slabs throughout. Prohibit dark-mode neon cyberpunk styling, glowing bloom and floating spheres. The visual reference is Bycroft’s movement from a whole model to a legible tensor face. [llm-viz](https://github.com/bbycroft/llm-viz)

Add **[ BF16 | FP8 Selective ]** beside the ruler. Selective mode preserves BF16 compute in the six attention bays and their six Mamba feeders **05, 12, 19, 26, 33, 42**. Indicate FP8 weights/activations in the other 40 bays, keeping every Mamba Conv1D strip BF16. All six attention caches switch to FP8 even beside BF16 attention bodies. Across 8 → 9 tokens the readout becomes **24 → 27 KiB (+3 KiB/token)** and the same-precision GQA reference **208 → 234 KiB (+26 KiB/token)**. Fixed Mamba storage stays **47.1 MiB**. [NVIDIA report, §4.2](https://arxiv.org/html/2512.20848v1#S4.SS2)

## 2. Three Distinct Block Geometries

In every cutaway, split the incoming strip at the left edge. One path goes straight to a square addition joint. The other rises through an RMSNorm wafer, traverses the selected operation and returns to the joint at width 2,688. Keep this bypass visible while the internals unfold.

| Block | Geometry and synthetic flow |
| --- | --- |
| **M · fixed-state register** | Behind the input projection, mount a four-slot causal convolution strip beside a square-cut register labeled **64 heads × 64 channels × 128 state values**. Slide the convolution window one slot. Trace a rectangular loop from prior state through the selective update and back into the same register. Change its face hatching in place. A separate gate rail meets the state readout at gated RMSNorm, followed by the 4,096 → 2,688 output projection. |
| **E · expert bank** | Set **128** individually countable expert slabs in an 8 × 16 tray. Mount the **shared** slab outside the tray, with double the intermediate-width dimension. From the input, one rail feeds the shared slab directly. A second rail reaches a 128-score sigmoid router. Raise exactly six routed slabs and draw only their six paths. Each routed inset reads **2,688 → 1,856 → 2,688**, with squared ReLU at the turn. The shared inset reads **2,688 → 3,712 → 2,688**. Weight and sum the six returns, then add the shared output. |
| **\* · grouped-query attention comb** | Split the input into Q, K and V projections. Cut Q into two combs of 16 teeth, each tooth 128 channels wide. Behind each comb place one K track and one V track, also 128 channels wide. Append the current K/V slices once to each pair. Select a Q tooth, trace its score against visible K positions through the current token, then use its softmax weights to mix the matching V slices. Join all 32 head outputs at a **4,096 → 2,688** projection. Keep the 16 teeth independently selectable even though they share a track pair. |

## 3. Guided Walkthrough Storyboard — Six Beats

Use a pauseable timeline with Previous, Next and Replay. Selecting a bay opens its cutaway and retains its number on a miniature spine. Ease between camera targets over 650 ms, keep layer order screen-right, and snap to the destination with reduced motion. Mark the timeline **Illustrative token flow**.

| Beat | Camera focus and synthetic action | On-screen takeaway |
| --- | --- | --- |
| 1 · Read the spine | Fit embedding through output, with the residual strip in the lower third. Sweep a flat index tab from 01 to 52, then leave the six attention glyphs outlined. | A token traverses all 52 layers in order. The pattern contains 23 recurrent updates, 23 expert mixtures and six attention lookups. |
| 2 · Enter the stream | Push in on the embedding face, 01 inlet and its addition joint. Lift one illustrative token row, bracket its width, then trace the bypass and normalized branch. | The embedding gives this token 2,688 features. Each layer reads a normalized representation and adds its contribution back to the residual stream. |
| 3 · Rewrite state | Pivot around 01 until the convolution slots and register face are both visible. Step between two adjacent token positions, shifting the window and replacing the hatching inside the same outline. | Mamba-2 carries information forward in a recurrent state and a short convolution window. Each new token updates both within a fixed footprint. |
| 4 · Choose experts | Track to 02 and tilt toward a plan view of the entire tray. Alternate two seeded token examples, hold their six selected IDs long enough to count, and keep the direct shared path inked throughout. | The router selects six of 128 experts for each token. Their weighted outputs combine with the wider shared expert that every token uses. |
| 5 · Share history | Track via 03–05 to 06, framing both combs and their cache depth ruler. Select Q1 and Q16 in sequence, then Q17, tracing each lookup to its paired tracks. | Sixteen query heads read the same key and value history while computing their own attention weights. The other sixteen read the second pair, giving all 32 heads access to context through two KV heads. |
| 6 · Add one token | Pull back to the full spine, keeping all six cache ends in frame. Advance from eight to nine tokens, adding one slice at each fork in traversal order, then land on the final norm and vocabulary projection. | Each additional token adds 6 KiB of BF16 key and value data across the model, while recurrent storage stays fixed. After the final layer, the output head maps the normalized residual to vocabulary scores. |

In beat 6, adapt the first takeaway sentence to the selected KV precision. Add two small source-labeled margin notes: **3.3× throughput** versus pure-attention Qwen3-30B-A3B-Thinking-2507 at 8K input / 16K output on one H200, both models benchmarked with FP8, best of vLLM and TRT-LLM per model (Figure 1); **86.3% RULER-100 at 1M tokens**, rounded from 86.34 (Table 3). These are independent reported results. [Figure 1 and Table 3](https://arxiv.org/html/2512.20848v1)

In the MoE inspector, note that linear projections have no biases and the model uses no dropout. During pretraining the aux-loss-free balancing mechanism updated expert selection biases at rate 10⁻³ alongside a conventional balancing loss with coefficient 10⁻⁴. RLVR froze router weights while expert bias updates continued. [NVIDIA report, §§2.1, 2.4, 3.2.5](https://arxiv.org/html/2512.20848v1)

## 4. Intentional Abstractions & Verification Flags

- **Scale.** Layer, expert and head counts are literal. Dimension brackets give exact logical widths. Bay spacing, housing thickness, arrow lengths and camera time serve legibility. A hatched register face stands for many values. The 8 × 16 tray is a packing choice. Enlarged cutaways reuse the selected bay’s identity.
- **Execution.** Illustrate one sequence decoding one token at a time. Preset token IDs, expert choices, attention weights and register marks are deterministic teaching examples. Condense Mamba’s input split, input-dependent coefficients and discretization into the update joint. Condense router selection bias, normalization and scaling into the routing and mix joints. Omit prefill scans, fused kernels, hardware placement, training and vocabulary sampling.
- **Memory.** For one sequence, the fixed baseline is 23 × [(64 × 64 × 128 × 4 bytes of FP32 SSM state) + (6,144 × 4 × 2 bytes of BF16 convolution state)] = **47.078125 MiB**, displayed as **47.1 MiB**. KV is a separate BF16 or FP8 payload proportional to cached token count. Both are logical tensor sizes under these dtype assumptions. Weights, quantization scales, allocator padding, batching and temporary attention buffers are outside the readout.
- **Checkpoint checks before coding.** Pin one BF16 revision for both config and model implementation. Derive bays from `hybrid_override_pattern` and assert the counts and one-based indices above. Check `hidden_size`, `vocab_size` (131,072), and `tie_word_embeddings` (false) for the bookends. For attention, read `head_dim` directly (128). Q projects to 4,096 and K and V each project to 256. Confirm RMSNorm placement and NoPE against the forward path. [Config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json) · [Reference implementation](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py)
- **Cutaway checks.** Mamba inner width comes from `mamba_num_heads × mamba_head_dim` (64 × 64). With eight state groups and state size 128, the convolution cache per sequence per M layer is **6,144 channels × 4 slots**. The SSM cache has **64 × 64 × 128** entries and its configured dtype is float32. For MoE, check routed/shared counts, both intermediate widths, `num_experts_per_tok`, and `mlp_hidden_act`. The reference router uses corrected sigmoid scores for selection and gathers the original selected scores for normalization and scaling by 2.5. [Config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json) · [Mixer and router](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py)
- **Static delivery check.** GitHub Pages serves the scene, labels and synthetic examples as local static assets under the repository base path. Opening the published project URL must support all six beats, token scrubbing and direct layer selection with model downloads and inference services absent.

### Rendering approach and project structure

Use **Vite + TypeScript with a small, purpose-built orthographic renderer in pure WebGL2**. The spine, cutaways and ruler share one instanced unit cuboid. Its instance data carries transform, face tone and selection state. A flat shader assigns three restrained paper-to-graphite face tones. Draw visible edges as depth-tested, antialiased screen-space strips about one CSS pixel wide. This keeps the line weight steady through zoom and avoids diagonal seams across slab faces. Reserve oxide red for the active selection or flow.

Use one camera matrix for drawing, projecting DOM label anchors and constructing a pointer ray. Pick the nearest intersected bay box on the CPU. Keep the selected bay ID stable through zoom and resize. Render on input, resize or animation ticks. Expose Vite’s build-time base path, with `/` for a root site and `/<repo>/` for project Pages, and publish the built `dist/` directory. [Vite’s GitHub Pages guide](https://vite.dev/guide/static-deploy.html#github-pages)

| Proposed path | Responsibility |
| --- | --- |
| `index.html`, `vite.config.ts`, `src/main.ts` | Static entry, configurable base, and wiring for selection, token count and camera state. |
| `src/model/` | Pinned architecture facts, layer pattern, derived bay indices and logical memory arithmetic. |
| `src/render/` | Orthographic camera, instanced solids and hairlines, shader sources, projection and hit testing. |
| `src/scene/` | Spine layout, ruler and cache extents. Add one layout per block family under `cutaways/` in Milestone 2. |
| `src/ui/` | Projected callouts, collision rules, accessible controls, readout and shared palette. |
| `src/story/` | Six beat definitions with camera targets, synthetic events, timing and copy. |
| `.github/workflows/pages.yml` | Build and publish static assets using the configured Pages base. |

### Build milestones

1. **Get the overview on screen.** Frame all 52 bays on warm paper with both bookends, the residual strip and six countable cache forks. Implement hover, click/tap selection, zoom, token scrubbing and the readout. **Exit check:** at desktop and narrow widths, eight static numbers remain distinct, bay 42 identifies correctly on inspection, and scrubbing 8 → 9 adds depth at exactly six forks while the Mamba value holds at 47.1 MiB. Review this working scene before starting cutaways.
2. **Build the three cutaways.** Reuse the overview’s primitives, selection and residual path. **Exit check:** step a token through 01, 02 and 06. The register updates within fixed bounds, six routed experts plus the shared branch activate, and Q16/Q17 resolve to opposite KV pairs. Returning to the spine preserves the selected bay and token position.
3. **Choreograph the six beats.** Bind each storyboard row to the working scenes. **Exit check:** Previous, Next, pause and replay reproduce the intended camera, labels and synthetic state. Reduced motion lands directly at each focus.
4. **Verify the Pages build.** Preview the production output at both root and repository base paths. **Exit check:** direct load and refresh resolve all assets, the ruler and picking stay aligned after resize, and the complete walkthrough runs from static files.
