#!/usr/bin/env node
/**
 * Automated N-Way Bundle-Isolation, Architectural-Boundary & Payload-Budget Gate
 * (`scripts/verify_bundle_isolation.mjs`)
 *
 * Scales dynamically with `ENTRIES` (currently 4: Nano, TwoTower, Diffusion, Uni; ready for 5–6+
 * entries by appending to `ENTRIES`). Enforces all 7 hard gates over a built Vite MPA output
 * directory (default `dist`, or `--dir <path>`):
 *   1. All entry HTML files exist.
 *   2. Parse all `<script src>`, `<link rel="modulepreload" href>`, and `<link rel="stylesheet" href>`
 *      references from all entry HTML files under root (`/`) or subpath (`/llm-viz/`) base paths
 *      and verify base-path consistency.
 *   3. Recursively walk every static ES `import` / `export ... from` edge (plus literal dynamic
 *      `import(...)` edges) inside every JS chunk reachable from all entries via TypeScript AST.
 *   4. Strict N-Way Source-Map Mutual Isolation & Architectural Boundary Gate: read `.map` `sources`
 *      for EVERY JS chunk reachable from each entry and FAIL if:
 *        - Any directional pair `(fromEntry, targetEntry)` (`from !== target`) leaks target-exclusive
 *          sources, OR
 *        - Any non-Nano entry (`entry.key !== 'nano'`) references Nano UI/controller/walkthrough
 *          modules (`src/main.ts`, `src/ui/{controls,overlay,memoryChart}.ts`,
 *          `src/story/walkthrough.ts`, `src/model/rcPlayback.ts`,
 *          `src/scene/{gqaCutaway,mambaCutaway}.ts`), OR
 *        - Any non-hybrid entry (`usesNanoHybridMath === false`) references Nano/TwoTower hybrid
 *          spine, math, or cutaway modules.
 *   5. Strict N-Way Payload Marker Mutual Isolation Gate across all directional pairs.
 *   6. Positive Presence Gate: confirm every entry's transitive JS sourcemaps include its own
 *      exclusive sources.
 *   7. N-Way Payload Budget Gate: compute exact raw & gzip (level 9) byte totals for every entry
 *      and shared chunks; FAIL if Nano transitive JS+CSS gzip exceeds `215 * 1024` bytes
 *      (`215.00 KiB` / `220,160 B`).
 *
 * Uses Node.js built-ins (`node:fs`, `node:path`, `node:zlib`) plus the devDependency TypeScript
 * parser (`typescript`) for exact ECMAScript AST import/export edge discovery across minified chunks.
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, extname, posix, relative, resolve, sep } from 'node:path';
import { gzipSync } from 'node:zlib';
import ts from 'typescript';

const NANO_GZIP_BUDGET_BYTES = 215 * 1024; // 215.00 KiB = 220,160 B

const FORBIDDEN_NANO_MARKERS = Object.freeze([
  '3D WebGL2 Canvas Unavailable',
  'Required stage DOM elements are missing',
  'Mamba-2 fixed floor',
  'mem-drawer-card',
  'memory-probe-stats',
]);

const FORBIDDEN_TWOTOWER_MARKERS = Object.freeze([
  'NemotronHTwoTowerForCausalLM',
  'twotower-denoiser-v1',
  'twotower-context-v1',
  'TwoTowerSession',
  'diffusion_block_commit',
  'TWOTOWER_SPEC',
]);

const FORBIDDEN_DIFFUSION_MARKERS = Object.freeze([
  'DIFFUSION_MINI_CONFIG',
  'DiffusionSession',
  'buildDiffusionSpatialScene',
  'DIFFUSION_DISCREPANCY_LEDGER',
  'linear_spec_lora',
  'DIFFUSION_DYADIC_YARN_INV_FREQ',
]);

const FORBIDDEN_UNI_MARKERS = Object.freeze([
  'UNI_MINI_CONFIG',
  'UniSession',
  'buildUniSpatialScene',
  'UNI_DISCREPANCY_LEDGER',
  'PackedLinearImageCodec',
  'UNI_DYADIC_YARN_INV_FREQ',
  'buildUniOpdExplorerScene',
]);

const NANO_ONLY_SOURCE_RE =
  /(?:^|\/)src\/(?:main\.ts|ui\/memoryChart\.ts)$|^(?:main\.ts|ui\/memoryChart\.ts)$/i;
const TWOTOWER_SOURCE_RE = /(?:^|\/)(?:src\/)?twotower\//i;
const DIFFUSION_SOURCE_RE = /(?:^|\/)(?:src\/)?diffusion\//i;
const UNI_SOURCE_RE = /(?:^|\/)(?:src\/)?uni\//i;
const CONTROLS_SOURCE_RE = /(?:^|\/)(?:src\/)?ui\/controls(?:\.ts)?$/i;

// Gate 4b: Nano UI, controller, walkthrough, and exclusive cutaway modules forbidden in EVERY non-Nano entry.
const NANO_UI_EXCLUSIVE_SOURCE_RE =
  /^(?:src\/)?(?:main\.ts|ui\/(?:controls|overlay|memoryChart)\.ts|story\/walkthrough\.ts|model\/rcPlayback\.ts|scene\/(?:gqaCutaway|mambaCutaway)\.ts)$/i;

// Gate 4c: Nano/TwoTower 52L hybrid core modules forbidden in EVERY non-hybrid entry (Diffusion, Uni, Models #5/#6).
const NANO_HYBRID_CORE_SOURCE_RE =
  /^(?:src\/)?(?:scene\/(?:spine|hybridPick|embeddingCutaway|outputCutaway|moeCutaway|gqaSpatial|mambaSpatial|moeSpatial|accumulation)\.ts|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode)\.ts|model\/(?:spec|memory|state)\.ts)$/i;

const ENTRIES = Object.freeze([
  {
    key: 'nano',
    label: 'Nano',
    shortCode: 'n',
    camelKey: 'nano',
    pascalKey: 'Nano',
    htmlRelPath: 'index.html',
    usesNanoHybridMath: true,
    sourceRe: NANO_ONLY_SOURCE_RE,
    sourceDesc: 'Nano-exclusive ("src/main.ts", "src/ui/memoryChart.ts")',
    markers: FORBIDDEN_NANO_MARKERS,
  },
  {
    key: 'twotower',
    label: 'TwoTower',
    shortCode: 'tt',
    camelKey: 'twoTower',
    pascalKey: 'TwoTower',
    htmlRelPath: 'twotower/index.html',
    usesNanoHybridMath: true,
    sourceRe: TWOTOWER_SOURCE_RE,
    sourceDesc: '"src/twotower/"',
    markers: FORBIDDEN_TWOTOWER_MARKERS,
  },
  {
    key: 'diffusion',
    label: 'Diffusion',
    shortCode: 'df',
    camelKey: 'diffusion',
    pascalKey: 'Diffusion',
    htmlRelPath: 'diffusion/index.html',
    usesNanoHybridMath: false,
    sourceRe: DIFFUSION_SOURCE_RE,
    sourceDesc: '"src/diffusion/"',
    markers: FORBIDDEN_DIFFUSION_MARKERS,
  },
  {
    key: 'uni',
    label: 'Uni',
    shortCode: 'uni',
    camelKey: 'uni',
    pascalKey: 'Uni',
    htmlRelPath: 'uni/index.html',
    usesNanoHybridMath: false,
    sourceRe: UNI_SOURCE_RE,
    sourceDesc: '"src/uni/"',
    markers: FORBIDDEN_UNI_MARKERS,
  },
]);

function toPosix(p) {
  return p.split(sep).join('/');
}

function parseCliArgs(argv) {
  let dir = 'dist';
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dir') {
      const next = argv[i + 1];
      if (!next || next.startsWith('-')) {
        throw new Error('Missing value for --dir');
      }
      dir = next;
      i++;
    } else if (arg.startsWith('--dir=')) {
      dir = arg.slice('--dir='.length);
      if (!dir) {
        throw new Error('Empty value for --dir=');
      }
    } else if (arg === '--help' || arg === '-h') {
      console.log('Usage: node scripts/verify_bundle_isolation.mjs [--dir <dist-path>]');
      process.exit(0);
    } else {
      throw new Error(`Unknown CLI argument: ${arg}`);
    }
  }
  return { outDir: resolve(process.cwd(), dir), outDirInput: dir };
}

function stripHtmlComments(html) {
  return html.replace(/<!--[\s\S]*?-->/g, '');
}

function parseTagAttributes(tagText) {
  const attrs = Object.create(null);
  const attrRe = /([a-zA-Z_:][a-zA-Z0-9_.:-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  const firstSpace = tagText.search(/\s/);
  if (firstSpace === -1) return attrs;
  const body = tagText.slice(firstSpace).replace(/\/?\s*>$/, '');
  let m;
  while ((m = attrRe.exec(body)) !== null) {
    const name = m[1].toLowerCase();
    const val = m[2] ?? m[3] ?? m[4] ?? '';
    attrs[name] = val;
  }
  return attrs;
}

function extractHtmlAssetRefs(html) {
  const cleaned = stripHtmlComments(html);
  const refs = [];

  const scriptRe = /<script\b[^>]*>/gi;
  let m;
  while ((m = scriptRe.exec(cleaned)) !== null) {
    const attrs = parseTagAttributes(m[0]);
    if (attrs.src) {
      refs.push({ kind: 'script', rawRef: attrs.src });
    }
  }

  const linkRe = /<link\b[^>]*>/gi;
  while ((m = linkRe.exec(cleaned)) !== null) {
    const attrs = parseTagAttributes(m[0]);
    const relTokens = (attrs.rel ?? '')
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    if (!attrs.href) continue;
    if (relTokens.includes('modulepreload')) {
      refs.push({ kind: 'modulepreload', rawRef: attrs.href });
    } else if (relTokens.includes('stylesheet')) {
      refs.push({ kind: 'stylesheet', rawRef: attrs.href });
    }
  }

  return refs;
}

function isExternalRef(rawRef) {
  return /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(rawRef.trim());
}

function stripQueryAndHash(rawRef) {
  return rawRef.split('#')[0].split('?')[0].trim();
}

function isInsideDir(targetPath, baseDir) {
  const rel = relative(baseDir, targetPath);
  return rel !== '' && !rel.startsWith('..') && !rel.startsWith(`..${sep}`);
}

/**
 * Resolve an HTML asset URL (`src` or `href`) to an absolute file path inside `outDir`,
 * supporting both root base (`/`) and subpath bases (e.g. `/llm-viz/`).
 */
function resolveHtmlAssetPath(rawRef, htmlFilePath, outDir, detectedBasePrefixes) {
  const clean = stripQueryAndHash(rawRef);
  if (!clean || isExternalRef(clean)) {
    return null;
  }

  let candidate = null;
  let basePrefix = null;

  if (clean.startsWith('/')) {
    const segments = clean.split('/').filter(Boolean);
    for (let drop = 0; drop < segments.length; drop++) {
      const subPath = segments.slice(drop).join('/');
      const testPath = resolve(outDir, subPath);
      if (isInsideDir(testPath, outDir) && existsSync(testPath)) {
        candidate = testPath;
        basePrefix = drop === 0 ? '/' : `/${segments.slice(0, drop).join('/')}/`;
        break;
      }
    }
  } else {
    const testPath = resolve(dirname(htmlFilePath), clean);
    if (isInsideDir(testPath, outDir) && existsSync(testPath)) {
      candidate = testPath;
      basePrefix = './';
    }
  }

  if (!candidate) {
    throw new Error(
      `Referenced asset "${rawRef}" from "${toPosix(relative(outDir, htmlFilePath))}" could not be resolved inside "${outDir}"`,
    );
  }

  if (basePrefix && basePrefix !== './') {
    detectedBasePrefixes.add(basePrefix);
  }
  return candidate;
}

/**
 * Extract every static ES `import` / `export ... from` module specifier (plus string-literal
 * dynamic `import("...")` specifiers) via TypeScript's ECMAScript AST parser, guaranteeing
 * exact parsing across minified chunks with regex literals, template strings, and comments.
 */
function extractJsImportSpecifiers(jsCode, fileName) {
  const sourceFile = ts.createSourceFile(
    fileName,
    jsCode,
    ts.ScriptTarget.Latest,
    false,
    ts.ScriptKind.JS,
  );
  const specifiers = new Set();

  const visit = (node) => {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
      specifiers.add(node.moduleSpecifier.text);
    } else if (
      ts.isExportDeclaration(node) &&
      node.moduleSpecifier &&
      ts.isStringLiteral(node.moduleSpecifier)
    ) {
      specifiers.add(node.moduleSpecifier.text);
    } else if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword &&
      node.arguments.length >= 1 &&
      ts.isStringLiteralLike(node.arguments[0])
    ) {
      specifiers.add(node.arguments[0].text);
    }
    ts.forEachChild(node, visit);
  };

  visit(sourceFile);
  return [...specifiers];
}

function isJsFile(filePath) {
  const ext = extname(filePath).toLowerCase();
  return ext === '.js' || ext === '.mjs' || ext === '.cjs';
}

function isCssFile(filePath) {
  return extname(filePath).toLowerCase() === '.css';
}

function resolveChunkImportPath(specifier, importerFilePath, outDir, detectedBasePrefixes) {
  const clean = stripQueryAndHash(specifier);
  if (!clean || isExternalRef(clean)) return null;

  if (clean.startsWith('./') || clean.startsWith('../')) {
    const resolved = resolve(dirname(importerFilePath), clean);
    if (!isInsideDir(resolved, outDir) || !existsSync(resolved)) {
      throw new Error(
        `Broken JS import "${specifier}" from "${toPosix(relative(outDir, importerFilePath))}"`,
      );
    }
    return resolved;
  }

  if (clean.startsWith('/')) {
    return resolveHtmlAssetPath(clean, importerFilePath, outDir, detectedBasePrefixes);
  }

  return null;
}

function walkEntryGraph(entryName, htmlFilePath, outDir, detectedBasePrefixes) {
  const html = readFileSync(htmlFilePath, 'utf8');
  const htmlRefs = extractHtmlAssetRefs(html);
  const directAssets = new Set();
  const jsQueue = [];
  const visitedJs = new Set();
  const cssAssets = new Set();
  const edges = [];

  for (const ref of htmlRefs) {
    const resolved = resolveHtmlAssetPath(ref.rawRef, htmlFilePath, outDir, detectedBasePrefixes);
    if (!resolved) continue;
    directAssets.add(resolved);
    if (isJsFile(resolved)) {
      if (!visitedJs.has(resolved)) {
        visitedJs.add(resolved);
        jsQueue.push(resolved);
      }
    } else if (isCssFile(resolved)) {
      cssAssets.add(resolved);
    }
  }

  if (visitedJs.size === 0) {
    throw new Error(
      `Entry "${entryName}" (${toPosix(relative(outDir, htmlFilePath))}) referenced 0 JS modules`,
    );
  }

  while (jsQueue.length > 0) {
    const currentJs = jsQueue.shift();
    const code = readFileSync(currentJs, 'utf8');
    const specifiers = extractJsImportSpecifiers(code, currentJs);
    for (const spec of specifiers) {
      const target = resolveChunkImportPath(spec, currentJs, outDir, detectedBasePrefixes);
      if (!target) continue;
      edges.push({
        from: toPosix(relative(outDir, currentJs)),
        to: toPosix(relative(outDir, target)),
        specifier: spec,
      });
      if (isJsFile(target)) {
        if (!visitedJs.has(target)) {
          visitedJs.add(target);
          jsQueue.push(target);
        }
      } else if (isCssFile(target)) {
        cssAssets.add(target);
      }
    }
  }

  const allAssets = new Set([...visitedJs, ...cssAssets]);
  return {
    entryName,
    htmlRelPath: toPosix(relative(outDir, htmlFilePath)),
    directAssets: [...directAssets].sort(),
    jsChunks: [...visitedJs].sort(),
    cssAssets: [...cssAssets].sort(),
    allAssets: [...allAssets].sort(),
    edges,
  };
}

function resolveSourcemapPath(jsFilePath, jsCode) {
  const matches = [...jsCode.matchAll(/\/\/[#@]\s*sourceMappingURL\s*=\s*(\S+)/g)];
  if (matches.length > 0) {
    const lastUrl = matches[matches.length - 1][1].trim();
    if (!lastUrl.startsWith('data:')) {
      return resolve(dirname(jsFilePath), stripQueryAndHash(lastUrl));
    }
  }
  return `${jsFilePath}.map`;
}

function normalizeSourcemapSource(rawSource, mapFilePath, repoRoot) {
  const cleaned = rawSource.replace(/\\/g, '/').replace(/^\0/, '');
  if (cleaned.startsWith('./') || cleaned.startsWith('../') || cleaned.startsWith('/')) {
    const absPath = cleaned.startsWith('/')
      ? cleaned
      : resolve(dirname(mapFilePath), cleaned);
    const relFromRepo = toPosix(relative(repoRoot, absPath));
    if (!relFromRepo.startsWith('..')) {
      return relFromRepo;
    }
    return posix.normalize(cleaned);
  }
  return posix.normalize(cleaned);
}

function inspectChunkSourcemap(jsFilePath, outDir, repoRoot, cache) {
  if (cache.has(jsFilePath)) {
    return cache.get(jsFilePath);
  }
  const jsCode = readFileSync(jsFilePath, 'utf8');
  const mapPath = resolveSourcemapPath(jsFilePath, jsCode);
  const relJs = toPosix(relative(outDir, jsFilePath));
  if (!existsSync(mapPath)) {
    throw new Error(`Missing sourcemap for chunk "${relJs}" (expected "${mapPath}")`);
  }
  const parsed = JSON.parse(readFileSync(mapPath, 'utf8'));
  if (!Array.isArray(parsed.sources) || parsed.sources.length === 0) {
    throw new Error(`Sourcemap for chunk "${relJs}" has an empty or missing "sources" array`);
  }
  const normalizedSources = parsed.sources.map((src) =>
    normalizeSourcemapSource(String(src), mapPath, repoRoot),
  );
  const info = {
    chunk: relJs,
    mapFile: toPosix(relative(outDir, mapPath)),
    rawSources: parsed.sources.map(String),
    normalizedSources,
  };
  cache.set(jsFilePath, info);
  return info;
}

function scanEntrySourcemaps(jsChunks, sourceRe, outDir, repoRoot, sourcemapCache) {
  const matches = [];
  for (const jsPath of jsChunks) {
    const info = inspectChunkSourcemap(jsPath, outDir, repoRoot, sourcemapCache);
    for (let i = 0; i < info.normalizedSources.length; i++) {
      const norm = info.normalizedSources[i];
      if (sourceRe.test(norm)) {
        matches.push({ chunk: info.chunk, source: norm, raw: info.rawSources[i] });
      }
    }
  }
  return matches;
}

function scanEntryMarkers(allAssets, markers, outDir, contentCache) {
  const hits = [];
  for (const assetPath of allAssets) {
    let content = contentCache.get(assetPath);
    if (content === undefined) {
      content = readFileSync(assetPath, 'utf8');
      contentCache.set(assetPath, content);
    }
    const relAsset = toPosix(relative(outDir, assetPath));
    for (const marker of markers) {
      if (content.includes(marker)) {
        hits.push({ asset: relAsset, marker });
      }
    }
  }
  return hits;
}

function measureFileSizes(filePath, outDir) {
  const buf = readFileSync(filePath);
  const gz = gzipSync(buf, { level: 9 });
  return {
    file: toPosix(relative(outDir, filePath)),
    kind: isJsFile(filePath) ? 'js' : isCssFile(filePath) ? 'css' : 'other',
    rawBytes: buf.byteLength,
    gzipBytes: gz.byteLength,
  };
}

function sumSizes(records) {
  let rawBytes = 0;
  let gzipBytes = 0;
  for (const r of records) {
    rawBytes += r.rawBytes;
    gzipBytes += r.gzipBytes;
  }
  return { rawBytes, gzipBytes };
}

function fmtKiB(bytes) {
  return `${(bytes / 1024).toFixed(2)} KiB`;
}

function classifyAssetScope(memberships) {
  if (memberships.length === ENTRIES.length) return `shared(all${ENTRIES.length})`;
  if (memberships.length >= 2) {
    return `shared(${memberships.map((e) => e.shortCode).join('+')})`;
  }
  return `${memberships[0].key}-only`;
}

function main() {
  const { outDir, outDirInput } = parseCliArgs(process.argv.slice(2));
  const repoRoot = process.cwd();
  const violations = [];

  // Gate 1: All configured entry HTML files must exist.
  const entryHtmlPaths = Object.create(null);
  for (const entry of ENTRIES) {
    const htmlPath = resolve(outDir, entry.htmlRelPath);
    entryHtmlPaths[entry.key] = htmlPath;
    if (!existsSync(htmlPath)) {
      violations.push(
        `Gate 1 [Missing Entry]: ${entry.label} entry HTML not found at "${htmlPath}"`,
      );
    }
  }
  if (violations.length > 0) {
    reportAndExit(violations, null);
  }

  // Gates 2 & 3: Parse HTML refs and recursively walk static ES import/export graphs for all entries.
  const detectedBasePrefixes = new Set();
  const graphs = Object.create(null);
  for (const entry of ENTRIES) {
    graphs[entry.key] = walkEntryGraph(
      entry.key,
      entryHtmlPaths[entry.key],
      outDir,
      detectedBasePrefixes,
    );
  }

  if (detectedBasePrefixes.size > 1) {
    violations.push(
      `Gate 2 [Inconsistent Base Path]: Multiple root base prefixes detected: ${[...detectedBasePrefixes].join(', ')}`,
    );
  }

  // Gate 4a: Strict N-Way Source-Map Mutual Isolation Gate across all N*(N-1) directional pairs.
  const sourcemapCache = new Map();
  const sourcemapLeaksByPair = Object.create(null);
  const sourcemapLeaksFlat = Object.create(null);
  const entrySourcemapLeakTotals = Object.create(null);
  let totalSourcemapLeaks = 0;

  for (const fromEntry of ENTRIES) {
    let entryTotal = 0;
    for (const targetEntry of ENTRIES) {
      if (fromEntry.key === targetEntry.key) continue;
      const leaks = scanEntrySourcemaps(
        graphs[fromEntry.key].jsChunks,
        targetEntry.sourceRe,
        outDir,
        repoRoot,
        sourcemapCache,
      );
      const pairKey = `${fromEntry.key}<-${targetEntry.key}`;
      const flatKey = `${fromEntry.camelKey}${targetEntry.pascalKey}SourcemapLeaks`;
      sourcemapLeaksByPair[pairKey] = leaks.length;
      sourcemapLeaksFlat[flatKey] = leaks.length;
      entryTotal += leaks.length;
      totalSourcemapLeaks += leaks.length;
      for (const leak of leaks) {
        violations.push(
          `Gate 4 [Source-Map Leak]: ${fromEntry.label} chunk "${leak.chunk}" includes ${targetEntry.label} source "${leak.source}"`,
        );
      }
    }
    entrySourcemapLeakTotals[fromEntry.key] = entryTotal;
  }

  // Gate 4b: Every non-Nano entry must have 0 references to Nano UI/controller/walkthrough/cutaway-exclusive sources
  // (generalizes Locked Decision D4 across TwoTower, Diffusion, Uni, and future Models #5/#6).
  const nanoUiBoundaryLeaksByEntry = Object.create(null);
  let totalNanoUiBoundaryLeaks = 0;
  for (const entry of ENTRIES) {
    if (entry.key === 'nano') continue;
    const leaks = scanEntrySourcemaps(
      graphs[entry.key].jsChunks,
      NANO_UI_EXCLUSIVE_SOURCE_RE,
      outDir,
      repoRoot,
      sourcemapCache,
    );
    nanoUiBoundaryLeaksByEntry[entry.key] = leaks.length;
    totalNanoUiBoundaryLeaks += leaks.length;
    for (const leak of leaks) {
      violations.push(
        `Gate 4b [Nano UI Boundary Leak]: ${entry.label} chunk "${leak.chunk}" includes forbidden Nano UI/controller source "${leak.source}"`,
      );
    }
  }

  // Gate 4c: Every non-hybrid entry (usesNanoHybridMath === false) must have 0 references to Nano/TwoTower hybrid core modules.
  const nonHybridCoreLeaksByEntry = Object.create(null);
  let totalNonHybridCoreLeaks = 0;
  for (const entry of ENTRIES) {
    if (entry.usesNanoHybridMath) continue;
    const leaks = scanEntrySourcemaps(
      graphs[entry.key].jsChunks,
      NANO_HYBRID_CORE_SOURCE_RE,
      outDir,
      repoRoot,
      sourcemapCache,
    );
    nonHybridCoreLeaksByEntry[entry.key] = leaks.length;
    totalNonHybridCoreLeaks += leaks.length;
    for (const leak of leaks) {
      violations.push(
        `Gate 4c [Non-Hybrid Core Leak]: ${entry.label} chunk "${leak.chunk}" includes forbidden Nano/TwoTower hybrid source "${leak.source}"`,
      );
    }
  }

  // Preserved legacy D4 counter for backwards-compatible JSON reporting.
  const uniControlsSourcemapLeaks = graphs.uni
    ? scanEntrySourcemaps(
        graphs.uni.jsChunks,
        CONTROLS_SOURCE_RE,
        outDir,
        repoRoot,
        sourcemapCache,
      ).length
    : 0;

  // Gate 5: Strict N-Way Payload Marker Mutual Isolation Gate across all directional pairs.
  const contentCache = new Map();
  const markerHitsByPair = Object.create(null);
  const markerHitsFlat = Object.create(null);
  const entryMarkerHitTotals = Object.create(null);
  let totalForbiddenMarkerHits = 0;

  for (const fromEntry of ENTRIES) {
    let entryTotal = 0;
    for (const targetEntry of ENTRIES) {
      if (fromEntry.key === targetEntry.key) continue;
      const hits = scanEntryMarkers(
        graphs[fromEntry.key].allAssets,
        targetEntry.markers,
        outDir,
        contentCache,
      );
      const pairKey = `${fromEntry.key}<-${targetEntry.key}`;
      const flatKey = `${fromEntry.camelKey}${targetEntry.pascalKey}MarkerHits`;
      markerHitsByPair[pairKey] = hits.length;
      markerHitsFlat[flatKey] = hits.length;
      entryTotal += hits.length;
      totalForbiddenMarkerHits += hits.length;
      for (const hit of hits) {
        violations.push(
          `Gate 5 [Payload Marker Leak]: ${fromEntry.label} asset "${hit.asset}" contains forbidden ${targetEntry.label} marker "${hit.marker}"`,
        );
      }
    }
    entryMarkerHitTotals[fromEntry.key] = entryTotal;
  }

  // Gate 6: Positive Presence Gate across all entries (proves sourcemaps are real and audited).
  const confirmedSources = Object.create(null);
  for (const entry of ENTRIES) {
    const matches = scanEntrySourcemaps(
      graphs[entry.key].jsChunks,
      entry.sourceRe,
      outDir,
      repoRoot,
      sourcemapCache,
    );
    confirmedSources[entry.key] = matches;
    if (matches.length === 0) {
      violations.push(
        `Gate 6 [Positive Presence]: ${entry.label} transitive JS sourcemaps contain zero ${entry.sourceDesc} sources`,
      );
    }
  }

  // Gate 7: Exact Raw & Gzip (level 9) Payload Budget Accounting across all entries.
  const assetSets = Object.create(null);
  const unionSet = new Set();
  for (const entry of ENTRIES) {
    assetSets[entry.key] = new Set(graphs[entry.key].allAssets);
    for (const p of graphs[entry.key].allAssets) {
      unionSet.add(p);
    }
  }
  const unionAssetPaths = [...unionSet].sort();

  const membershipsFor = (p) => ENTRIES.filter((entry) => assetSets[entry.key].has(p));

  const sharedAssetPaths = unionAssetPaths.filter((p) => membershipsFor(p).length >= 2);
  const onlyAssetPaths = Object.create(null);
  for (const entry of ENTRIES) {
    onlyAssetPaths[entry.key] = [...assetSets[entry.key]]
      .filter((p) => membershipsFor(p).length === 1)
      .sort();
  }

  const sizeCache = new Map();
  const getSize = (p) => {
    if (!sizeCache.has(p)) {
      sizeCache.set(p, measureFileSizes(p, outDir));
    }
    return sizeCache.get(p);
  };

  const entryAssetsSized = Object.create(null);
  const entryOnlySized = Object.create(null);
  const entryTotals = Object.create(null);
  const entryOnlyTotals = Object.create(null);
  const htmlEntrySizes = Object.create(null);

  for (const entry of ENTRIES) {
    entryAssetsSized[entry.key] = graphs[entry.key].allAssets.map(getSize);
    entryOnlySized[entry.key] = onlyAssetPaths[entry.key].map(getSize);
    entryTotals[entry.key] = sumSizes(entryAssetsSized[entry.key]);
    entryOnlyTotals[entry.key] = sumSizes(entryOnlySized[entry.key]);
    htmlEntrySizes[entry.key] = measureFileSizes(entryHtmlPaths[entry.key], outDir);
  }

  const sharedAssetsSized = sharedAssetPaths.map((p) => ({
    scope: classifyAssetScope(membershipsFor(p)),
    ...getSize(p),
  }));
  const allAssetsTable = unionAssetPaths.map((p) => ({
    scope: classifyAssetScope(membershipsFor(p)),
    ...getSize(p),
  }));
  const sharedTotals = sumSizes(sharedAssetsSized);

  const nanoTotals = entryTotals.nano;
  const nanoGzipHeadroomBytes = NANO_GZIP_BUDGET_BYTES - nanoTotals.gzipBytes;

  if (nanoTotals.gzipBytes > NANO_GZIP_BUDGET_BYTES) {
    violations.push(
      `Gate 7 [Nano Payload Budget]: Nano transitive assets gzip (${nanoTotals.gzipBytes} B / ${fmtKiB(nanoTotals.gzipBytes)}) exceeds limit (${NANO_GZIP_BUDGET_BYTES} B / ${fmtKiB(NANO_GZIP_BUDGET_BYTES)})`,
    );
  }

  const perEntryGateCounts = Object.create(null);
  const perEntrySourcemapTotals = Object.create(null);
  const perEntryMarkerTotals = Object.create(null);
  const perEntryPositiveCounts = Object.create(null);
  const perEntryTotals = Object.create(null);
  const perEntryChunks = Object.create(null);
  const perEntryConfirmedSources = Object.create(null);

  for (const entry of ENTRIES) {
    perEntryGateCounts[`${entry.camelKey}TransitiveJsCount`] = graphs[entry.key].jsChunks.length;
    perEntryGateCounts[`${entry.camelKey}TransitiveCssCount`] = graphs[entry.key].cssAssets.length;
    perEntryGateCounts[`${entry.camelKey}ImportEdgeCount`] = graphs[entry.key].edges.length;
    perEntrySourcemapTotals[`${entry.camelKey}SourcemapLeaks`] = entrySourcemapLeakTotals[entry.key];
    perEntryMarkerTotals[`${entry.camelKey}ForbiddenMarkerHits`] = entryMarkerHitTotals[entry.key];
    perEntryPositiveCounts[`${entry.camelKey}PositiveSourceCount`] =
      confirmedSources[entry.key].length;
    perEntryTotals[`${entry.camelKey}TransitiveAssets`] = entryTotals[entry.key];
    perEntryTotals[`${entry.camelKey}OnlyAssets`] = entryOnlyTotals[entry.key];
    perEntryChunks[entry.key] = entryAssetsSized[entry.key];
    perEntryConfirmedSources[`${entry.camelKey}ConfirmedSources`] = confirmedSources[entry.key].map(
      (s) => s.source,
    );
  }

  const report = {
    status: violations.length === 0 ? 'PASS' : 'FAIL',
    entryCount: ENTRIES.length,
    outDir: toPosix(relative(repoRoot, outDir)) || outDirInput,
    detectedBasePath: [...detectedBasePrefixes][0] ?? '/',
    gates: {
      entriesExist: true,
      ...perEntryGateCounts,
      totalSourcemapLeaks,
      ...perEntrySourcemapTotals,
      uniControlsSourcemapLeaks,
      totalNanoUiBoundaryLeaks,
      nanoUiBoundaryLeaksByEntry,
      totalNonHybridCoreLeaks,
      nonHybridCoreLeaksByEntry,
      sourcemapLeaksByPair,
      ...sourcemapLeaksFlat,
      totalForbiddenMarkerHits,
      ...perEntryMarkerTotals,
      markerHitsByPair,
      ...markerHitsFlat,
      ...perEntryPositiveCounts,
      nanoGzipBudgetBytes: NANO_GZIP_BUDGET_BYTES,
      nanoGzipBytes: nanoTotals.gzipBytes,
      nanoGzipHeadroomBytes,
      nanoGzipWithinBudget: nanoTotals.gzipBytes <= NANO_GZIP_BUDGET_BYTES,
    },
    totals: {
      ...perEntryTotals,
      sharedAssets: sharedTotals,
      htmlEntries: htmlEntrySizes,
    },
    chunks: {
      ...perEntryChunks,
      shared: sharedAssetsSized,
      all: allAssetsTable,
    },
    ...perEntryConfirmedSources,
    violations,
  };

  printHumanSummary(report);
  console.log('\n--- JSON REPORT ---');
  console.log(JSON.stringify(report, null, 2));

  if (violations.length > 0) {
    reportAndExit(violations, report);
  }
}

function formatPairBreakdown(byPair, fromEntry) {
  return ENTRIES.filter((e) => e.key !== fromEntry.key)
    .map((target) => `${target.shortCode}=${byPair[`${fromEntry.key}<-${target.key}`]}`)
    .join(', ');
}

function formatMapBreakdown(mapObj) {
  return Object.entries(mapObj)
    .map(([k, v]) => `${k}=${v}`)
    .join(', ');
}

function printHumanSummary(report) {
  const { gates, totals, chunks } = report;
  const headroomSign = gates.nanoGzipHeadroomBytes >= 0 ? '+' : '-';
  const absHeadroom = Math.abs(gates.nanoGzipHeadroomBytes);
  console.log(
    `=== ${ENTRIES.length}-Way ${ENTRIES.map((e) => e.label).join(' / ')} Bundle Isolation Audit ===`,
  );
  console.log(`Output dir        : ${report.outDir}`);
  console.log(`Detected base     : ${report.detectedBasePath}`);
  for (const entry of ENTRIES) {
    const jsCount = gates[`${entry.camelKey}TransitiveJsCount`];
    const cssCount = gates[`${entry.camelKey}TransitiveCssCount`];
    const assetTotals = totals[`${entry.camelKey}TransitiveAssets`];
    const labelCol = `${entry.label} graph`.padEnd(18);
    const budgetSuffix =
      entry.key === 'nano'
        ? ` / budget ${fmtKiB(gates.nanoGzipBudgetBytes)} (headroom ${headroomSign}${absHeadroom.toLocaleString('en-US')} B / ${headroomSign}${fmtKiB(absHeadroom)})`
        : '';
    console.log(
      `${labelCol}: ${jsCount} JS, ${cssCount} CSS | raw ${assetTotals.rawBytes.toLocaleString('en-US')} B (${fmtKiB(assetTotals.rawBytes)}) | gzip ${assetTotals.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(assetTotals.gzipBytes)})${budgetSuffix}`,
    );
  }
  console.log(
    `Shared chunks     : ${chunks.shared.length} file(s) | raw ${totals.sharedAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.rawBytes)}) | gzip ${totals.sharedAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.gzipBytes)})`,
  );
  console.log(
    `Source-map leaks  : total=${gates.totalSourcemapLeaks} | ${ENTRIES.map((e) => `${e.shortCode}(${formatPairBreakdown(gates.sourcemapLeaksByPair, e)})`).join(', ')}`,
  );
  console.log(
    `Boundary gates    : nanoUI=${gates.totalNanoUiBoundaryLeaks} (${formatMapBreakdown(gates.nanoUiBoundaryLeaksByEntry)}) | nonHybridCore=${gates.totalNonHybridCoreLeaks} (${formatMapBreakdown(gates.nonHybridCoreLeaksByEntry)})`,
  );
  console.log(
    `Marker leaks      : total=${gates.totalForbiddenMarkerHits} | ${ENTRIES.map((e) => `${e.shortCode}(${formatPairBreakdown(gates.markerHitsByPair, e)})`).join(', ')}`,
  );
  console.log(
    `Positive presence : ${ENTRIES.map((e) => `${e.camelKey}Sources=${gates[`${e.camelKey}PositiveSourceCount`]}`).join(', ')}`,
  );

  console.log('\nScope               Kind   Raw (B)    Gzip (B)   Asset');
  console.log('------------------  -----  ---------  ---------  ----------------------------------------');
  for (const r of chunks.all) {
    const scope = r.scope.padEnd(18);
    const kind = r.kind.padEnd(5);
    const raw = String(r.rawBytes).padStart(9);
    const gz = String(r.gzipBytes).padStart(9);
    console.log(`${scope}  ${kind}  ${raw}  ${gz}  ${r.file}`);
  }
  console.log(`\nOverall status    : ${report.status}`);
}

function reportAndExit(violations, _report) {
  console.error('\nBundle isolation verification FAILED:');
  for (const v of violations) {
    console.error(`  - ${v}`);
  }
  process.exit(1);
}

try {
  main();
} catch (err) {
  console.error(
    `Bundle isolation verification ERROR: ${err instanceof Error ? err.message : String(err)}`,
  );
  process.exit(1);
}
