#!/usr/bin/env node
/**
 * Automated 4-Way Nano / TwoTower / Diffusion / Uni Bundle-Isolation & Payload-Budget Gate
 * (`scripts/verify_bundle_isolation.mjs`)
 *
 * Enforces all 7 hard gates over a built 4-entry Vite MPA output directory (default `dist`,
 * or `--dir <path>`):
 *   1. All 4 entry HTML files exist: `index.html` (Nano), `twotower/index.html` (TwoTower),
 *      `diffusion/index.html` (Diffusion), and `uni/index.html` (Uni).
 *   2. Parse all `<script src>`, `<link rel="modulepreload" href>`, and `<link rel="stylesheet" href>`
 *      references from all 4 entry HTML files under root (`/`) or subpath (`/llm-viz/`) base paths
 *      and verify base-path consistency.
 *   3. Recursively walk every static ES `import` / `export ... from` edge (plus literal dynamic
 *      `import(...)` edges) inside every JS chunk reachable from all 4 entries via TypeScript AST.
 *   4. Strict 4-Way Source-Map Mutual Isolation Gate + D4 Controls Gate: read `.map` `sources` for
 *      EVERY JS chunk reachable from each entry and FAIL if:
 *        - Any of the 12 directional pairs `(fromEntry, targetEntry)` (`from !== target`) leaks
 *          target-exclusive sources (`NANO_ONLY_SOURCE_RE`, `TWOTOWER_SOURCE_RE`,
 *          `DIFFUSION_SOURCE_RE`, `UNI_SOURCE_RE`), OR
 *        - Uni `.map` sources reference `src/ui/controls.ts` (`CONTROLS_SOURCE_RE`, Locked Decision D4).
 *   5. Strict 4-Way Payload Marker Mutual Isolation Gate: scan every JS/CSS file in each entry's
 *      transitive graph across all 12 directional pairs (`from !== target`) and FAIL if any entry's
 *      JS/CSS contains forbidden markers belonging to another entry (`FORBIDDEN_NANO_MARKERS`,
 *      `FORBIDDEN_TWOTOWER_MARKERS`, `FORBIDDEN_DIFFUSION_MARKERS`, `FORBIDDEN_UNI_MARKERS`).
 *   6. Positive Presence Gate: confirm Nano sourcemaps include Nano-exclusive sources (`src/main.ts`),
 *      TwoTower sourcemaps include `src/twotower/` sources, Diffusion sourcemaps include
 *      `src/diffusion/` sources, AND Uni sourcemaps include `src/uni/` sources.
 *   7. 4-Way Payload Budget Gate: compute exact raw & gzip (level 9) byte totals for Nano,
 *      TwoTower, Diffusion, Uni, and shared chunks; FAIL if Nano transitive JS+CSS gzip exceeds
 *      `215 * 1024` bytes (`215.00 KiB` / `220,160 B`).
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

const ENTRIES = Object.freeze([
  {
    key: 'nano',
    label: 'Nano',
    shortCode: 'n',
    camelKey: 'nano',
    pascalKey: 'Nano',
    htmlRelPath: 'index.html',
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
  if (memberships.length === ENTRIES.length) return 'shared(all4)';
  if (memberships.length >= 2) {
    return `shared(${memberships.map((e) => e.shortCode).join('+')})`;
  }
  return `${memberships[0].key}-only`;
}

function main() {
  const { outDir, outDirInput } = parseCliArgs(process.argv.slice(2));
  const repoRoot = process.cwd();
  const violations = [];

  // Gate 1: All 4 entry HTML files must exist.
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

  // Gates 2 & 3: Parse HTML refs and recursively walk static ES import/export graphs for all 4 entries.
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

  // Gate 4: Strict 4-Way Source-Map Mutual Isolation Gate (all 12 directional pairs) + D4 Controls Gate.
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

  // Locked Decision D4: Uni's transitive JS sourcemaps must contain zero src/ui/controls.ts references.
  const uniControlsSourcemapLeaks = scanEntrySourcemaps(
    graphs.uni.jsChunks,
    CONTROLS_SOURCE_RE,
    outDir,
    repoRoot,
    sourcemapCache,
  );
  for (const leak of uniControlsSourcemapLeaks) {
    violations.push(
      `Gate 4 [D4 Controls Leak]: Uni chunk "${leak.chunk}" includes forbidden Nano controls source "${leak.source}"`,
    );
  }

  // Gate 5: Strict 4-Way Payload Marker Mutual Isolation Gate (all 12 directional pairs).
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

  // Gate 6: Positive Presence Gate across all 4 entries (proves sourcemaps are real and audited).
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

  // Gate 7: Exact Raw & Gzip (level 9) Payload Budget Accounting across all 4 entries.
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

  const report = {
    status: violations.length === 0 ? 'PASS' : 'FAIL',
    outDir: toPosix(relative(repoRoot, outDir)) || outDirInput,
    detectedBasePath: [...detectedBasePrefixes][0] ?? '/',
    gates: {
      entriesExist: true,
      nanoTransitiveJsCount: graphs.nano.jsChunks.length,
      nanoTransitiveCssCount: graphs.nano.cssAssets.length,
      nanoImportEdgeCount: graphs.nano.edges.length,
      twoTowerTransitiveJsCount: graphs.twotower.jsChunks.length,
      twoTowerTransitiveCssCount: graphs.twotower.cssAssets.length,
      twoTowerImportEdgeCount: graphs.twotower.edges.length,
      diffusionTransitiveJsCount: graphs.diffusion.jsChunks.length,
      diffusionTransitiveCssCount: graphs.diffusion.cssAssets.length,
      diffusionImportEdgeCount: graphs.diffusion.edges.length,
      uniTransitiveJsCount: graphs.uni.jsChunks.length,
      uniTransitiveCssCount: graphs.uni.cssAssets.length,
      uniImportEdgeCount: graphs.uni.edges.length,
      totalSourcemapLeaks,
      nanoSourcemapLeaks: entrySourcemapLeakTotals.nano,
      twoTowerSourcemapLeaks: entrySourcemapLeakTotals.twotower,
      diffusionSourcemapLeaks: entrySourcemapLeakTotals.diffusion,
      uniSourcemapLeaks: entrySourcemapLeakTotals.uni,
      uniControlsSourcemapLeaks: uniControlsSourcemapLeaks.length,
      sourcemapLeaksByPair,
      ...sourcemapLeaksFlat,
      totalForbiddenMarkerHits,
      nanoForbiddenMarkerHits: entryMarkerHitTotals.nano,
      twoTowerForbiddenMarkerHits: entryMarkerHitTotals.twotower,
      diffusionForbiddenMarkerHits: entryMarkerHitTotals.diffusion,
      uniForbiddenMarkerHits: entryMarkerHitTotals.uni,
      markerHitsByPair,
      ...markerHitsFlat,
      nanoPositiveSourceCount: confirmedSources.nano.length,
      twoTowerPositiveSourceCount: confirmedSources.twotower.length,
      diffusionPositiveSourceCount: confirmedSources.diffusion.length,
      uniPositiveSourceCount: confirmedSources.uni.length,
      nanoGzipBudgetBytes: NANO_GZIP_BUDGET_BYTES,
      nanoGzipBytes: nanoTotals.gzipBytes,
      nanoGzipHeadroomBytes,
      nanoGzipWithinBudget: nanoTotals.gzipBytes <= NANO_GZIP_BUDGET_BYTES,
    },
    totals: {
      nanoTransitiveAssets: entryTotals.nano,
      nanoOnlyAssets: entryOnlyTotals.nano,
      twoTowerTransitiveAssets: entryTotals.twotower,
      twoTowerOnlyAssets: entryOnlyTotals.twotower,
      diffusionTransitiveAssets: entryTotals.diffusion,
      diffusionOnlyAssets: entryOnlyTotals.diffusion,
      uniTransitiveAssets: entryTotals.uni,
      uniOnlyAssets: entryOnlyTotals.uni,
      sharedAssets: sharedTotals,
      htmlEntries: htmlEntrySizes,
    },
    chunks: {
      nano: entryAssetsSized.nano,
      twotower: entryAssetsSized.twotower,
      diffusion: entryAssetsSized.diffusion,
      uni: entryAssetsSized.uni,
      shared: sharedAssetsSized,
      all: allAssetsTable,
    },
    nanoConfirmedSources: confirmedSources.nano.map((s) => s.source),
    twoTowerConfirmedSources: confirmedSources.twotower.map((s) => s.source),
    diffusionConfirmedSources: confirmedSources.diffusion.map((s) => s.source),
    uniConfirmedSources: confirmedSources.uni.map((s) => s.source),
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

function printHumanSummary(report) {
  const { gates, totals, chunks } = report;
  const headroomSign = gates.nanoGzipHeadroomBytes >= 0 ? '+' : '-';
  const absHeadroom = Math.abs(gates.nanoGzipHeadroomBytes);
  console.log('=== 4-Way Nano / TwoTower / Diffusion / Uni Bundle Isolation Audit ===');
  console.log(`Output dir        : ${report.outDir}`);
  console.log(`Detected base     : ${report.detectedBasePath}`);
  console.log(
    `Nano graph        : ${gates.nanoTransitiveJsCount} JS, ${gates.nanoTransitiveCssCount} CSS | raw ${totals.nanoTransitiveAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.nanoTransitiveAssets.rawBytes)}) | gzip ${totals.nanoTransitiveAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.nanoTransitiveAssets.gzipBytes)}) / budget ${fmtKiB(gates.nanoGzipBudgetBytes)} (headroom ${headroomSign}${absHeadroom.toLocaleString('en-US')} B / ${headroomSign}${fmtKiB(absHeadroom)})`,
  );
  console.log(
    `TwoTower graph    : ${gates.twoTowerTransitiveJsCount} JS, ${gates.twoTowerTransitiveCssCount} CSS | raw ${totals.twoTowerTransitiveAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.twoTowerTransitiveAssets.rawBytes)}) | gzip ${totals.twoTowerTransitiveAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.twoTowerTransitiveAssets.gzipBytes)})`,
  );
  console.log(
    `Diffusion graph   : ${gates.diffusionTransitiveJsCount} JS, ${gates.diffusionTransitiveCssCount} CSS | raw ${totals.diffusionTransitiveAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.diffusionTransitiveAssets.rawBytes)}) | gzip ${totals.diffusionTransitiveAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.diffusionTransitiveAssets.gzipBytes)})`,
  );
  console.log(
    `Uni graph         : ${gates.uniTransitiveJsCount} JS, ${gates.uniTransitiveCssCount} CSS | raw ${totals.uniTransitiveAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.uniTransitiveAssets.rawBytes)}) | gzip ${totals.uniTransitiveAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.uniTransitiveAssets.gzipBytes)})`,
  );
  console.log(
    `Shared chunks     : ${chunks.shared.length} file(s) | raw ${totals.sharedAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.rawBytes)}) | gzip ${totals.sharedAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.gzipBytes)})`,
  );
  console.log(
    `Source-map leaks  : total=${gates.totalSourcemapLeaks} | nano(${formatPairBreakdown(gates.sourcemapLeaksByPair, ENTRIES[0])}), tt(${formatPairBreakdown(gates.sourcemapLeaksByPair, ENTRIES[1])}), df(${formatPairBreakdown(gates.sourcemapLeaksByPair, ENTRIES[2])}), uni(${formatPairBreakdown(gates.sourcemapLeaksByPair, ENTRIES[3])}), d4:uni<-controls=${gates.uniControlsSourcemapLeaks}`,
  );
  console.log(
    `Marker leaks      : total=${gates.totalForbiddenMarkerHits} | nano(${formatPairBreakdown(gates.markerHitsByPair, ENTRIES[0])}), tt(${formatPairBreakdown(gates.markerHitsByPair, ENTRIES[1])}), df(${formatPairBreakdown(gates.markerHitsByPair, ENTRIES[2])}), uni(${formatPairBreakdown(gates.markerHitsByPair, ENTRIES[3])})`,
  );
  console.log(
    `Positive presence : nanoSources=${gates.nanoPositiveSourceCount}, twoTowerSources=${gates.twoTowerPositiveSourceCount}, diffusionSources=${gates.diffusionPositiveSourceCount}, uniSources=${gates.uniPositiveSourceCount}`,
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
