#!/usr/bin/env node
/**
 * Automated 3-Way Nano / TwoTower / Diffusion Bundle-Isolation & Payload-Budget Gate
 * (`scripts/verify_bundle_isolation.mjs`)
 *
 * Enforces all 7 hard gates over a built 3-entry Vite MPA output directory (default `dist`,
 * or `--dir <path>`):
 *   1. All 3 entry HTML files exist: `index.html` (Nano), `twotower/index.html` (TwoTower),
 *      and `diffusion/index.html` (Diffusion).
 *   2. Parse all `<script src>`, `<link rel="modulepreload" href>`, and `<link rel="stylesheet" href>`
 *      references from all 3 entry HTML files under root (`/`) or subpath (`/llm-viz/`) base paths
 *      and verify base-path consistency.
 *   3. Recursively walk every static ES `import` / `export ... from` edge (plus literal dynamic
 *      `import(...)` edges) inside every JS chunk reachable from all 3 entries via TypeScript AST.
 *   4. Strict 3-Way Source-Map Mutual Isolation Gate: read `.map` `sources` for EVERY JS chunk
 *      reachable from each entry and FAIL if:
 *        - Nano `.map` sources reference `src/twotower/` (`twotower/`) OR `src/diffusion/` (`diffusion/`),
 *        - TwoTower `.map` sources reference `src/diffusion/` (`diffusion/`), or
 *        - Diffusion `.map` sources reference `src/twotower/` (`twotower/`).
 *   5. Strict 3-Way Payload Marker Mutual Isolation Gate: scan every JS/CSS file in each entry's
 *      transitive graph and FAIL if:
 *        - Nano JS/CSS contains any `FORBIDDEN_TWOTOWER_MARKERS` OR `FORBIDDEN_DIFFUSION_MARKERS`,
 *        - TwoTower JS/CSS contains any `FORBIDDEN_DIFFUSION_MARKERS`, or
 *        - Diffusion JS/CSS contains any `FORBIDDEN_TWOTOWER_MARKERS`.
 *   6. Positive Presence Gate: confirm TwoTower's transitive JS sourcemaps include `src/twotower/`
 *      sources AND Diffusion's transitive JS sourcemaps include `src/diffusion/` sources.
 *   7. Nano Payload Budget Gate: compute exact raw & gzip (level 9) byte totals for Nano,
 *      TwoTower, Diffusion, and shared chunks; FAIL if Nano transitive JS+CSS gzip exceeds
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

const TWOTOWER_SOURCE_RE = /(?:^|\/)(?:src\/)?twotower\//i;
const DIFFUSION_SOURCE_RE = /(?:^|\/)(?:src\/)?diffusion\//i;

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

function classifyAssetScope(inNano, inTwoTower, inDiffusion) {
  const count = (inNano ? 1 : 0) + (inTwoTower ? 1 : 0) + (inDiffusion ? 1 : 0);
  if (count === 3) return 'shared(all3)';
  if (count === 2) {
    if (inNano && inTwoTower) return 'shared(n+tt)';
    if (inNano && inDiffusion) return 'shared(n+df)';
    return 'shared(tt+df)';
  }
  if (inNano) return 'nano-only';
  if (inTwoTower) return 'twotower-only';
  return 'diffusion-only';
}

function main() {
  const { outDir, outDirInput } = parseCliArgs(process.argv.slice(2));
  const repoRoot = process.cwd();
  const violations = [];

  // Gate 1: All 3 entry HTML files must exist.
  const nanoHtmlPath = resolve(outDir, 'index.html');
  const twoTowerHtmlPath = resolve(outDir, 'twotower/index.html');
  const diffusionHtmlPath = resolve(outDir, 'diffusion/index.html');
  if (!existsSync(nanoHtmlPath)) {
    violations.push(`Gate 1 [Missing Entry]: Nano entry HTML not found at "${nanoHtmlPath}"`);
  }
  if (!existsSync(twoTowerHtmlPath)) {
    violations.push(
      `Gate 1 [Missing Entry]: TwoTower entry HTML not found at "${twoTowerHtmlPath}"`,
    );
  }
  if (!existsSync(diffusionHtmlPath)) {
    violations.push(
      `Gate 1 [Missing Entry]: Diffusion entry HTML not found at "${diffusionHtmlPath}"`,
    );
  }
  if (violations.length > 0) {
    reportAndExit(violations, null);
  }

  // Gates 2 & 3: Parse HTML refs and recursively walk static ES import/export graphs for all 3 entries.
  const detectedBasePrefixes = new Set();
  const nanoGraph = walkEntryGraph('nano', nanoHtmlPath, outDir, detectedBasePrefixes);
  const twoTowerGraph = walkEntryGraph('twotower', twoTowerHtmlPath, outDir, detectedBasePrefixes);
  const diffusionGraph = walkEntryGraph('diffusion', diffusionHtmlPath, outDir, detectedBasePrefixes);

  if (detectedBasePrefixes.size > 1) {
    violations.push(
      `Gate 2 [Inconsistent Base Path]: Multiple root base prefixes detected: ${[...detectedBasePrefixes].join(', ')}`,
    );
  }

  // Gate 4: Strict 3-Way Source-Map Mutual Isolation Gate.
  const sourcemapCache = new Map();
  const nanoTwoTowerSourcemapLeaks = scanEntrySourcemaps(
    nanoGraph.jsChunks,
    TWOTOWER_SOURCE_RE,
    outDir,
    repoRoot,
    sourcemapCache,
  );
  const nanoDiffusionSourcemapLeaks = scanEntrySourcemaps(
    nanoGraph.jsChunks,
    DIFFUSION_SOURCE_RE,
    outDir,
    repoRoot,
    sourcemapCache,
  );
  const twoTowerDiffusionSourcemapLeaks = scanEntrySourcemaps(
    twoTowerGraph.jsChunks,
    DIFFUSION_SOURCE_RE,
    outDir,
    repoRoot,
    sourcemapCache,
  );
  const diffusionTwoTowerSourcemapLeaks = scanEntrySourcemaps(
    diffusionGraph.jsChunks,
    TWOTOWER_SOURCE_RE,
    outDir,
    repoRoot,
    sourcemapCache,
  );

  for (const leak of nanoTwoTowerSourcemapLeaks) {
    violations.push(
      `Gate 4 [Source-Map Leak]: Nano chunk "${leak.chunk}" includes TwoTower source "${leak.source}"`,
    );
  }
  for (const leak of nanoDiffusionSourcemapLeaks) {
    violations.push(
      `Gate 4 [Source-Map Leak]: Nano chunk "${leak.chunk}" includes Diffusion source "${leak.source}"`,
    );
  }
  for (const leak of twoTowerDiffusionSourcemapLeaks) {
    violations.push(
      `Gate 4 [Source-Map Leak]: TwoTower chunk "${leak.chunk}" includes Diffusion source "${leak.source}"`,
    );
  }
  for (const leak of diffusionTwoTowerSourcemapLeaks) {
    violations.push(
      `Gate 4 [Source-Map Leak]: Diffusion chunk "${leak.chunk}" includes TwoTower source "${leak.source}"`,
    );
  }

  // Gate 5: Strict 3-Way Payload Marker Mutual Isolation Gate.
  const contentCache = new Map();
  const nanoTwoTowerMarkerHits = scanEntryMarkers(
    nanoGraph.allAssets,
    FORBIDDEN_TWOTOWER_MARKERS,
    outDir,
    contentCache,
  );
  const nanoDiffusionMarkerHits = scanEntryMarkers(
    nanoGraph.allAssets,
    FORBIDDEN_DIFFUSION_MARKERS,
    outDir,
    contentCache,
  );
  const twoTowerDiffusionMarkerHits = scanEntryMarkers(
    twoTowerGraph.allAssets,
    FORBIDDEN_DIFFUSION_MARKERS,
    outDir,
    contentCache,
  );
  const diffusionTwoTowerMarkerHits = scanEntryMarkers(
    diffusionGraph.allAssets,
    FORBIDDEN_TWOTOWER_MARKERS,
    outDir,
    contentCache,
  );

  for (const hit of nanoTwoTowerMarkerHits) {
    violations.push(
      `Gate 5 [Payload Marker Leak]: Nano asset "${hit.asset}" contains forbidden TwoTower marker "${hit.marker}"`,
    );
  }
  for (const hit of nanoDiffusionMarkerHits) {
    violations.push(
      `Gate 5 [Payload Marker Leak]: Nano asset "${hit.asset}" contains forbidden Diffusion marker "${hit.marker}"`,
    );
  }
  for (const hit of twoTowerDiffusionMarkerHits) {
    violations.push(
      `Gate 5 [Payload Marker Leak]: TwoTower asset "${hit.asset}" contains forbidden Diffusion marker "${hit.marker}"`,
    );
  }
  for (const hit of diffusionTwoTowerMarkerHits) {
    violations.push(
      `Gate 5 [Payload Marker Leak]: Diffusion asset "${hit.asset}" contains forbidden TwoTower marker "${hit.marker}"`,
    );
  }

  // Gate 6: Positive Presence Gate for TwoTower and Diffusion (proves sourcemaps are real and audited).
  const twoTowerConfirmedSources = scanEntrySourcemaps(
    twoTowerGraph.jsChunks,
    TWOTOWER_SOURCE_RE,
    outDir,
    repoRoot,
    sourcemapCache,
  );
  const diffusionConfirmedSources = scanEntrySourcemaps(
    diffusionGraph.jsChunks,
    DIFFUSION_SOURCE_RE,
    outDir,
    repoRoot,
    sourcemapCache,
  );

  if (twoTowerConfirmedSources.length === 0) {
    violations.push(
      'Gate 6 [Positive Presence]: TwoTower transitive JS sourcemaps contain zero "src/twotower/" sources',
    );
  }
  if (diffusionConfirmedSources.length === 0) {
    violations.push(
      'Gate 6 [Positive Presence]: Diffusion transitive JS sourcemaps contain zero "src/diffusion/" sources',
    );
  }

  // Gate 7: Exact Raw & Gzip (level 9) Payload Budget Accounting across all 3 entries.
  const nanoAssetSet = new Set(nanoGraph.allAssets);
  const twoTowerAssetSet = new Set(twoTowerGraph.allAssets);
  const diffusionAssetSet = new Set(diffusionGraph.allAssets);
  const unionAssetPaths = [
    ...new Set([...nanoGraph.allAssets, ...twoTowerGraph.allAssets, ...diffusionGraph.allAssets]),
  ].sort();

  const sharedAssetPaths = unionAssetPaths.filter((p) => {
    const memberships =
      (nanoAssetSet.has(p) ? 1 : 0) +
      (twoTowerAssetSet.has(p) ? 1 : 0) +
      (diffusionAssetSet.has(p) ? 1 : 0);
    return memberships >= 2;
  });
  const nanoOnlyAssetPaths = [...nanoAssetSet]
    .filter((p) => !twoTowerAssetSet.has(p) && !diffusionAssetSet.has(p))
    .sort();
  const twoTowerOnlyAssetPaths = [...twoTowerAssetSet]
    .filter((p) => !nanoAssetSet.has(p) && !diffusionAssetSet.has(p))
    .sort();
  const diffusionOnlyAssetPaths = [...diffusionAssetSet]
    .filter((p) => !nanoAssetSet.has(p) && !twoTowerAssetSet.has(p))
    .sort();

  const sizeCache = new Map();
  const getSize = (p) => {
    if (!sizeCache.has(p)) {
      sizeCache.set(p, measureFileSizes(p, outDir));
    }
    return sizeCache.get(p);
  };

  const nanoAssetsSized = nanoGraph.allAssets.map(getSize);
  const twoTowerAssetsSized = twoTowerGraph.allAssets.map(getSize);
  const diffusionAssetsSized = diffusionGraph.allAssets.map(getSize);
  const sharedAssetsSized = sharedAssetPaths.map((p) => ({
    scope: classifyAssetScope(
      nanoAssetSet.has(p),
      twoTowerAssetSet.has(p),
      diffusionAssetSet.has(p),
    ),
    ...getSize(p),
  }));
  const nanoOnlySized = nanoOnlyAssetPaths.map(getSize);
  const twoTowerOnlySized = twoTowerOnlyAssetPaths.map(getSize);
  const diffusionOnlySized = diffusionOnlyAssetPaths.map(getSize);
  const allAssetsTable = unionAssetPaths.map((p) => ({
    scope: classifyAssetScope(
      nanoAssetSet.has(p),
      twoTowerAssetSet.has(p),
      diffusionAssetSet.has(p),
    ),
    ...getSize(p),
  }));

  const nanoHtmlSize = measureFileSizes(nanoHtmlPath, outDir);
  const twoTowerHtmlSize = measureFileSizes(twoTowerHtmlPath, outDir);
  const diffusionHtmlSize = measureFileSizes(diffusionHtmlPath, outDir);

  const nanoTotals = sumSizes(nanoAssetsSized);
  const twoTowerTotals = sumSizes(twoTowerAssetsSized);
  const diffusionTotals = sumSizes(diffusionAssetsSized);
  const sharedTotals = sumSizes(sharedAssetsSized);
  const nanoOnlyTotals = sumSizes(nanoOnlySized);
  const twoTowerOnlyTotals = sumSizes(twoTowerOnlySized);
  const diffusionOnlyTotals = sumSizes(diffusionOnlySized);

  const nanoGzipHeadroomBytes = NANO_GZIP_BUDGET_BYTES - nanoTotals.gzipBytes;

  if (nanoTotals.gzipBytes > NANO_GZIP_BUDGET_BYTES) {
    violations.push(
      `Gate 7 [Nano Payload Budget]: Nano transitive assets gzip (${nanoTotals.gzipBytes} B / ${fmtKiB(nanoTotals.gzipBytes)}) exceeds limit (${NANO_GZIP_BUDGET_BYTES} B / ${fmtKiB(NANO_GZIP_BUDGET_BYTES)})`,
    );
  }

  const nanoSourcemapLeaksTotal =
    nanoTwoTowerSourcemapLeaks.length + nanoDiffusionSourcemapLeaks.length;
  const nanoMarkerHitsTotal = nanoTwoTowerMarkerHits.length + nanoDiffusionMarkerHits.length;

  const report = {
    status: violations.length === 0 ? 'PASS' : 'FAIL',
    outDir: toPosix(relative(repoRoot, outDir)) || outDirInput,
    detectedBasePath: [...detectedBasePrefixes][0] ?? '/',
    gates: {
      entriesExist: true,
      nanoTransitiveJsCount: nanoGraph.jsChunks.length,
      nanoTransitiveCssCount: nanoGraph.cssAssets.length,
      nanoImportEdgeCount: nanoGraph.edges.length,
      twoTowerTransitiveJsCount: twoTowerGraph.jsChunks.length,
      twoTowerTransitiveCssCount: twoTowerGraph.cssAssets.length,
      twoTowerImportEdgeCount: twoTowerGraph.edges.length,
      diffusionTransitiveJsCount: diffusionGraph.jsChunks.length,
      diffusionTransitiveCssCount: diffusionGraph.cssAssets.length,
      diffusionImportEdgeCount: diffusionGraph.edges.length,
      nanoSourcemapLeaks: nanoSourcemapLeaksTotal,
      nanoTwoTowerSourcemapLeaks: nanoTwoTowerSourcemapLeaks.length,
      nanoDiffusionSourcemapLeaks: nanoDiffusionSourcemapLeaks.length,
      twoTowerDiffusionSourcemapLeaks: twoTowerDiffusionSourcemapLeaks.length,
      diffusionTwoTowerSourcemapLeaks: diffusionTwoTowerSourcemapLeaks.length,
      nanoForbiddenMarkerHits: nanoMarkerHitsTotal,
      nanoTwoTowerMarkerHits: nanoTwoTowerMarkerHits.length,
      nanoDiffusionMarkerHits: nanoDiffusionMarkerHits.length,
      twoTowerDiffusionMarkerHits: twoTowerDiffusionMarkerHits.length,
      diffusionTwoTowerMarkerHits: diffusionTwoTowerMarkerHits.length,
      twoTowerPositiveSourceCount: twoTowerConfirmedSources.length,
      diffusionPositiveSourceCount: diffusionConfirmedSources.length,
      nanoGzipBudgetBytes: NANO_GZIP_BUDGET_BYTES,
      nanoGzipBytes: nanoTotals.gzipBytes,
      nanoGzipHeadroomBytes,
      nanoGzipWithinBudget: nanoTotals.gzipBytes <= NANO_GZIP_BUDGET_BYTES,
    },
    totals: {
      nanoTransitiveAssets: nanoTotals,
      nanoOnlyAssets: nanoOnlyTotals,
      twoTowerTransitiveAssets: twoTowerTotals,
      twoTowerOnlyAssets: twoTowerOnlyTotals,
      diffusionTransitiveAssets: diffusionTotals,
      diffusionOnlyAssets: diffusionOnlyTotals,
      sharedAssets: sharedTotals,
      htmlEntries: {
        nano: nanoHtmlSize,
        twotower: twoTowerHtmlSize,
        diffusion: diffusionHtmlSize,
      },
    },
    chunks: {
      nano: nanoAssetsSized,
      twotower: twoTowerAssetsSized,
      diffusion: diffusionAssetsSized,
      shared: sharedAssetsSized,
      all: allAssetsTable,
    },
    twoTowerConfirmedSources: twoTowerConfirmedSources.map((s) => s.source),
    diffusionConfirmedSources: diffusionConfirmedSources.map((s) => s.source),
    violations,
  };

  printHumanSummary(report);
  console.log('\n--- JSON REPORT ---');
  console.log(JSON.stringify(report, null, 2));

  if (violations.length > 0) {
    reportAndExit(violations, report);
  }
}

function printHumanSummary(report) {
  const { gates, totals, chunks } = report;
  const headroomSign = gates.nanoGzipHeadroomBytes >= 0 ? '+' : '-';
  const absHeadroom = Math.abs(gates.nanoGzipHeadroomBytes);
  console.log('=== 3-Way Nano / TwoTower / Diffusion Bundle Isolation Audit ===');
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
    `Shared chunks     : ${chunks.shared.length} file(s) | raw ${totals.sharedAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.rawBytes)}) | gzip ${totals.sharedAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.gzipBytes)})`,
  );
  console.log(
    `Source-map leaks  : nano=${gates.nanoSourcemapLeaks} (tt=${gates.nanoTwoTowerSourcemapLeaks}, df=${gates.nanoDiffusionSourcemapLeaks}), twotower<-df=${gates.twoTowerDiffusionSourcemapLeaks}, diffusion<-tt=${gates.diffusionTwoTowerSourcemapLeaks}`,
  );
  console.log(
    `Marker leaks      : nano=${gates.nanoForbiddenMarkerHits} (tt=${gates.nanoTwoTowerMarkerHits}, df=${gates.nanoDiffusionMarkerHits}), twotower<-df=${gates.twoTowerDiffusionMarkerHits}, diffusion<-tt=${gates.diffusionTwoTowerMarkerHits}`,
  );
  console.log(
    `Positive presence : twoTowerSources=${gates.twoTowerPositiveSourceCount}, diffusionSources=${gates.diffusionPositiveSourceCount}`,
  );

  console.log('\nScope           Kind   Raw (B)    Gzip (B)   Asset');
  console.log('--------------  -----  ---------  ---------  ----------------------------------------');
  for (const r of chunks.all) {
    const scope = r.scope.padEnd(14);
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
