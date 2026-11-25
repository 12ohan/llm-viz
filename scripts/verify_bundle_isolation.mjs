#!/usr/bin/env node
/**
 * Automated Nano / TwoTower Topology A Bundle-Isolation & Payload-Budget Gate
 * (`scripts/verify_bundle_isolation.mjs`)
 *
 * Enforces all 7 hard gates over a built Vite MPA output directory (default `dist`,
 * or `--dir <path>`):
 *   1. Both `index.html` (Nano) and `twotower/index.html` (TwoTower) exist.
 *   2. Parse all `<script src>`, `<link rel="modulepreload" href>`, and `<link rel="stylesheet" href>`
 *      references from both entry HTML files under root (`/`) or subpath (`/llm-viz/`) base paths.
 *   3. Recursively walk every static ES `import` / `export ... from` edge (plus literal dynamic
 *      `import(...)` edges) inside every JS chunk reachable from each entry.
 *   4. Strict Source-Map Isolation Gate: read `.map` `sources` for EVERY JS chunk in Nano's
 *      transitive graph and FAIL if any source path references `src/twotower/` or `twotower/`.
 *   5. Strict Payload Marker Gate: scan every JS/CSS file in Nano's transitive graph and FAIL if
 *      any TwoTower-specific marker appears.
 *   6. TwoTower Positive Presence Gate: confirm TwoTower's transitive graph DOES include
 *      `src/twotower/` sources in its sourcemaps.
 *   7. Nano Payload Budget Gate: compute exact raw & gzip (level 9) byte totals for Nano,
 *      TwoTower, and shared chunks; FAIL if Nano total gzip exceeds `215 * 1024` bytes (`215 KiB`).
 *
 * Uses Node.js built-ins (`node:fs`, `node:path`, `node:zlib`) plus the devDependency TypeScript
 * parser (`typescript`) for exact ECMAScript AST import/export edge discovery across minified chunks.
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, extname, posix, relative, resolve, sep } from 'node:path';
import { gzipSync } from 'node:zlib';
import ts from 'typescript';

const NANO_GZIP_BUDGET_BYTES = 215 * 1024; // 215 KiB = 220,160 B

const FORBIDDEN_NANO_MARKERS = Object.freeze([
  'NemotronHTwoTowerForCausalLM',
  'twotower-denoiser-v1',
  'twotower-context-v1',
  'TwoTowerSession',
  'diffusion_block_commit',
  'TWOTOWER_SPEC',
]);

const TWOTOWER_SOURCE_RE = /(?:^|\/)(?:src\/)?twotower\//i;

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
  // Skip the tag name itself
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
    // Try stripping 0, 1, 2, ... leading base-path segments until the file exists in outDir.
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

function isInsideDir(targetPath, baseDir) {
  const rel = relative(baseDir, targetPath);
  return rel !== '' && !rel.startsWith('..') && !rel.startsWith(`..${sep}`);
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
  // Prefer explicit sourceMappingURL trailer when present; otherwise default `<chunk>.map`.
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

function inspectChunkSourcemap(jsFilePath, outDir, repoRoot) {
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
  return {
    chunk: relJs,
    mapFile: toPosix(relative(outDir, mapPath)),
    rawSources: parsed.sources.map(String),
    normalizedSources,
  };
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

function main() {
  const { outDir, outDirInput } = parseCliArgs(process.argv.slice(2));
  const repoRoot = process.cwd();
  const violations = [];

  // Gate 1: Both entry HTML files must exist.
  const nanoHtmlPath = resolve(outDir, 'index.html');
  const twoTowerHtmlPath = resolve(outDir, 'twotower/index.html');
  if (!existsSync(nanoHtmlPath)) {
    violations.push(`Gate 1 [Missing Entry]: Nano entry HTML not found at "${nanoHtmlPath}"`);
  }
  if (!existsSync(twoTowerHtmlPath)) {
    violations.push(
      `Gate 1 [Missing Entry]: TwoTower entry HTML not found at "${twoTowerHtmlPath}"`,
    );
  }
  if (violations.length > 0) {
    reportAndExit(violations, null);
  }

  // Gates 2 & 3: Parse HTML refs and recursively walk static ES import/export graphs.
  const detectedBasePrefixes = new Set();
  const nanoGraph = walkEntryGraph('nano', nanoHtmlPath, outDir, detectedBasePrefixes);
  const twoTowerGraph = walkEntryGraph('twotower', twoTowerHtmlPath, outDir, detectedBasePrefixes);

  if (detectedBasePrefixes.size > 1) {
    violations.push(
      `Gate 2 [Inconsistent Base Path]: Multiple root base prefixes detected: ${[...detectedBasePrefixes].join(', ')}`,
    );
  }

  // Gate 4: Strict Source-Map Isolation Gate for Nano.
  const nanoSourcemaps = [];
  const nanoSourcemapLeaks = [];
  for (const jsPath of nanoGraph.jsChunks) {
    const info = inspectChunkSourcemap(jsPath, outDir, repoRoot);
    nanoSourcemaps.push(info);
    for (let i = 0; i < info.normalizedSources.length; i++) {
      const norm = info.normalizedSources[i];
      if (TWOTOWER_SOURCE_RE.test(norm)) {
        nanoSourcemapLeaks.push({ chunk: info.chunk, source: norm, raw: info.rawSources[i] });
      }
    }
  }
  if (nanoSourcemapLeaks.length > 0) {
    for (const leak of nanoSourcemapLeaks) {
      violations.push(
        `Gate 4 [Source-Map Leak]: Nano chunk "${leak.chunk}" includes TwoTower source "${leak.source}"`,
      );
    }
  }

  // Gate 5: Strict Payload Marker Gate across every JS/CSS file in Nano's transitive graph.
  const nanoMarkerHits = [];
  for (const assetPath of nanoGraph.allAssets) {
    const content = readFileSync(assetPath, 'utf8');
    const relAsset = toPosix(relative(outDir, assetPath));
    for (const marker of FORBIDDEN_NANO_MARKERS) {
      if (content.includes(marker)) {
        nanoMarkerHits.push({ asset: relAsset, marker });
      }
    }
  }
  if (nanoMarkerHits.length > 0) {
    for (const hit of nanoMarkerHits) {
      violations.push(
        `Gate 5 [Payload Marker Leak]: Nano asset "${hit.asset}" contains forbidden TwoTower marker "${hit.marker}"`,
      );
    }
  }

  // Gate 6: TwoTower Positive Presence Gate (proves sourcemaps are real and audited).
  const twoTowerSourcemaps = [];
  const twoTowerConfirmedSources = [];
  for (const jsPath of twoTowerGraph.jsChunks) {
    const info = inspectChunkSourcemap(jsPath, outDir, repoRoot);
    twoTowerSourcemaps.push(info);
    for (const norm of info.normalizedSources) {
      if (TWOTOWER_SOURCE_RE.test(norm)) {
        twoTowerConfirmedSources.push({ chunk: info.chunk, source: norm });
      }
    }
  }
  if (twoTowerConfirmedSources.length === 0) {
    violations.push(
      'Gate 6 [Positive Presence]: TwoTower transitive JS sourcemaps contain zero "src/twotower/" sources',
    );
  }

  // Gate 7: Exact Raw & Gzip (level 9) Payload Budget Accounting.
  const nanoAssetSet = new Set(nanoGraph.allAssets);
  const twoTowerAssetSet = new Set(twoTowerGraph.allAssets);
  const sharedAssetPaths = [...nanoAssetSet].filter((p) => twoTowerAssetSet.has(p)).sort();
  const nanoOnlyAssetPaths = [...nanoAssetSet].filter((p) => !twoTowerAssetSet.has(p)).sort();
  const twoTowerOnlyAssetPaths = [...twoTowerAssetSet].filter((p) => !nanoAssetSet.has(p)).sort();

  const sizeCache = new Map();
  const getSize = (p) => {
    if (!sizeCache.has(p)) {
      sizeCache.set(p, measureFileSizes(p, outDir));
    }
    return sizeCache.get(p);
  };

  const nanoAssetsSized = nanoGraph.allAssets.map(getSize);
  const twoTowerAssetsSized = twoTowerGraph.allAssets.map(getSize);
  const sharedAssetsSized = sharedAssetPaths.map(getSize);
  const nanoOnlySized = nanoOnlyAssetPaths.map(getSize);
  const twoTowerOnlySized = twoTowerOnlyAssetPaths.map(getSize);

  const nanoHtmlSize = measureFileSizes(nanoHtmlPath, outDir);
  const twoTowerHtmlSize = measureFileSizes(twoTowerHtmlPath, outDir);

  const nanoTotals = sumSizes(nanoAssetsSized);
  const twoTowerTotals = sumSizes(twoTowerAssetsSized);
  const sharedTotals = sumSizes(sharedAssetsSized);
  const nanoOnlyTotals = sumSizes(nanoOnlySized);
  const twoTowerOnlyTotals = sumSizes(twoTowerOnlySized);

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
      nanoTransitiveJsCount: nanoGraph.jsChunks.length,
      nanoTransitiveCssCount: nanoGraph.cssAssets.length,
      nanoImportEdgeCount: nanoGraph.edges.length,
      twoTowerTransitiveJsCount: twoTowerGraph.jsChunks.length,
      twoTowerTransitiveCssCount: twoTowerGraph.cssAssets.length,
      twoTowerImportEdgeCount: twoTowerGraph.edges.length,
      nanoSourcemapLeaks: nanoSourcemapLeaks.length,
      nanoForbiddenMarkerHits: nanoMarkerHits.length,
      twoTowerPositiveSourceCount: twoTowerConfirmedSources.length,
      nanoGzipBudgetBytes: NANO_GZIP_BUDGET_BYTES,
      nanoGzipBytes: nanoTotals.gzipBytes,
      nanoGzipWithinBudget: nanoTotals.gzipBytes <= NANO_GZIP_BUDGET_BYTES,
    },
    totals: {
      nanoTransitiveAssets: nanoTotals,
      nanoOnlyAssets: nanoOnlyTotals,
      twoTowerTransitiveAssets: twoTowerTotals,
      twoTowerOnlyAssets: twoTowerOnlyTotals,
      sharedAssets: sharedTotals,
      htmlEntries: {
        nano: nanoHtmlSize,
        twotower: twoTowerHtmlSize,
      },
    },
    chunks: {
      nano: nanoAssetsSized,
      twotower: twoTowerAssetsSized,
      shared: sharedAssetsSized,
    },
    twoTowerConfirmedSources: twoTowerConfirmedSources.map((s) => s.source),
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
  console.log('=== Nano / TwoTower Bundle Isolation Audit ===');
  console.log(`Output dir        : ${report.outDir}`);
  console.log(`Detected base     : ${report.detectedBasePath}`);
  console.log(
    `Nano graph        : ${gates.nanoTransitiveJsCount} JS, ${gates.nanoTransitiveCssCount} CSS | raw ${totals.nanoTransitiveAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.nanoTransitiveAssets.rawBytes)}) | gzip ${totals.nanoTransitiveAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.nanoTransitiveAssets.gzipBytes)}) / budget ${fmtKiB(gates.nanoGzipBudgetBytes)}`,
  );
  console.log(
    `TwoTower graph    : ${gates.twoTowerTransitiveJsCount} JS, ${gates.twoTowerTransitiveCssCount} CSS | raw ${totals.twoTowerTransitiveAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.twoTowerTransitiveAssets.rawBytes)}) | gzip ${totals.twoTowerTransitiveAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.twoTowerTransitiveAssets.gzipBytes)})`,
  );
  console.log(
    `Shared chunks     : ${chunks.shared.length} file(s) | raw ${totals.sharedAssets.rawBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.rawBytes)}) | gzip ${totals.sharedAssets.gzipBytes.toLocaleString('en-US')} B (${fmtKiB(totals.sharedAssets.gzipBytes)})`,
  );
  console.log(
    `Isolation gates   : sourcemapLeaks=${gates.nanoSourcemapLeaks}, markerHits=${gates.nanoForbiddenMarkerHits}, twoTowerSources=${gates.twoTowerPositiveSourceCount}`,
  );

  const sharedSet = new Set(chunks.shared.map((c) => c.file));
  const rows = [
    ...chunks.nano.map((c) => ({
      scope: sharedSet.has(c.file) ? 'shared' : 'nano-only',
      ...c,
    })),
    ...chunks.twotower
      .filter((c) => !sharedSet.has(c.file))
      .map((c) => ({ scope: 'twotower-only', ...c })),
  ];

  console.log('\nScope           Kind   Raw (B)    Gzip (B)   Asset');
  console.log('--------------  -----  ---------  ---------  ----------------------------------------');
  for (const r of rows) {
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
  console.error(`Bundle isolation verification ERROR: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}
