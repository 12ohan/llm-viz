import type { ViewportMetrics } from './camera';

/** Display-space sRGB palette (§2). */
export const PALETTE = {
  paper: [246 / 255, 243 / 255, 235 / 255] as const,     // #F6F3EB
  topFace: [237 / 255, 233 / 255, 223 / 255] as const,   // #EDE9DF
  frontFace: [221 / 255, 216 / 255, 203 / 255] as const, // #DDD8CB
  rightFace: [203 / 255, 197 / 255, 184 / 255] as const, // #CBC5B8
  ink: [64 / 255, 61 / 255, 56 / 255] as const,          // #403D38
  hairline: [69 / 255, 65 / 255, 58 / 255] as const,     // #45413A
  oxide: [160 / 255, 71 / 255, 50 / 255] as const,       // #A04732
  fp8Stone: [191 / 255, 183 / 255, 168 / 255] as const,  // #BFB7A8
  guideTint: [198 / 255, 192 / 255, 180 / 255] as const, // subtle matte guide rails/residual
} as const;

export interface FramebufferViewport {
  readonly backingWidth: number;
  readonly backingHeight: number;
  readonly vpX: number;
  readonly vpY: number;
  readonly vpWidth: number;
  readonly vpHeight: number;
}

export function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error('Failed to create WebGL shader');
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${info}`);
  }
  return shader;
}

export function linkProgram(
  gl: WebGL2RenderingContext,
  vertexSrc: string,
  fragmentSrc: string,
): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertexSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error('Failed to create WebGL program');
  }
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program) || 'Unknown link error';
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${info}`);
  }
  return program;
}

export function computeFramebufferViewport(metrics: ViewportMetrics): FramebufferViewport {
  const backingWidth = Math.max(1, Math.round(metrics.hostWidth * metrics.dpr));
  const backingHeight = Math.max(1, Math.round(metrics.hostHeight * metrics.dpr));
  const topPx = Math.round(metrics.insetTop * metrics.dpr);
  const bottomPx = Math.round(metrics.insetBottom * metrics.dpr);
  const vpHeight = Math.max(1, backingHeight - topPx - bottomPx);
  return {
    backingWidth,
    backingHeight,
    vpX: 0,
    vpY: bottomPx,
    vpWidth: backingWidth,
    vpHeight,
  };
}

/**
 * Manages the WebGL2 context, RGBA8 + DEPTH_COMPONENT24 multisampled scene FBO,
 * canvas blit, and context loss / restoration lifecycle (§3).
 */
export class RenderDevice {
  readonly canvas: HTMLCanvasElement;
  readonly gl: WebGL2RenderingContext;
  private fbo: WebGLFramebuffer | null = null;
  private colorRb: WebGLRenderbuffer | null = null;
  private depthRb: WebGLRenderbuffer | null = null;
  private resolveFbo: WebGLFramebuffer | null = null;
  private resolveColorRb: WebGLRenderbuffer | null = null;
  private width = 0;
  private height = 0;
  private samples = 0;
  private lost = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    // Default backbuffer must be single-sample (antialias: false) so WebGL2 allows
    // glBlitFramebuffer into it; 4x MSAA is performed in our offscreen scene FBO (§3).
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      depth: true,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      throw new Error('WebGL2 is not available in this browser.');
    }
    this.gl = gl;
    this.samples = this.queryBestSampleCount();
  }

  get isLost(): boolean {
    return this.lost;
  }

  get sampleCount(): number {
    return this.samples;
  }

  markLost(value: boolean): void {
    this.lost = value;
    if (value) {
      this.fbo = null;
      this.colorRb = null;
      this.depthRb = null;
      this.resolveFbo = null;
      this.resolveColorRb = null;
      this.width = 0;
      this.height = 0;
    } else {
      this.samples = this.queryBestSampleCount();
    }
  }

  private queryBestSampleCount(): number {
    const gl = this.gl;
    try {
      const colorSamples = gl.getInternalformatParameter(
        gl.RENDERBUFFER,
        gl.RGBA8,
        gl.SAMPLES,
      ) as Int32Array | null;
      const depthSamples = gl.getInternalformatParameter(
        gl.RENDERBUFFER,
        gl.DEPTH_COMPONENT24,
        gl.SAMPLES,
      ) as Int32Array | null;

      if (!colorSamples || !depthSamples) return 0;
      const depthSet = new Set<number>(Array.from(depthSamples));
      let best = 0;
      for (const s of colorSamples) {
        if (s <= 4 && s > best && depthSet.has(s)) {
          best = s;
        }
      }
      return best;
    } catch {
      return 0;
    }
  }

  /** Ensure backing canvas and offscreen RGBA8 + DEPTH_COMPONENT24 FBO match metrics. */
  ensureSize(metrics: ViewportMetrics): FramebufferViewport {
    const fb = computeFramebufferViewport(metrics);
    if (this.canvas.width !== fb.backingWidth || this.canvas.height !== fb.backingHeight) {
      this.canvas.width = fb.backingWidth;
      this.canvas.height = fb.backingHeight;
    }
    if (this.fbo && this.width === fb.backingWidth && this.height === fb.backingHeight) {
      return fb;
    }
    this.allocateFramebuffer(fb.backingWidth, fb.backingHeight);
    return fb;
  }

  private allocateFramebuffer(width: number, height: number): void {
    const gl = this.gl;
    this.destroyFramebuffer();

    const fbo = gl.createFramebuffer();
    const colorRb = gl.createRenderbuffer();
    const depthRb = gl.createRenderbuffer();
    if (!fbo || !colorRb || !depthRb) {
      this.destroyFramebuffer();
      return;
    }

    const tryBuild = (samples: number): boolean => {
      gl.bindRenderbuffer(gl.RENDERBUFFER, colorRb);
      if (samples > 1) {
        gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA8, width, height);
      } else {
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, width, height);
      }

      gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
      if (samples > 1) {
        gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, width, height);
      } else {
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
      }

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRb);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb);

      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
      return status === gl.FRAMEBUFFER_COMPLETE;
    };

    let ok = this.samples > 1 ? tryBuild(this.samples) : false;
    if (!ok) {
      this.samples = 0;
      ok = tryBuild(0);
    }

    if (!ok) {
      gl.deleteFramebuffer(fbo);
      gl.deleteRenderbuffer(colorRb);
      gl.deleteRenderbuffer(depthRb);
      this.width = width;
      this.height = height;
      return;
    }

    // When MSAA is active, allocate a single-sample RGBA8 resolve target so blitting
    // to a multisampled default canvas backbuffer is always valid in WebGL2 (§3).
    if (this.samples > 1) {
      const resolveFbo = gl.createFramebuffer();
      const resolveColorRb = gl.createRenderbuffer();
      if (resolveFbo && resolveColorRb) {
        gl.bindRenderbuffer(gl.RENDERBUFFER, resolveColorRb);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, width, height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, resolveFbo);
        gl.framebufferRenderbuffer(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.RENDERBUFFER,
          resolveColorRb,
        );
        const rStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);
        if (rStatus === gl.FRAMEBUFFER_COMPLETE) {
          this.resolveFbo = resolveFbo;
          this.resolveColorRb = resolveColorRb;
        } else {
          gl.deleteFramebuffer(resolveFbo);
          gl.deleteRenderbuffer(resolveColorRb);
        }
      }
    }

    this.fbo = fbo;
    this.colorRb = colorRb;
    this.depthRb = depthRb;
    this.width = width;
    this.height = height;
  }

  /** Bind scene FBO (or default canvas fallback), clear to warm paper and depth=1, and scissor to inner viewport. */
  beginFrame(fb: FramebufferViewport): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, fb.backingWidth, fb.backingHeight);
    gl.depthMask(true);
    gl.colorMask(true, true, true, true);
    gl.clearColor(PALETTE.paper[0], PALETTE.paper[1], PALETTE.paper[2], 1.0);
    gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Constrain scene passes cleanly to the inner drawing viewport.
    gl.viewport(fb.vpX, fb.vpY, fb.vpWidth, fb.vpHeight);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(fb.vpX, fb.vpY, fb.vpWidth, fb.vpHeight);
  }

  /** Restore GL state, resolve multisampled color to single-sample if needed, and blit to canvas. */
  endFrame(fb: FramebufferViewport): void {
    const gl = this.gl;
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.depthMask(true);

    if (this.fbo) {
      if (this.resolveFbo) {
        // Step 1: resolve MSAA scene FBO -> single-sample RGBA8 FBO
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.fbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.resolveFbo);
        gl.blitFramebuffer(
          0, 0, fb.backingWidth, fb.backingHeight,
          0, 0, fb.backingWidth, fb.backingHeight,
          gl.COLOR_BUFFER_BIT,
          gl.NEAREST,
        );
        // Step 2: blit single-sample RGBA8 FBO -> default canvas backbuffer
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.resolveFbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
        gl.blitFramebuffer(
          0, 0, fb.backingWidth, fb.backingHeight,
          0, 0, fb.backingWidth, fb.backingHeight,
          gl.COLOR_BUFFER_BIT,
          gl.NEAREST,
        );
      } else {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.fbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
        gl.blitFramebuffer(
          0, 0, fb.backingWidth, fb.backingHeight,
          0, 0, fb.backingWidth, fb.backingHeight,
          gl.COLOR_BUFFER_BIT,
          gl.NEAREST,
        );
      }
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    }
  }

  destroyFramebuffer(): void {
    const gl = this.gl;
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.colorRb) gl.deleteRenderbuffer(this.colorRb);
    if (this.depthRb) gl.deleteRenderbuffer(this.depthRb);
    if (this.resolveFbo) gl.deleteFramebuffer(this.resolveFbo);
    if (this.resolveColorRb) gl.deleteRenderbuffer(this.resolveColorRb);
    this.fbo = null;
    this.colorRb = null;
    this.depthRb = null;
    this.resolveFbo = null;
    this.resolveColorRb = null;
    this.width = 0;
    this.height = 0;
  }

  dispose(): void {
    this.destroyFramebuffer();
  }
}
