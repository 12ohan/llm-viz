import type { ViewportMetrics } from './camera';

/** Crisp studio white + llm-viz charcoal-slate & tensor cell palette. */
export const PALETTE = {
  studioWhite: [248 / 255, 249 / 255, 251 / 255] as const, // #F8F9FB
  slabTop: [75 / 255, 82 / 255, 98 / 255] as const,        // #4B5262
  slabFront: [65 / 255, 71 / 255, 86 / 255] as const,      // #414756
  slabSide: [54 / 255, 60 / 255, 73 / 255] as const,       // #363C49
  cellDark: [30 / 255, 34 / 255, 43 / 255] as const,       // #1E222B
  cellBlue: [99 / 255, 123 / 255, 245 / 255] as const,     // #637BF5 periwinkle weights
  cellGreen: [74 / 255, 222 / 255, 128 / 255] as const,    // #4ADE80 emerald activations
  cellCyan: [56 / 255, 189 / 255, 248 / 255] as const,     // #38BDF8 Mamba SSM cyan
  cellAmber: [245 / 255, 158 / 255, 11 / 255] as const,    // #F59E0B MoE/norm amber
  fp8Frosted: [203 / 255, 210 / 255, 222 / 255] as const,  // #CBD2DE pale frosted slate
  hairline: [32 / 255, 37 / 255, 48 / 255] as const,       // #202530 crisp dark edge
  guideLine: [148 / 255, 163 / 255, 184 / 255] as const,   // #94A3B8 subtle structural guide
  activeStroke: [224 / 255, 79 / 255, 51 / 255] as const,  // #E04F33 vivid active selection
  bf16Gold: [245 / 255, 158 / 255, 11 / 255] as const,     // #F59E0B BF16 preserved border
  ribbonEmerald: [74 / 255, 222 / 255, 128 / 255] as const,
  ribbonPeriwinkle: [129 / 255, 140 / 255, 248 / 255] as const,
  ribbonCyan: [56 / 255, 189 / 255, 248 / 255] as const,
  ribbonAmber: [245 / 255, 158 / 255, 11 / 255] as const,
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
 * WebGL2 device managing 4x MSAA offscreen RGBA8 + DEPTH_COMPONENT24 FBO,
 * single-sample resolve FBO, and crisp blit to default canvas backbuffer.
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

  beginFrame(fb: FramebufferViewport): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, fb.backingWidth, fb.backingHeight);
    gl.depthMask(true);
    gl.colorMask(true, true, true, true);
    gl.clearColor(
      PALETTE.studioWhite[0],
      PALETTE.studioWhite[1],
      PALETTE.studioWhite[2],
      1.0,
    );
    gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.viewport(fb.vpX, fb.vpY, fb.vpWidth, fb.vpHeight);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(fb.vpX, fb.vpY, fb.vpWidth, fb.vpHeight);
  }

  endFrame(fb: FramebufferViewport): void {
    const gl = this.gl;
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.depthMask(true);

    if (this.fbo) {
      if (this.resolveFbo) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.fbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.resolveFbo);
        gl.blitFramebuffer(
          0, 0, fb.backingWidth, fb.backingHeight,
          0, 0, fb.backingWidth, fb.backingHeight,
          gl.COLOR_BUFFER_BIT,
          gl.NEAREST,
        );
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
