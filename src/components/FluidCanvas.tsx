"use client";

import { useEffect, useRef } from "react";

/**
 * A lightweight, GPU-driven fluid simulation (Navier-Stokes on a grid).
 *
 * Concept: the viewport is an "infinite tank" of clear water. The cursor is the
 * source of blue dye. A persistent outward drift continuously pulls colour
 * toward the edges of the tank, where it drains away — so the centre stays calm
 * and readable while ink keeps streaming outward.
 *
 * Reliability strategy (why this file is structured the way it is):
 * Mobile Safari can hand back a WebGL context that is *born lost* (GPU process
 * over its context budget, Low Power Mode, memory pressure during page load).
 * Once a canvas hands out a context, `getContext` always returns that same
 * (dead) object — so the only way to recover is to throw the canvas away and
 * create a brand-new one. We therefore:
 *   1. Defer acquisition until after `load` + idle, and only while the page is
 *      visible, so we aren't fighting for the GPU during hydration.
 *   2. On a born-lost / unrecoverable context, discard the canvas and retry on
 *      a FRESH canvas with exponential backoff.
 *   3. Fall back to the always-present CSS gradient (`.fluid-fallback`) if the
 *      device simply won't grant a usable context, so the page is never blank.
 */

type FBO = {
  texture: WebGLTexture;
  fbo: WebGLFramebuffer;
  width: number;
  height: number;
  texelSizeX: number;
  texelSizeY: number;
  attach: (id: number) => number;
};

type DoubleFBO = {
  width: number;
  height: number;
  texelSizeX: number;
  texelSizeY: number;
  read: FBO;
  write: FBO;
  swap: () => void;
};

type Program = {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
};

// ---- Tunable simulation parameters -----------------------------------------
const SIM_RESOLUTION = 128; // velocity / pressure grid
const DYE_RESOLUTION = 512; // dye (visual) grid
const PRESSURE_ITERATIONS = 20;
const PRESSURE_DISSIPATION = 0.8;
const VELOCITY_DISSIPATION = 0.2;
const DENSITY_DISSIPATION = 0.6; // how fast ink fades
const CURL = 28; // vorticity / swirl strength
const SPLAT_RADIUS = 0.0035;
const SPLAT_FORCE = 6200;
const INK_STRENGTH = 0.18; // dye added per pointer move
const EDGE_DRAIN = 0.05; // extra fade near the tank edges
const OUTWARD_DRIFT = 0.65; // how hard the tank sucks ink toward edges

// ---- Context lifecycle ------------------------------------------------------
const MAX_ACQUIRE_ATTEMPTS = 8; // fresh-canvas attempts before the CSS fallback
const RESTORE_TIMEOUT_MS = 2500; // wait for `restored` before recreating a context

// ---- shaders ----------------------------------------------------------------
const baseVertex = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPosition;
out vec2 vUv;
out vec2 vL;
out vec2 vR;
out vec2 vT;
out vec2 vB;
uniform vec2 texelSize;
void main () {
  vUv = aPosition * 0.5 + 0.5;
  vL = vUv - vec2(texelSize.x, 0.0);
  vR = vUv + vec2(texelSize.x, 0.0);
  vT = vUv + vec2(0.0, texelSize.y);
  vB = vUv - vec2(0.0, texelSize.y);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const splatShader = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
void main () {
  vec2 p = vUv - point.xy;
  p.x *= aspectRatio;
  vec3 splat = exp(-dot(p, p) / radius) * color;
  vec3 base = texture(uTarget, vUv).xyz;
  fragColor = vec4(base + splat, 1.0);
}`;

const advectionShader = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 uTexelSize; // velocity grid texel size
uniform float dt;
uniform float dissipation;
uniform float drift;     // outward pull (dye only)
uniform float edgeDrain; // extra fade at edges (dye only)
void main () {
  vec2 vel = texture(uVelocity, vUv).xy;
  // velocity is in grid texels/sec -> scale to uv with the grid texel size
  vec2 coord = vUv - dt * vel * uTexelSize;
  // radial outward drift: sample from further inward so ink moves outward
  vec2 outward = vUv - vec2(0.5);
  coord -= dt * drift * outward;
  vec4 result = texture(uSource, coord);
  float decay = 1.0 / (1.0 + dissipation * dt);
  result *= decay;
  float e = max(abs(vUv.x - 0.5), abs(vUv.y - 0.5)) * 2.0;
  result *= (1.0 - edgeDrain * smoothstep(0.5, 1.0, e));
  fragColor = result;
}`;

const divergenceShader = `#version 300 es
precision highp float;
in vec2 vUv;
in vec2 vL;
in vec2 vR;
in vec2 vT;
in vec2 vB;
out vec4 fragColor;
uniform sampler2D uVelocity;
void main () {
  float L = texture(uVelocity, vL).x;
  float R = texture(uVelocity, vR).x;
  float T = texture(uVelocity, vT).y;
  float B = texture(uVelocity, vB).y;
  vec2 C = texture(uVelocity, vUv).xy;
  if (vL.x < 0.0) { L = -C.x; }
  if (vR.x > 1.0) { R = -C.x; }
  if (vT.y > 1.0) { T = -C.y; }
  if (vB.y < 0.0) { B = -C.y; }
  float div = 0.5 * (R - L + T - B);
  fragColor = vec4(div, 0.0, 0.0, 1.0);
}`;

const curlShader = `#version 300 es
precision highp float;
in vec2 vUv;
in vec2 vL;
in vec2 vR;
in vec2 vT;
in vec2 vB;
out vec4 fragColor;
uniform sampler2D uVelocity;
void main () {
  float L = texture(uVelocity, vL).y;
  float R = texture(uVelocity, vR).y;
  float T = texture(uVelocity, vT).x;
  float B = texture(uVelocity, vB).x;
  float vorticity = R - L - T + B;
  fragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}`;

const vorticityShader = `#version 300 es
precision highp float;
in vec2 vUv;
in vec2 vL;
in vec2 vR;
in vec2 vT;
in vec2 vB;
out vec4 fragColor;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
  float L = texture(uCurl, vL).x;
  float R = texture(uCurl, vR).x;
  float T = texture(uCurl, vT).x;
  float B = texture(uCurl, vB).x;
  float C = texture(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  force *= curl * C;
  force.y *= -1.0;
  vec2 vel = texture(uVelocity, vUv).xy;
  vel += force * dt;
  vel = clamp(vel, -1000.0, 1000.0);
  fragColor = vec4(vel, 0.0, 1.0);
}`;

const pressureShader = `#version 300 es
precision highp float;
in vec2 vUv;
in vec2 vL;
in vec2 vR;
in vec2 vT;
in vec2 vB;
out vec4 fragColor;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
void main () {
  float L = texture(uPressure, vL).x;
  float R = texture(uPressure, vR).x;
  float T = texture(uPressure, vT).x;
  float B = texture(uPressure, vB).x;
  float divergence = texture(uDivergence, vUv).x;
  float pressure = (L + R + B + T - divergence) * 0.25;
  fragColor = vec4(pressure, 0.0, 0.0, 1.0);
}`;

const gradientSubtractShader = `#version 300 es
precision highp float;
in vec2 vUv;
in vec2 vL;
in vec2 vR;
in vec2 vT;
in vec2 vB;
out vec4 fragColor;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
void main () {
  float L = texture(uPressure, vL).x;
  float R = texture(uPressure, vR).x;
  float T = texture(uPressure, vT).x;
  float B = texture(uPressure, vB).x;
  vec2 velocity = texture(uVelocity, vUv).xy;
  velocity -= vec2(R - L, T - B);
  fragColor = vec4(velocity, 0.0, 1.0);
}`;

const clearShader = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform float value;
void main () {
  fragColor = value * texture(uTexture, vUv);
}`;

const displayShader = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTexture;
void main () {
  float c = clamp(texture(uTexture, vUv).x, 0.0, 1.0);
  vec3 water = vec3(0.957, 0.969, 0.984);
  vec3 inkLight = vec3(0.36, 0.64, 0.97);
  vec3 inkDeep = vec3(0.03, 0.20, 0.62);
  float t = pow(c, 0.72);
  vec3 ink = mix(inkLight, inkDeep, smoothstep(0.15, 1.0, c));
  vec3 color = mix(water, ink, clamp(t, 0.0, 1.0));
  // soft vignette toward the edges of the tank
  vec2 d = vUv - 0.5;
  float vignette = 1.0 - 0.10 * dot(d, d) * 4.0;
  fragColor = vec4(color * vignette, 1.0);
}`;

type SimConfig = {
  dyeResolution: number;
  simResolution: number;
  pressureIterations: number;
  dprCap: number;
};

type Fmt = { internalFormat: number; format: number };

/**
 * A simulation bound to one specific (healthy) WebGL2 context. Everything that
 * touches `gl` lives in this closure so the whole thing can be discarded and
 * rebuilt against a fresh context after a loss. Input handling and the rAF loop
 * live in the component effect (they survive context recreation).
 */
type Simulation = {
  canvas: HTMLCanvasElement;
  resizeIfNeeded: () => boolean;
  initFramebuffers: () => void;
  rebuildResources: () => boolean;
  splat: (
    x: number,
    y: number,
    dx: number,
    dy: number,
    amount: number,
    radius: number,
  ) => void;
  step: (dt: number) => void;
  render: () => void;
  dispose: () => void;
};

function createSimulation(
  canvas: HTMLCanvasElement,
  gl: WebGL2RenderingContext,
  cfg: SimConfig,
): Simulation {
  // Float render targets need an explicit color-buffer extension. iOS Safari
  // (and some Android GPUs) only expose the half-float variant, so request
  // both; without one of these, half-float framebuffers are incomplete and
  // every draw silently no-ops, leaving a black canvas.
  gl.getExtension("EXT_color_buffer_float");
  gl.getExtension("EXT_color_buffer_half_float");
  gl.disable(gl.BLEND);

  const halfFloat = gl.HALF_FLOAT;

  // ---- format probing ----------------------------------------------------
  const supportRenderTextureFormat = (
    internalFormat: number,
    format: number,
    type: number,
  ) => {
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    );
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(texture);
    return status === gl.FRAMEBUFFER_COMPLETE;
  };

  const getSupportedFormat = (
    internalFormat: number,
    format: number,
    type: number,
  ): Fmt | null => {
    if (!supportRenderTextureFormat(internalFormat, format, type)) {
      switch (internalFormat) {
        case gl.R16F:
          return getSupportedFormat(gl.RG16F, gl.RG, type);
        case gl.RG16F:
          return getSupportedFormat(gl.RGBA16F, gl.RGBA, type);
        default:
          return null;
      }
    }
    return { internalFormat, format };
  };

  let formatR!: Fmt;
  let formatRG!: Fmt;
  const resolveFormats = () => {
    // RGBA is the ultimate fallback target; if even it isn't renderable the
    // context isn't usable (typically because it's lost).
    const rgba = getSupportedFormat(gl.RGBA16F, gl.RGBA, halfFloat);
    const rg = getSupportedFormat(gl.RG16F, gl.RG, halfFloat);
    const r = getSupportedFormat(gl.R16F, gl.RED, halfFloat);
    console.log("[fluid] renderable formats", {
      rgba: !!rgba,
      rg: !!rg,
      r: !!r,
      isContextLost: gl.isContextLost(),
    });
    if (!rgba || !rg || !r) return false;
    formatRG = rg;
    formatR = r;
    return true;
  };

  // ---- shader / program helpers -----------------------------------------
  const compileShader = (type: number, source: string) => {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader), source);
    }
    return shader;
  };

  const getUniforms = (program: WebGLProgram) => {
    const uniforms: Record<string, WebGLUniformLocation | null> = {};
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(program, i);
      if (!info) continue;
      uniforms[info.name] = gl.getUniformLocation(program, info.name);
    }
    return uniforms;
  };

  const createProgram = (vs: string, fs: string): Program => {
    const program = gl.createProgram()!;
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
    }
    return { program, uniforms: getUniforms(program) };
  };

  // ---- GPU resources -----------------------------------------------------
  let splatProgram: Program | undefined;
  let advectionProgram: Program | undefined;
  let divergenceProgram: Program | undefined;
  let curlProgram: Program | undefined;
  let vorticityProgram: Program | undefined;
  let pressureProgram: Program | undefined;
  let gradientProgram: Program | undefined;
  let clearProgram: Program | undefined;
  let displayProgram: Program | undefined;
  let vao: WebGLVertexArrayObject | undefined;
  let quadBuffer: WebGLBuffer | undefined;
  let indexBuffer: WebGLBuffer | undefined;

  let dye: DoubleFBO | undefined;
  let velocity: DoubleFBO | undefined;
  let divergence: FBO | undefined;
  let curl: FBO | undefined;
  let pressure: DoubleFBO | undefined;

  const blit = (target: FBO | null) => {
    if (target == null) {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      gl.viewport(0, 0, target.width, target.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  };

  const createFBO = (
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    param: number,
  ): FBO => {
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    );
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return {
      texture,
      fbo,
      width: w,
      height: h,
      texelSizeX: 1 / w,
      texelSizeY: 1 / h,
      attach(id: number) {
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        return id;
      },
    };
  };

  const createDoubleFBO = (
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    param: number,
  ): DoubleFBO => {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);
    return {
      width: w,
      height: h,
      texelSizeX: 1 / w,
      texelSizeY: 1 / h,
      get read() {
        return fbo1;
      },
      set read(value) {
        fbo1 = value;
      },
      get write() {
        return fbo2;
      },
      set write(value) {
        fbo2 = value;
      },
      swap() {
        const temp = fbo1;
        fbo1 = fbo2;
        fbo2 = temp;
      },
    };
  };

  const getResolution = (resolution: number) => {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1) aspectRatio = 1 / aspectRatio;
    const min = Math.round(resolution);
    const max = Math.round(resolution * aspectRatio);
    if (gl.drawingBufferWidth > gl.drawingBufferHeight)
      return { width: max, height: min };
    return { width: min, height: max };
  };

  const deleteFBO = (f: FBO | undefined) => {
    if (!f) return;
    gl.deleteTexture(f.texture);
    gl.deleteFramebuffer(f.fbo);
  };

  const initFramebuffers = () => {
    if (dye) {
      deleteFBO(dye.read);
      deleteFBO(dye.write);
    }
    if (velocity) {
      deleteFBO(velocity.read);
      deleteFBO(velocity.write);
    }
    deleteFBO(divergence);
    deleteFBO(curl);
    if (pressure) {
      deleteFBO(pressure.read);
      deleteFBO(pressure.write);
    }

    const simRes = getResolution(cfg.simResolution);
    const dyeRes = getResolution(cfg.dyeResolution);
    const type = halfFloat;

    dye = createDoubleFBO(
      dyeRes.width,
      dyeRes.height,
      formatR.internalFormat,
      formatR.format,
      type,
      gl.LINEAR,
    );
    velocity = createDoubleFBO(
      simRes.width,
      simRes.height,
      formatRG.internalFormat,
      formatRG.format,
      type,
      gl.LINEAR,
    );
    divergence = createFBO(
      simRes.width,
      simRes.height,
      formatR.internalFormat,
      formatR.format,
      type,
      gl.NEAREST,
    );
    curl = createFBO(
      simRes.width,
      simRes.height,
      formatR.internalFormat,
      formatR.format,
      type,
      gl.NEAREST,
    );
    pressure = createDoubleFBO(
      simRes.width,
      simRes.height,
      formatR.internalFormat,
      formatR.format,
      type,
      gl.NEAREST,
    );
  };

  const resizeIfNeeded = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, cfg.dprCap);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (w === 0 || h === 0) return false;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      return true;
    }
    return false;
  };

  const deletePrograms = () => {
    [
      splatProgram,
      advectionProgram,
      divergenceProgram,
      curlProgram,
      vorticityProgram,
      pressureProgram,
      gradientProgram,
      clearProgram,
      displayProgram,
    ].forEach((p) => p && gl.deleteProgram(p.program));
  };

  // Build (or rebuild, after a context restore) every GPU resource. Returns
  // false when the formats can't be resolved — i.e. the context isn't actually
  // usable yet — so the caller can recreate the context instead.
  const rebuildResources = () => {
    if (gl.isContextLost()) return false;
    if (!resolveFormats()) return false;

    deletePrograms();
    splatProgram = createProgram(baseVertex, splatShader);
    advectionProgram = createProgram(baseVertex, advectionShader);
    divergenceProgram = createProgram(baseVertex, divergenceShader);
    curlProgram = createProgram(baseVertex, curlShader);
    vorticityProgram = createProgram(baseVertex, vorticityShader);
    pressureProgram = createProgram(baseVertex, pressureShader);
    gradientProgram = createProgram(baseVertex, gradientSubtractShader);
    clearProgram = createProgram(baseVertex, clearShader);
    displayProgram = createProgram(baseVertex, displayShader);

    // full-screen quad
    if (vao) gl.deleteVertexArray(vao);
    if (quadBuffer) gl.deleteBuffer(quadBuffer);
    if (indexBuffer) gl.deleteBuffer(indexBuffer);
    vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    quadBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]),
      gl.STATIC_DRAW,
    );
    indexBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(
      gl.ELEMENT_ARRAY_BUFFER,
      new Uint16Array([0, 1, 2, 0, 2, 3]),
      gl.STATIC_DRAW,
    );
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    resizeIfNeeded();
    initFramebuffers();
    return true;
  };

  // ---- simulation passes -------------------------------------------------
  const correctRadius = (radius: number) => {
    const aspectRatio = canvas.width / canvas.height;
    return aspectRatio > 1 ? radius * aspectRatio : radius;
  };

  const splat = (
    x: number,
    y: number,
    dx: number,
    dy: number,
    amount: number,
    radius: number,
  ) => {
    if (!splatProgram || !velocity || !dye) return;
    gl.useProgram(splatProgram.program);
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(
      splatProgram.uniforms.aspectRatio,
      canvas.width / canvas.height,
    );
    gl.uniform2f(splatProgram.uniforms.point, x, y);
    gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0);
    gl.uniform1f(splatProgram.uniforms.radius, correctRadius(radius));
    blit(velocity.write);
    velocity.swap();

    gl.uniform1i(splatProgram.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(splatProgram.uniforms.color, amount, amount, amount);
    blit(dye.write);
    dye.swap();
  };

  const step = (dt: number) => {
    if (
      !curlProgram ||
      !vorticityProgram ||
      !divergenceProgram ||
      !clearProgram ||
      !pressureProgram ||
      !gradientProgram ||
      !advectionProgram ||
      !velocity ||
      !dye ||
      !divergence ||
      !curl ||
      !pressure
    )
      return;

    gl.disable(gl.BLEND);

    // curl
    gl.useProgram(curlProgram.program);
    gl.uniform2f(
      curlProgram.uniforms.texelSize,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(curl);

    // vorticity confinement
    gl.useProgram(vorticityProgram.program);
    gl.uniform2f(
      vorticityProgram.uniforms.texelSize,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl.attach(1));
    gl.uniform1f(vorticityProgram.uniforms.curl, CURL);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();

    // divergence
    gl.useProgram(divergenceProgram.program);
    gl.uniform2f(
      divergenceProgram.uniforms.texelSize,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence);

    // clear pressure
    gl.useProgram(clearProgram.program);
    gl.uniform1i(clearProgram.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, PRESSURE_DISSIPATION);
    blit(pressure.write);
    pressure.swap();

    // pressure solve (Jacobi)
    gl.useProgram(pressureProgram.program);
    gl.uniform2f(
      pressureProgram.uniforms.texelSize,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence.attach(0));
    for (let i = 0; i < cfg.pressureIterations; i++) {
      gl.uniform1i(pressureProgram.uniforms.uPressure, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    // subtract pressure gradient
    gl.useProgram(gradientProgram.program);
    gl.uniform2f(
      gradientProgram.uniforms.texelSize,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(gradientProgram.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(gradientProgram.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    // advect velocity
    gl.useProgram(advectionProgram.program);
    gl.uniform2f(
      advectionProgram.uniforms.texelSize,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform2f(
      advectionProgram.uniforms.uTexelSize,
      velocity.texelSizeX,
      velocity.texelSizeY,
    );
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, velocity.read.attach(0));
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, VELOCITY_DISSIPATION);
    gl.uniform1f(advectionProgram.uniforms.drift, 0);
    gl.uniform1f(advectionProgram.uniforms.edgeDrain, 0);
    blit(velocity.write);
    velocity.swap();

    // advect dye (with outward drift + edge drain)
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(advectionProgram.uniforms.dissipation, DENSITY_DISSIPATION);
    gl.uniform1f(advectionProgram.uniforms.drift, OUTWARD_DRIFT);
    gl.uniform1f(advectionProgram.uniforms.edgeDrain, EDGE_DRAIN);
    blit(dye.write);
    dye.swap();
  };

  const render = () => {
    if (!displayProgram || !dye) return;
    gl.useProgram(displayProgram.program);
    gl.uniform1i(displayProgram.uniforms.uTexture, dye.read.attach(0));
    blit(null);
  };

  const dispose = () => {
    if (!gl.isContextLost()) {
      deletePrograms();
      if (dye) {
        deleteFBO(dye.read);
        deleteFBO(dye.write);
      }
      if (velocity) {
        deleteFBO(velocity.read);
        deleteFBO(velocity.write);
      }
      deleteFBO(divergence);
      deleteFBO(curl);
      if (pressure) {
        deleteFBO(pressure.read);
        deleteFBO(pressure.write);
      }
      if (quadBuffer) gl.deleteBuffer(quadBuffer);
      if (indexBuffer) gl.deleteBuffer(indexBuffer);
      if (vao) gl.deleteVertexArray(vao);
    }
    // Explicitly release the context so we don't leak it into Safari's limited
    // pool of live contexts (a cause of born-lost contexts on iOS).
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  };

  return {
    canvas,
    resizeIfNeeded,
    initFramebuffers,
    rebuildResources,
    splat,
    step,
    render,
    dispose,
  };
}

export default function FluidCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const host = hostRef.current;
    if (!container || !host) return;
    // Stable non-null bindings (these refs never change for the component's
    // life) so TS keeps the narrowing inside the nested acquire/loop closures.
    const containerEl: HTMLDivElement = container;
    const hostEl: HTMLDivElement = host;

    // ---- device-aware budgets --------------------------------------------
    // Mobile GPUs have far less headroom and aggressively drop the WebGL
    // context under memory pressure. Smaller grids + a lower DPR cap + fewer
    // pressure iterations keep us well within budget and make born-lost
    // contexts less likely in the first place.
    const isMobile =
      (typeof navigator !== "undefined" &&
        ((navigator.maxTouchPoints || 0) > 0 ||
          /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent))) ||
      (typeof matchMedia !== "undefined" &&
        matchMedia("(pointer: coarse)").matches);

    const cfg: SimConfig = {
      dyeResolution: isMobile ? 224 : DYE_RESOLUTION,
      simResolution: isMobile ? 96 : SIM_RESOLUTION,
      pressureIterations: isMobile ? 14 : PRESSURE_ITERATIONS,
      dprCap: isMobile ? 1.25 : 2,
    };

    const debugStill =
      typeof window !== "undefined" &&
      window.location.search.includes("still");

    // ---- input state (persists across context recreation) ----------------
    const pointer = { x: 0.5, y: 0.5, dx: 0, dy: 0, active: false };
    const splatQueue: {
      x: number;
      y: number;
      dx: number;
      dy: number;
      amount: number;
      radius: number;
    }[] = [];

    // Pointer is mapped against the (full-screen) container so it stays valid
    // even as the underlying canvas element is replaced on context recreation.
    const updatePointer = (clientX: number, clientY: number) => {
      const rect = containerEl.getBoundingClientRect();
      const x = (clientX - rect.left) / rect.width;
      const y = 1 - (clientY - rect.top) / rect.height;
      pointer.dx = x - pointer.x;
      pointer.dy = y - pointer.y;
      pointer.x = x;
      pointer.y = y;
      if (
        pointer.active &&
        (Math.abs(pointer.dx) > 0 || Math.abs(pointer.dy) > 0)
      ) {
        splatQueue.push({
          x,
          y,
          dx: pointer.dx * SPLAT_FORCE,
          dy: pointer.dy * SPLAT_FORCE,
          amount: INK_STRENGTH,
          radius: SPLAT_RADIUS,
        });
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!pointer.active) {
        const rect = containerEl.getBoundingClientRect();
        pointer.x = (e.clientX - rect.left) / rect.width;
        pointer.y = 1 - (e.clientY - rect.top) / rect.height;
        pointer.active = true;
        return;
      }
      updatePointer(e.clientX, e.clientY);
    };

    const onPointerDown = (e: PointerEvent) => {
      const rect = containerEl.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = 1 - (e.clientY - rect.top) / rect.height;
      pointer.x = x;
      pointer.y = y;
      pointer.active = true;
      splatQueue.push({
        x,
        y,
        dx: 0,
        dy: 0,
        amount: INK_STRENGTH * 1.6,
        radius: SPLAT_RADIUS * 1.5,
      });
    };

    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerdown", onPointerDown, { passive: true });

    // entrance bloom: an azure flourish that blooms across the tank and then
    // gets sucked toward the edges, so the page never loads empty.
    const introBloom = (canvas: HTMLCanvasElement) => {
      const cx = 0.5;
      const cy = 0.5;
      const aspect = canvas.width / canvas.height || 1;
      splatQueue.push({
        x: cx,
        y: cy,
        dx: 0,
        dy: 0,
        amount: INK_STRENGTH * 1.4,
        radius: SPLAT_RADIUS * 3.5,
      });
      const points = 10;
      for (let i = 0; i < points; i++) {
        const a = (i / points) * Math.PI * 2 + 0.3;
        const r = 0.26;
        splatQueue.push({
          x: cx + Math.cos(a) * r * (aspect > 1 ? 1 : 1 / aspect),
          y: cy + Math.sin(a) * r * (aspect > 1 ? aspect : 1),
          dx: Math.cos(a) * 480,
          dy: Math.sin(a) * 480,
          amount: INK_STRENGTH * 1.1,
          radius: SPLAT_RADIUS * 2.2,
        });
      }
    };

    // ---- lifecycle state --------------------------------------------------
    let sim: Simulation | null = null;
    let running = false;
    let raf = 0;
    let lastTime = performance.now();
    let firstFramePainted = false;

    let disposed = false;
    let fellBack = false;
    let acquireAttempts = 0;
    let acquireTimer: ReturnType<typeof setTimeout> | undefined;
    let restoreWatchdog: ReturnType<typeof setTimeout> | undefined;

    // handlers attached to the *current* canvas; tracked so we can detach
    let activeCanvas: HTMLCanvasElement | null = null;
    let onCtxLost: ((e: Event) => void) | null = null;
    let onCtxRestored: (() => void) | null = null;

    const revealCanvas = (canvas: HTMLCanvasElement) => {
      canvas.style.opacity = "1";
    };

    const frame = () => {
      if (!running || !sim) return;
      if (sim.resizeIfNeeded()) sim.initFramebuffers();

      const now = performance.now();
      const dt = Math.min((now - lastTime) / 1000, 1 / 60);
      lastTime = now;

      while (splatQueue.length > 0) {
        const s = splatQueue.shift()!;
        sim.splat(s.x, s.y, s.dx, s.dy, s.amount, s.radius);
      }

      sim.step(dt);
      sim.render();

      if (!firstFramePainted) {
        firstFramePainted = true;
        revealCanvas(sim.canvas);
      }
      raf = requestAnimationFrame(frame);
    };

    const startLoop = () => {
      running = true;
      lastTime = performance.now();
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    };

    const detachContextHandlers = () => {
      if (activeCanvas && onCtxLost)
        activeCanvas.removeEventListener("webglcontextlost", onCtxLost);
      if (activeCanvas && onCtxRestored)
        activeCanvas.removeEventListener(
          "webglcontextrestored",
          onCtxRestored,
        );
      onCtxLost = null;
      onCtxRestored = null;
    };

    const teardownSession = () => {
      running = false;
      cancelAnimationFrame(raf);
      clearTimeout(restoreWatchdog);
      detachContextHandlers();
      if (sim) {
        sim.dispose();
        sim = null;
      }
      activeCanvas = null;
    };

    const goToFallback = (reason: string) => {
      if (fellBack || disposed) return;
      fellBack = true;
      console.warn("[fluid] falling back to CSS background:", reason);
      teardownSession();
      hostEl.replaceChildren(); // drop the canvas; the gradient shows through
      containerEl.dataset.fluid = "fallback";
    };

    const scheduleRetry = (reason: string) => {
      teardownSession();
      if (disposed || fellBack) return;
      acquireAttempts += 1;
      if (acquireAttempts >= MAX_ACQUIRE_ATTEMPTS) {
        goToFallback(`${reason} (out of attempts)`);
        return;
      }
      const delay = Math.min(300 * 2 ** (acquireAttempts - 1), 4000);
      console.warn(
        `[fluid] ${reason}; recreating context in ${delay}ms (attempt ${acquireAttempts}/${MAX_ACQUIRE_ATTEMPTS})`,
      );
      acquireTimer = setTimeout(acquire, delay);
    };

    // Create a brand-new canvas + context and, if it's healthy, build the sim.
    // A fresh canvas is essential: once a canvas yields a (born-)lost context,
    // getContext keeps returning that same dead object forever.
    function acquire() {
      if (disposed || fellBack || sim) return;
      clearTimeout(acquireTimer);

      // Don't burn an attempt while the tab is hidden — wait for visibility.
      if (typeof document !== "undefined" && document.hidden) return;

      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-hidden", "true");
      canvas.style.position = "absolute";
      canvas.style.inset = "0";
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      canvas.style.opacity = debugStill ? "1" : "0";
      canvas.style.transition = "opacity 800ms ease";
      hostEl.replaceChildren(canvas);
      activeCanvas = canvas;
      firstFramePainted = false;

      const gl = canvas.getContext("webgl2", {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
        powerPreference: "low-power",
        failIfMajorPerformanceCaveat: false,
      });

      if (!gl) {
        // No WebGL2 at all on this device — the CSS fallback is the right home.
        goToFallback("webgl2 unavailable");
        return;
      }

      console.log(
        "[fluid] webgl2 context created; isContextLost =",
        gl.isContextLost(),
      );

      if (gl.isContextLost()) {
        // Born lost. Throw the canvas away and try again on a fresh one.
        scheduleRetry("context born lost");
        return;
      }

      // Healthy context: wire up loss/restore handling for THIS canvas.
      onCtxLost = (e: Event) => {
        e.preventDefault(); // tell the browser we intend to restore
        console.warn("[fluid] webglcontextlost event fired");
        running = false;
        cancelAnimationFrame(raf);
        // A lost opaque canvas has undefined contents (often black). Fade it out
        // so the gradient shows during the gap, and re-arm the fade-in so the
        // restored simulation eases back over it instead of popping.
        canvas.style.opacity = "0";
        firstFramePainted = false;
        // Give the browser a chance to restore; if it doesn't, recreate.
        clearTimeout(restoreWatchdog);
        restoreWatchdog = setTimeout(() => {
          console.warn("[fluid] restore timed out; recreating context");
          scheduleRetry("context lost (no restore)");
        }, RESTORE_TIMEOUT_MS);
      };
      onCtxRestored = () => {
        console.log("[fluid] webglcontextrestored event fired");
        clearTimeout(restoreWatchdog);
        if (disposed || fellBack || !sim) return;
        if (!sim.rebuildResources()) {
          scheduleRetry("rebuild after restore failed");
          return;
        }
        introBloom(sim.canvas);
        startLoop();
      };
      canvas.addEventListener("webglcontextlost", onCtxLost);
      canvas.addEventListener("webglcontextrestored", onCtxRestored);

      const built = createSimulation(canvas, gl, cfg);
      if (!built.rebuildResources()) {
        // Formats unresolvable on a "healthy" context → treat as unusable.
        scheduleRetry("formats unresolvable");
        return;
      }
      sim = built;
      acquireAttempts = 0; // success resets the budget for any future losses

      introBloom(canvas);

      if (debugStill) {
        // Deterministic single render for headless verification.
        const params = new URLSearchParams(window.location.search);
        const steps = parseInt(params.get("steps") || "6", 10);
        sim.splat(0.5, 0.5, 0, 0, 0.8, SPLAT_RADIUS * 5);
        for (let i = 0; i < steps; i++) sim.step(1 / 60);
        sim.render();
        revealCanvas(canvas);
        console.log("[fluid] still frame rendered");
        return;
      }

      startLoop();
      console.log("[fluid] simulation started");
    }

    // Defer the first acquisition until the page has loaded and the main thread
    // is idle, so we aren't competing for the GPU during hydration (a common
    // trigger for born-lost contexts on iOS).
    const idle = (cb: () => void) => {
      const ric = (
        window as unknown as {
          requestIdleCallback?: (
            cb: () => void,
            opts?: { timeout: number },
          ) => number;
        }
      ).requestIdleCallback;
      if (ric) ric(cb, { timeout: 1200 });
      else setTimeout(cb, 200);
    };

    const kickoff = () => {
      if (disposed || fellBack || sim) return;
      if (typeof document !== "undefined" && document.hidden) return; // wait for visible
      idle(() => {
        if (!disposed && !fellBack && !sim) acquire();
      });
    };

    if (debugStill) {
      acquire();
    } else if (document.readyState === "complete") {
      kickoff();
    } else {
      window.addEventListener("load", kickoff, { once: true });
    }

    const onVisibility = () => {
      if (document.hidden) {
        running = false;
        cancelAnimationFrame(raf);
      } else if (!sim && !fellBack && !disposed) {
        // Became visible and we still have no simulation — (re)start acquisition.
        kickoff();
      } else if (sim && !running) {
        startLoop();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    // ---- cleanup ----------------------------------------------------------
    return () => {
      disposed = true;
      clearTimeout(acquireTimer);
      clearTimeout(restoreWatchdog);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("load", kickoff);
      document.removeEventListener("visibilitychange", onVisibility);
      teardownSession();
      hostEl.replaceChildren();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 h-full w-full"
    >
      <div className="fluid-fallback absolute inset-0" />
      <div ref={hostRef} className="absolute inset-0" />
    </div>
  );
}
