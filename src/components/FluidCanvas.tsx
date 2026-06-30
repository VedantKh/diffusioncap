"use client";

import { useEffect, useRef } from "react";

/**
 * A lightweight, GPU-driven fluid simulation (Navier-Stokes on a grid).
 *
 * Concept: the viewport is an "infinite tank" of clear water. The cursor is the
 * source of blue dye. A persistent outward drift continuously pulls colour
 * toward the edges of the tank, where it drains away — so the centre stays calm
 * and readable while ink keeps streaming outward.
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

export default function FluidCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      console.warn("WebGL2 not available; fluid background disabled.");
      // Avoid leaving an opaque black canvas covering the page.
      canvas.style.display = "none";
      return;
    }

    console.log(
      "[fluid] webgl2 context created; isContextLost =",
      gl.isContextLost(),
    );

    // Float render targets need an explicit color-buffer extension. iOS Safari
    // (and some Android GPUs) only expose the half-float variant, so request
    // both; without one of these, half-float framebuffers are incomplete and
    // every draw silently no-ops, leaving a black canvas.
    gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("EXT_color_buffer_half_float");
    gl.disable(gl.BLEND);

    // Probe whether a given texture format is actually color-renderable on this
    // device, falling back to a wider-channel format when it is not. Many
    // mobile GPUs cannot render to single/dual-channel half-float textures even
    // when the extension is present.
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
    ): { internalFormat: number; format: number } | null => {
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

    const halfFloat = gl.HALF_FLOAT;
    // Formats are resolved lazily: when the context is born lost (common on
    // iOS in Low Power Mode / when too many WebGL contexts are alive), the
    // probe returns nothing, so we wait and resolve again once the context is
    // actually usable instead of permanently disabling the canvas.
    type Fmt = { internalFormat: number; format: number };
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

    // Mobile GPUs have far less headroom and aggressively drop the WebGL
    // context under memory pressure. Use a smaller simulation/dye grid and a
    // lower device-pixel-ratio cap there to stay well within budget.
    const isMobile =
      (typeof navigator !== "undefined" &&
        ((navigator.maxTouchPoints || 0) > 0 ||
          /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent))) ||
      (typeof matchMedia !== "undefined" &&
        matchMedia("(pointer: coarse)").matches);
    const dyeResolution = isMobile ? 256 : DYE_RESOLUTION;
    const simResolution = isMobile ? 96 : SIM_RESOLUTION;
    const dprCap = isMobile ? 1.5 : 2;

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

    // ---- shaders -----------------------------------------------------------
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

    // All GPU resources (programs, buffers, VAO, framebuffers) live in these
    // mutable bindings so they can be rebuilt if the WebGL context is lost and
    // later restored — common on mobile GPUs under memory pressure.
    let splatProgram!: Program;
    let advectionProgram!: Program;
    let divergenceProgram!: Program;
    let curlProgram!: Program;
    let vorticityProgram!: Program;
    let pressureProgram!: Program;
    let gradientProgram!: Program;
    let clearProgram!: Program;
    let displayProgram!: Program;
    let vao!: WebGLVertexArrayObject;
    let quadBuffer!: WebGLBuffer;
    let indexBuffer!: WebGLBuffer;

    const createGLResources = () => {
      splatProgram = createProgram(baseVertex, splatShader);
      advectionProgram = createProgram(baseVertex, advectionShader);
      divergenceProgram = createProgram(baseVertex, divergenceShader);
      curlProgram = createProgram(baseVertex, curlShader);
      vorticityProgram = createProgram(baseVertex, vorticityShader);
      pressureProgram = createProgram(baseVertex, pressureShader);
      gradientProgram = createProgram(baseVertex, gradientSubtractShader);
      clearProgram = createProgram(baseVertex, clearShader);
      displayProgram = createProgram(baseVertex, displayShader);

      // ---- full-screen quad ------------------------------------------------
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

      initFramebuffers();
    };

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

    // ---- framebuffers ------------------------------------------------------
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

    let dye: DoubleFBO;
    let velocity: DoubleFBO;
    let divergence: FBO;
    let curl: FBO;
    let pressure: DoubleFBO;

    const getResolution = (resolution: number) => {
      let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
      if (aspectRatio < 1) aspectRatio = 1 / aspectRatio;
      const min = Math.round(resolution);
      const max = Math.round(resolution * aspectRatio);
      if (gl.drawingBufferWidth > gl.drawingBufferHeight)
        return { width: max, height: min };
      return { width: min, height: max };
    };

    const deleteFBO = (f: FBO) => {
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
      if (divergence) deleteFBO(divergence);
      if (curl) deleteFBO(curl);
      if (pressure) {
        deleteFBO(pressure.read);
        deleteFBO(pressure.write);
      }

      const simRes = getResolution(simResolution);
      const dyeRes = getResolution(dyeResolution);
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

    const resizeCanvas = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, dprCap);
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

    // Resources/loop are created in start(), which only runs once the context
    // is confirmed usable (see below).

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
      for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
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
      gl.useProgram(displayProgram.program);
      gl.uniform1i(displayProgram.uniforms.uTexture, dye.read.attach(0));
      blit(null);
    };

    // ---- pointer input -----------------------------------------------------
    const pointer = {
      x: 0.5,
      y: 0.5,
      dx: 0,
      dy: 0,
      moved: false,
      active: false,
    };
    const splatQueue: {
      x: number;
      y: number;
      dx: number;
      dy: number;
      amount: number;
      radius: number;
    }[] = [];

    const updatePointer = (clientX: number, clientY: number) => {
      const rect = canvas.getBoundingClientRect();
      const x = (clientX - rect.left) / rect.width;
      const y = 1 - (clientY - rect.top) / rect.height;
      pointer.dx = x - pointer.x;
      pointer.dy = y - pointer.y;
      pointer.x = x;
      pointer.y = y;
      if (pointer.active && (Math.abs(pointer.dx) > 0 || Math.abs(pointer.dy) > 0)) {
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
        // first interaction: seed position without a giant splat
        const rect = canvas.getBoundingClientRect();
        pointer.x = (e.clientX - rect.left) / rect.width;
        pointer.y = 1 - (e.clientY - rect.top) / rect.height;
        pointer.active = true;
        return;
      }
      updatePointer(e.clientX, e.clientY);
    };

    const onPointerDown = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
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
    const introBloom = () => {
      const cx = 0.5;
      const cy = 0.5;
      const aspect = canvas.width / canvas.height || 1;
      // a soft core
      splatQueue.push({
        x: cx,
        y: cy,
        dx: 0,
        dy: 0,
        amount: INK_STRENGTH * 1.4,
        radius: SPLAT_RADIUS * 3.5,
      });
      // an outer ring spread wide (scaled by aspect so it reaches past the text)
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

    // ---- main loop ---------------------------------------------------------
    let lastTime = performance.now();
    let rafId = 0;
    let running = true;
    let started = false;

    // Ambient self-driven flow: phones have no cursor, so without this the
    // intro bloom would fade to flat "water" and look static. Trace a slow
    // wandering source that keeps emitting a little ink every frame.
    let ambientAccum = 0;
    const ambientEmit = (now: number, dt: number) => {
      ambientAccum += dt;
      if (ambientAccum < 0.09) return;
      ambientAccum = 0;
      const t = now * 0.001;
      const aspect = canvas.width / canvas.height || 1;
      const sx = aspect > 1 ? 1 : 1 / aspect;
      const sy = aspect > 1 ? aspect : 1;
      const x = 0.5 + 0.26 * Math.sin(t * 0.34) * sx;
      const y = 0.5 + 0.2 * Math.cos(t * 0.47) * sy;
      splatQueue.push({
        x,
        y,
        dx: Math.cos(t * 0.34) * 260,
        dy: -Math.sin(t * 0.47) * 260,
        amount: INK_STRENGTH * 0.55,
        radius: SPLAT_RADIUS * 1.7,
      });
    };

    const frame = () => {
      if (!running) return;
      if (resizeCanvas()) initFramebuffers();

      const now = performance.now();
      const dt = Math.min((now - lastTime) / 1000, 1 / 60);
      lastTime = now;

      ambientEmit(now, dt);

      // drain queued splats
      while (splatQueue.length > 0) {
        const s = splatQueue.shift()!;
        splat(s.x, s.y, s.dx, s.dy, s.amount, s.radius);
      }

      step(dt);
      render();
      rafId = requestAnimationFrame(frame);
    };

    // Build all GPU resources and start the loop. Only succeeds when the
    // context is actually usable; returns false otherwise so the caller can
    // wait for the context to recover.
    const start = () => {
      if (started || gl.isContextLost()) return false;
      if (!resolveFormats()) return false;
      canvas.style.display = "";
      resizeCanvas();
      createGLResources();
      introBloom();
      running = true;
      lastTime = performance.now();
      rafId = requestAnimationFrame(frame);
      started = true;
      console.log("[fluid] simulation started");
      return true;
    };

    // TEMP debug: deterministic single render for headless verification
    const debugStill =
      typeof window !== "undefined" &&
      window.location.search.includes("still");
    if (debugStill) {
      if (resolveFormats()) {
        resizeCanvas();
        createGLResources();
        const params = new URLSearchParams(window.location.search);
        const steps = parseInt(params.get("steps") || "6", 10);
        splat(0.5, 0.5, 0, 0, 0.8, SPLAT_RADIUS * 5);
        for (let i = 0; i < steps; i++) step(1 / 60);
        render();
      }
    } else if (!start()) {
      // Context was born lost (e.g. iOS Low Power Mode / context limit). Keep
      // the canvas hidden and keep retrying; webglcontextrestored (below) will
      // also kick off start() if/when the browser hands us a live context.
      console.warn(
        "[fluid] context not usable yet (isContextLost =",
        gl.isContextLost(),
        "); will retry / await restore",
      );
      canvas.style.display = "none";
      let attempts = 0;
      const retry = () => {
        if (started) return;
        if (start()) return;
        if (++attempts < 10) setTimeout(retry, 400);
        else console.warn("[fluid] gave up after retries; canvas disabled");
      };
      setTimeout(retry, 400);
    }

    const onVisibility = () => {
      if (document.hidden) {
        running = false;
        cancelAnimationFrame(rafId);
      } else if (!running && started && !gl.isContextLost()) {
        running = true;
        lastTime = performance.now();
        rafId = requestAnimationFrame(frame);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    // iOS Safari (and memory-constrained GPUs) can drop the WebGL context at
    // any time. Once lost, every draw silently no-ops and the opaque canvas
    // turns black. Calling preventDefault() in the lost handler tells the
    // browser we intend to restore, then we rebuild every GPU resource in the
    // restored handler so the animation comes back instead of staying black.
    const onContextLost = (e: Event) => {
      e.preventDefault();
      console.warn("[fluid] webglcontextlost event fired");
      running = false;
      cancelAnimationFrame(rafId);
    };
    const onContextRestored = () => {
      console.log("[fluid] webglcontextrestored event fired");
      // If we never managed to start (born-lost context), do a full start now.
      if (!started) {
        start();
        return;
      }
      if (!resolveFormats()) return;
      canvas.style.display = "";
      createGLResources();
      running = true;
      lastTime = performance.now();
      introBloom();
      rafId = requestAnimationFrame(frame);
    };
    canvas.addEventListener("webglcontextlost", onContextLost);
    canvas.addEventListener("webglcontextrestored", onContextRestored);

    // ---- cleanup -----------------------------------------------------------
    return () => {
      running = false;
      cancelAnimationFrame(rafId);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("visibilitychange", onVisibility);
      canvas.removeEventListener("webglcontextlost", onContextLost);
      canvas.removeEventListener("webglcontextrestored", onContextRestored);

      // Resources only exist if start() ran; guard so a never-started (born
      // lost) context doesn't throw during cleanup.
      if (started && !gl.isContextLost()) {
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
        ].forEach((p) => gl.deleteProgram(p.program));

        if (dye) {
          deleteFBO(dye.read);
          deleteFBO(dye.write);
        }
        if (velocity) {
          deleteFBO(velocity.read);
          deleteFBO(velocity.write);
        }
        if (divergence) deleteFBO(divergence);
        if (curl) deleteFBO(curl);
        if (pressure) {
          deleteFBO(pressure.read);
          deleteFBO(pressure.write);
        }
        gl.deleteBuffer(quadBuffer);
        gl.deleteBuffer(indexBuffer);
        gl.deleteVertexArray(vao);
      }

      // Explicitly release the WebGL context so we don't leak it into Safari's
      // limited pool of live contexts (a cause of born-lost contexts on iOS).
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 h-full w-full"
    />
  );
}
