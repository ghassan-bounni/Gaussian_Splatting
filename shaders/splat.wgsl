/* ─────────────────────────────────────────────────────────────
   3D Gaussian Splatting — vertex / fragment shader
   ───────────────────────────────────────────────────────────── */

struct Uniforms {
  viewMatrix : mat4x4<f32>,   // offset  0  (64 bytes)
  projMatrix : mat4x4<f32>,   // offset 64  (64 bytes)
  focal      : vec2<f32>,     // offset 128 (8 bytes)  — focal lengths in pixels
  viewport   : vec2<f32>,     // offset 136 (8 bytes)  — canvas width, height
  cameraPos  : vec3<f32>,     // offset 144 (12 bytes) — world-space eye position
  shDegree   : f32,           // offset 156 (4 bytes)  — SH degree (0–3)
};

@group(0) @binding(0) var<uniform>       uniforms    : Uniforms;
@group(0) @binding(1) var<storage, read> positions   : array<f32>;  // N × 3
@group(0) @binding(2) var<storage, read> colors      : array<f32>;  // N × 4 (raw DC r,g,b + sigmoid opacity)
@group(0) @binding(3) var<storage, read> cov3d       : array<f32>;  // N × 6 (upper triangle)
@group(0) @binding(4) var<storage, read> sortIndices : array<u32>;  // N     (sorted back-to-front)
@group(0) @binding(5) var<storage, read> shCoeffs    : array<f32>;  // N × K (higher-order SH, interleaved RGB)

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) vOffset  : vec2<f32>,  // pixel offset from Gaussian center
  @location(1) vColor   : vec3<f32>,
  @location(2) vOpacity : f32,
  @location(3) vConic   : vec3<f32>,  // inverse 2D cov: (a, b, c) for [[a,b],[b,c]]
};

/* ── SH constants (must match the CUDA 3DGS rasterizer exactly) ── */
const SH_C0 : f32 = 0.28209479177387814;
const SH_C1 : f32 = 0.4886025119029199;
const SH_C2_0 : f32 = 1.0925484305920792;
const SH_C2_1 : f32 = 1.0925484305920792;
const SH_C2_2 : f32 = 0.31539156525252005;
const SH_C2_3 : f32 = 1.0925484305920792;
const SH_C2_4 : f32 = 0.5462742152960396;
const SH_C3_0 : f32 = 0.5900435899266435;
const SH_C3_1 : f32 = 2.890611442640554;
const SH_C3_2 : f32 = 0.4570457994644658;
const SH_C3_3 : f32 = 0.3731763325901154;
const SH_C3_4 : f32 = 0.4570457994644658;
const SH_C3_5 : f32 = 1.445305721320277;
const SH_C3_6 : f32 = 0.5900435899266435;

/* ── quad corners for a triangle-strip ── */
const quadPos = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0,  1.0),
);

/* ── Evaluate Spherical Harmonics color ── */
fn evaluateSH(gIdx : u32, dir : vec3<f32>, deg : u32) -> vec3<f32> {
  // Read raw DC coefficients
  let dc = vec3<f32>(
    colors[gIdx * 4u],
    colors[gIdx * 4u + 1u],
    colors[gIdx * 4u + 2u],
  );
  var result = SH_C0 * dc;

  if (deg < 1u) {
    return result;
  }

  // Direction components (COLMAP space)
  let x = dir.x;
  let y = dir.y;
  let z = dir.z;

  // Per-Gaussian stride in the SH buffer: ((deg+1)^2 - 1) * 3 floats
  let shPerCh = (deg + 1u) * (deg + 1u) - 1u;
  let base = gIdx * shPerCh * 3u;

  // Degree 1: 3 coefficients (indices 0, 1, 2)
  let sh1 = vec3<f32>(shCoeffs[base],      shCoeffs[base + 1u],  shCoeffs[base + 2u]);
  let sh2 = vec3<f32>(shCoeffs[base + 3u], shCoeffs[base + 4u],  shCoeffs[base + 5u]);
  let sh3 = vec3<f32>(shCoeffs[base + 6u], shCoeffs[base + 7u],  shCoeffs[base + 8u]);

  result += -SH_C1 * y * sh1 + SH_C1 * z * sh2 - SH_C1 * x * sh3;

  if (deg < 2u) {
    return result;
  }

  // Degree 2: 5 coefficients (indices 3–7)
  let xx = x * x; let yy = y * y; let zz = z * z;
  let xy = x * y; let yz = y * z; let xz = x * z;

  let sh4 = vec3<f32>(shCoeffs[base + 9u],  shCoeffs[base + 10u], shCoeffs[base + 11u]);
  let sh5 = vec3<f32>(shCoeffs[base + 12u], shCoeffs[base + 13u], shCoeffs[base + 14u]);
  let sh6 = vec3<f32>(shCoeffs[base + 15u], shCoeffs[base + 16u], shCoeffs[base + 17u]);
  let sh7 = vec3<f32>(shCoeffs[base + 18u], shCoeffs[base + 19u], shCoeffs[base + 20u]);
  let sh8 = vec3<f32>(shCoeffs[base + 21u], shCoeffs[base + 22u], shCoeffs[base + 23u]);

  result += SH_C2_0 * xy * sh4
          + SH_C2_1 * yz * sh5
          + SH_C2_2 * (2.0 * zz - xx - yy) * sh6
          + SH_C2_3 * xz * sh7
          + SH_C2_4 * (xx - yy) * sh8;

  if (deg < 3u) {
    return result;
  }

  // Degree 3: 7 coefficients (indices 8–14)
  let sh9  = vec3<f32>(shCoeffs[base + 24u], shCoeffs[base + 25u], shCoeffs[base + 26u]);
  let sh10 = vec3<f32>(shCoeffs[base + 27u], shCoeffs[base + 28u], shCoeffs[base + 29u]);
  let sh11 = vec3<f32>(shCoeffs[base + 30u], shCoeffs[base + 31u], shCoeffs[base + 32u]);
  let sh12 = vec3<f32>(shCoeffs[base + 33u], shCoeffs[base + 34u], shCoeffs[base + 35u]);
  let sh13 = vec3<f32>(shCoeffs[base + 36u], shCoeffs[base + 37u], shCoeffs[base + 38u]);
  let sh14 = vec3<f32>(shCoeffs[base + 39u], shCoeffs[base + 40u], shCoeffs[base + 41u]);
  let sh15 = vec3<f32>(shCoeffs[base + 42u], shCoeffs[base + 43u], shCoeffs[base + 44u]);

  result += SH_C3_0 * y * (3.0 * xx - yy) * sh9
          + SH_C3_1 * xy * z * sh10
          + SH_C3_2 * y * (4.0 * zz - xx - yy) * sh11
          + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh12
          + SH_C3_4 * x * (4.0 * zz - xx - yy) * sh13
          + SH_C3_5 * z * (xx - yy) * sh14
          + SH_C3_6 * x * (xx - 3.0 * yy) * sh15;

  return result;
}

@vertex
fn vertexMain(
  @builtin(vertex_index)   vid : u32,
  @builtin(instance_index) iid : u32,
) -> VertexOutput {
  var out : VertexOutput;

  let gIdx = sortIndices[iid];

  /* ── read world position ── */
  let worldPos = vec3<f32>(
    positions[gIdx * 3u],
    positions[gIdx * 3u + 1u],
    positions[gIdx * 3u + 2u],
  );

  /* ── transform to view space ── */
  let viewPos4 = uniforms.viewMatrix * vec4<f32>(worldPos, 1.0);
  let viewPos  = viewPos4.xyz;

  // Depth along camera forward (camera looks -Z ⇒ visible points have viewPos.z < 0)
  let tz = -viewPos.z;

  // Cull points too close to the camera; match official rasterizer near threshold.
  if (tz < 0.2) {
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0); // degenerate — clipped
    out.vOffset  = vec2<f32>(0.0);
    out.vColor   = vec3<f32>(0.0);
    out.vOpacity = 0.0;
    out.vConic   = vec3<f32>(0.0);
    return out;
  }

  /* ── read 3D covariance (upper triangle of symmetric 3×3) ── */
  let c0 = cov3d[gIdx * 6u];
  let c1 = cov3d[gIdx * 6u + 1u];
  let c2 = cov3d[gIdx * 6u + 2u];
  let c3 = cov3d[gIdx * 6u + 3u];
  let c4 = cov3d[gIdx * 6u + 4u];
  let c5 = cov3d[gIdx * 6u + 5u];

  let Vrk = mat3x3<f32>(
    c0, c1, c2,
    c1, c3, c4,
    c2, c4, c5,
  );

  /* ── Jacobian of projection: view → pixel ── */
  let fx = uniforms.focal.x;
  let fy = uniforms.focal.y;
  let tz2 = tz * tz;

  let vx = viewPos.x;
  let vy = viewPos.y;
  let tanFovX = uniforms.viewport.x / (2.0 * fx);
  let tanFovY = uniforms.viewport.y / (2.0 * fy);
  let limX = 1.3 * tanFovX;
  let limY = 1.3 * tanFovY;
  let tx = clamp(vx / tz, -limX, limX);
  let ty = clamp(vy / tz, -limY, limY);

  // IMPORTANT: mat3x3 constructor is column-major in WGSL.
  // This layout gives:
  // [ fx/tz,    0,   -fx*vx/tz2 ]
  // [   0,    fy/tz, -fy*vy/tz2 ]
  // [   0,      0,        0      ]
  let J = mat3x3<f32>(
    fx / tz,               0.0,                  0.0,
    0.0,                   fy / tz,              0.0,
    -fx * tx / tz,        -fy * ty / tz,         0.0,
  );

  /* ── upper-left 3×3 of view matrix (world → view rotation) ── */
  let W = mat3x3<f32>(
    uniforms.viewMatrix[0].xyz,
    uniforms.viewMatrix[1].xyz,
    uniforms.viewMatrix[2].xyz,
  );

  /* ── project covariance to 2D ── */
  let T = J * W;
  let cov2d = T * Vrk * transpose(T);

  // Extract 2×2 and add low-pass filter (anti-aliasing / numerical stability)
  // Force a symmetric 2x2 covariance before inversion to avoid numerical skew.
  let a = cov2d[0][0] + 0.3;
  let bRaw = 0.5 * (cov2d[0][1] + cov2d[1][0]);
  let c = cov2d[1][1] + 0.3;

  // Limit cross-correlation to avoid extremely thin needle-like ellipses.
  let corrLimit = 0.85 * sqrt(max(a * c, 1e-6));
  let b = clamp(bRaw, -corrLimit, corrLimit);

  // Determinant and inverse (conic)
  let det = a * c - b * b;
  if (a <= 0.0 || c <= 0.0 || det <= 1e-6) {
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.vOffset  = vec2<f32>(0.0);
    out.vColor   = vec3<f32>(0.0);
    out.vOpacity = 0.0;
    out.vConic   = vec3<f32>(0.0);
    return out;
  }
  let detInv = 1.0 / det;
  let conic = vec3<f32>(c * detInv, -b * detInv, a * detInv);

  /* ── compute splat radius (3-sigma ellipse) ── */
  let mid  = 0.5 * (a + c);
  let disc = max(0.01, mid * mid - det);
  let lambda1 = mid + sqrt(disc);
  let lambda2 = mid - sqrt(disc);
  let radius = min(ceil(3.0 * sqrt(max(lambda1, lambda2))), 1024.0);

  /* ── project centre to screen pixels ── */
  let clipPos    = uniforms.projMatrix * viewPos4;
  let ndc        = clipPos.xy / clipPos.w;
  let screenCenter = (ndc * vec2<f32>(0.5, -0.5) + 0.5) * uniforms.viewport;

  /* ── position quad vertex ── */
  let corner    = quadPos[vid];
  let offset    = corner * radius;                       // pixel offset
  let pixelPos  = screenCenter + offset;

  // Convert pixel position back to NDC for the rasteriser
  let finalNDC = (pixelPos / uniforms.viewport - 0.5) * vec2<f32>(2.0, -2.0);

  out.position = vec4<f32>(finalNDC, clipPos.z / clipPos.w, 1.0);
  // Negate Y: offset is in screen-pixel space (Y-down) but the 2D covariance
  // was computed via a Jacobian with positive fy, placing it in Y-up projection
  // space.  Flipping d.y ensures the cross-term evaluates with the correct sign.
  out.vOffset  = vec2<f32>(offset.x, -offset.y);

  /* ── evaluate Spherical Harmonics for view-dependent color ── */
  let deg = u32(uniforms.shDegree);
  // Direction from camera to Gaussian in COLMAP space (SH were trained in COLMAP coords).
  // GL world coords are (x, -y, -z) relative to COLMAP, so invert Y and Z of the GL direction.
  let dirGL = normalize(worldPos - uniforms.cameraPos);
  let dirCOLMAP = vec3<f32>(dirGL.x, -dirGL.y, -dirGL.z);
  var color = evaluateSH(gIdx, dirCOLMAP, deg);
  color = clamp(color + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));

  out.vColor   = color;
  out.vOpacity = colors[gIdx * 4u + 3u];
  out.vConic   = conic;

  return out;
}

/* ── fragment: evaluate 2D Gaussian ── */

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4<f32> {
  let d = in.vOffset;
  let power = -0.5 * (in.vConic.x * d.x * d.x
                     + 2.0 * in.vConic.y * d.x * d.y
                     + in.vConic.z * d.y * d.y);

  // Outside the Gaussian kernel — discard
  if (power > 0.0) { discard; }

  let alpha = min(0.99, in.vOpacity * exp(power));

  // Nearly transparent — discard
  if (alpha < 1.0 / 255.0) { discard; }

  // Premultiplied alpha output
  return vec4<f32>(in.vColor * alpha, alpha);
}
