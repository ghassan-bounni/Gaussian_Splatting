/* ─────────────────────────────────────────────────────────────
   Point Cloud rendering — vertex / fragment shader
   Uses the same bind group layout as splat.wgsl so both
   pipelines can share a single bind group.
   ───────────────────────────────────────────────────────────── */

struct Uniforms {
  viewMatrix : mat4x4<f32>,
  projMatrix : mat4x4<f32>,
  focal      : vec2<f32>,
  viewport   : vec2<f32>,
  cameraPos  : vec3<f32>,
  pointSize  : f32,           // reuses the shDegree slot
};

@group(0) @binding(0) var<uniform>       uniforms    : Uniforms;
@group(0) @binding(1) var<storage, read> positions   : array<f32>;
@group(0) @binding(2) var<storage, read> colors      : array<f32>;
@group(0) @binding(3) var<storage, read> cov3d       : array<f32>;   // unused
@group(0) @binding(4) var<storage, read> sortIndices : array<u32>;
@group(0) @binding(5) var<storage, read> shCoeffs    : array<f32>;   // unused

const SH_C0 : f32 = 0.28209479177387814;

const quadPos = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0,  1.0),
);

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) vColor  : vec3<f32>,
  @location(1) vOffset : vec2<f32>,
};

@vertex
fn vertexMain(
  @builtin(vertex_index)   vid : u32,
  @builtin(instance_index) iid : u32,
) -> VertexOutput {
  var out : VertexOutput;

  let gIdx = sortIndices[iid];

  let worldPos = vec3<f32>(
    positions[gIdx * 3u],
    positions[gIdx * 3u + 1u],
    positions[gIdx * 3u + 2u],
  );

  let viewPos4 = uniforms.viewMatrix * vec4<f32>(worldPos, 1.0);
  let tz = -viewPos4.z;

  if (tz < 0.2) {
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.vColor   = vec3<f32>(0.0);
    out.vOffset  = vec2<f32>(0.0);
    return out;
  }

  // DC colour only: SH_C0 * dc + 0.5
  let dc = vec3<f32>(
    colors[gIdx * 4u],
    colors[gIdx * 4u + 1u],
    colors[gIdx * 4u + 2u],
  );
  let color = clamp(SH_C0 * dc + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));

  let clipPos      = uniforms.projMatrix * viewPos4;
  let ndc          = clipPos.xy / clipPos.w;
  let screenCenter = (ndc * vec2<f32>(0.5, -0.5) + 0.5) * uniforms.viewport;

  let corner  = quadPos[vid];
  let radius  = max(uniforms.pointSize, 1.0);
  let offset  = corner * radius;
  let pixelPos = screenCenter + offset;

  let finalNDC = (pixelPos / uniforms.viewport - 0.5) * vec2<f32>(2.0, -2.0);

  out.position = vec4<f32>(finalNDC, clipPos.z / clipPos.w, 1.0);
  out.vColor   = color;
  out.vOffset  = corner;

  return out;
}

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4<f32> {
  // Round point — discard outside unit circle
  let distSq = dot(in.vOffset, in.vOffset);
  if (distSq > 1.0) { discard; }
  return vec4<f32>(in.vColor, 1.0);
}
