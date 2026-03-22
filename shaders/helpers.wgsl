/* ─────────────────────────────────────────────────────────────
   Simple vertex-coloured line shader for 3D helpers
   (up-vector arrow, ground-plane grid, etc.)
   ───────────────────────────────────────────────────────────── */

struct Uniforms {
  viewMatrix : mat4x4<f32>,
  projMatrix : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) color : vec4<f32>,
};

@vertex
fn vertexMain(
  @location(0) pos   : vec3<f32>,
  @location(1) color : vec4<f32>,
) -> VertexOutput {
  var out : VertexOutput;
  out.position = uniforms.projMatrix * uniforms.viewMatrix * vec4<f32>(pos, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4<f32> {
  return in.color;
}
