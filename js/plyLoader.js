/**
 * Parse a binary PLY file from the 3D Gaussian Splatting pipeline.
 * Returns pre-processed Gaussian data ready for rendering.
 *
 * @param {ArrayBuffer} buffer — raw file contents
 * @returns {{ count, positions, colors, cov3d, shDegree, shCoeffs }}
 */
export function parsePly(buffer) {
  /* ---------- 1. Parse header ---------- */
  const headerEnd = findHeaderEnd(buffer);
  const headerText = new TextDecoder().decode(
    new Uint8Array(buffer, 0, headerEnd),
  );
  const lines = headerText.split("\n");

  let vertexCount = 0;
  const properties = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "element" && parts[1] === "vertex") {
      vertexCount = parseInt(parts[2], 10);
    } else if (parts[0] === "property") {
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  if (vertexCount === 0) throw new Error("PLY: no vertices found");

  // Build property-name → index map
  const propIdx = {};
  properties.forEach((p, i) => {
    propIdx[p.name] = i;
  });

  const floatsPerVertex = properties.length; // all properties are float32

  /* ---------- 2. Read binary body ---------- */
  const dataOffset = headerEnd;
  const raw = new Float32Array(
    buffer,
    dataOffset,
    vertexCount * floatsPerVertex,
  );

  /* ---------- 3. Determine SH degree ---------- */
  let shRestCount = 0;
  while (propIdx["f_rest_" + shRestCount] !== undefined) {
    shRestCount++;
  }
  const shDim = Math.floor(shRestCount / 3); // higher-order coefficients per channel

  let shDegree;
  if (shDim < 3) shDegree = 0;
  else if (shDim < 8) shDegree = 1;
  else if (shDim < 15) shDegree = 2;
  else shDegree = 3;

  // Number of higher-order coefficients per channel for the detected degree
  const shCoeffsPerChannel = (shDegree + 1) * (shDegree + 1) - 1;

  /* ---------- 4. Extract & activate per-Gaussian data ---------- */
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 4); // raw DC (R,G,B) + activated opacity
  const cov3d = new Float32Array(vertexCount * 6);
  const shCoeffs = new Float32Array(vertexCount * shCoeffsPerChannel * 3);

  for (let i = 0; i < vertexCount; i++) {
    const base = i * floatsPerVertex;

    // Position — convert COLMAP (Y-down, Z-forward) → OpenGL (Y-up, Z-backward)
    const x = raw[base + propIdx["x"]];
    const y = -raw[base + propIdx["y"]];
    const z = -raw[base + propIdx["z"]];
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    // Store raw DC coefficients — shader applies SH_C0 * dc + higher-order + 0.5
    colors[i * 4] = raw[base + propIdx["f_dc_0"]];
    colors[i * 4 + 1] = raw[base + propIdx["f_dc_1"]];
    colors[i * 4 + 2] = raw[base + propIdx["f_dc_2"]];

    // Opacity: sigmoid activation
    const rawOpacity = raw[base + propIdx["opacity"]];
    colors[i * 4 + 3] = 1.0 / (1.0 + Math.exp(-rawOpacity));

    // Higher-order SH coefficients — interleaved RGB per coefficient.
    // PLY stores channel-first: f_rest_0..N are R, f_rest_{N+1}..2N are G, etc.
    const shOut = i * shCoeffsPerChannel * 3;
    for (let j = 0; j < shCoeffsPerChannel; j++) {
      shCoeffs[shOut + j * 3] = raw[base + propIdx["f_rest_" + j]];
      shCoeffs[shOut + j * 3 + 1] = raw[base + propIdx["f_rest_" + (j + shDim)]];
      shCoeffs[shOut + j * 3 + 2] = raw[base + propIdx["f_rest_" + (j + 2 * shDim)]];
    }

    // Scale: exp activation
    const sx = Math.exp(raw[base + propIdx["scale_0"]]);
    const sy = Math.exp(raw[base + propIdx["scale_1"]]);
    const sz = Math.exp(raw[base + propIdx["scale_2"]]);

    // Rotation quaternion (w, x, y, z) — normalize
    let qw = raw[base + propIdx["rot_0"]];
    let qx = raw[base + propIdx["rot_1"]];
    let qy = raw[base + propIdx["rot_2"]];
    let qz = raw[base + propIdx["rot_3"]];
    const qlen = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw /= qlen;
    qx /= qlen;
    qy /= qlen;
    qz /= qlen;

    // Build rotation matrix R from quaternion (column-major 3×3)
    const r00 = 1 - 2 * (qy * qy + qz * qz);
    const r01 = 2 * (qx * qy - qw * qz);
    const r02 = 2 * (qx * qz + qw * qy);
    const r10 = 2 * (qx * qy + qw * qz);
    const r11 = 1 - 2 * (qx * qx + qz * qz);
    const r12 = 2 * (qy * qz - qw * qx);
    const r20 = 2 * (qx * qz - qw * qy);
    const r21 = 2 * (qy * qz + qw * qx);
    const r22 = 1 - 2 * (qx * qx + qy * qy);

    // M = R * diag(sx, sy, sz)
    const m00 = r00 * sx,
      m01 = r01 * sy,
      m02 = r02 * sz;
    const m10 = r10 * sx,
      m11 = r11 * sy,
      m12 = r12 * sz;
    const m20 = r20 * sx,
      m21 = r21 * sy,
      m22 = r22 * sz;

    // 3D Covariance Σ = M * Mᵀ  (symmetric, store upper triangle)
    const cxx = m00 * m00 + m01 * m01 + m02 * m02;
    const cxy = m00 * m10 + m01 * m11 + m02 * m12;
    const cxz = m00 * m20 + m01 * m21 + m02 * m22;
    const cyy = m10 * m10 + m11 * m11 + m12 * m12;
    const cyz = m10 * m20 + m11 * m21 + m12 * m22;
    const czz = m20 * m20 + m21 * m21 + m22 * m22;

    // Apply COLMAP→OpenGL transform diag(1,-1,-1):  negate xy and xz cross-terms
    cov3d[i * 6] = cxx;
    cov3d[i * 6 + 1] = -cxy;
    cov3d[i * 6 + 2] = -cxz;
    cov3d[i * 6 + 3] = cyy;
    cov3d[i * 6 + 4] = cyz; // (-1)*(-1) = +1
    cov3d[i * 6 + 5] = czz;
  }

  return { count: vertexCount, positions, colors, cov3d, shDegree, shCoeffs };
}

/* ---- helpers ---- */

function findHeaderEnd(buffer) {
  const bytes = new Uint8Array(buffer);
  const marker = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114]; // "end_header"
  for (let i = 0; i < Math.min(bytes.length, 4096); i++) {
    let match = true;
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      // Skip past "end_header\n"
      let end = i + marker.length;
      if (bytes[end] === 0x0d) end++; // \r
      if (bytes[end] === 0x0a) end++; // \n
      return end;
    }
  }
  throw new Error("PLY: could not find end_header");
}
