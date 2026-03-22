export function mat4Identity() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

export function mat4Translate(x, y, z) {
  const m = mat4Identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

export function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

export function mat4Invert(a) {
  const out = new Float32Array(16);
  const a00 = a[0],  a01 = a[1],  a02 = a[2],  a03 = a[3];
  const a10 = a[4],  a11 = a[5],  a12 = a[6],  a13 = a[7];
  const a20 = a[8],  a21 = a[9],  a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = a00*a11 - a01*a10, b01 = a00*a12 - a02*a10;
  const b02 = a00*a13 - a03*a10, b03 = a01*a12 - a02*a11;
  const b04 = a01*a13 - a03*a11, b05 = a02*a13 - a03*a12;
  const b06 = a20*a31 - a21*a30, b07 = a20*a32 - a22*a30;
  const b08 = a20*a33 - a23*a30, b09 = a21*a32 - a22*a31;
  const b10 = a21*a33 - a23*a31, b11 = a22*a33 - a23*a32;
  let det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (!det) return null;
  det = 1.0 / det;
  out[0]  = ( a11*b11 - a12*b10 + a13*b09) * det;
  out[1]  = (-a01*b11 + a02*b10 - a03*b09) * det;
  out[2]  = ( a31*b05 - a32*b04 + a33*b03) * det;
  out[3]  = (-a21*b05 + a22*b04 - a23*b03) * det;
  out[4]  = (-a10*b11 + a12*b08 - a13*b07) * det;
  out[5]  = ( a00*b11 - a02*b08 + a03*b07) * det;
  out[6]  = (-a30*b05 + a32*b02 - a33*b01) * det;
  out[7]  = ( a20*b05 - a22*b02 + a23*b01) * det;
  out[8]  = ( a10*b10 - a11*b08 + a13*b06) * det;
  out[9]  = (-a00*b10 + a01*b08 - a03*b06) * det;
  out[10] = ( a30*b04 - a31*b02 + a33*b00) * det;
  out[11] = (-a20*b04 + a21*b02 - a23*b00) * det;
  out[12] = (-a10*b09 + a11*b07 - a12*b06) * det;
  out[13] = ( a00*b09 - a01*b07 + a02*b06) * det;
  out[14] = (-a30*b03 + a31*b01 - a32*b00) * det;
  out[15] = ( a20*b03 - a21*b01 + a22*b00) * det;
  return out;
}

export function mat4LookAt(eye, center, up) {
  const fx = center[0]-eye[0], fy = center[1]-eye[1], fz = center[2]-eye[2];
  let len = Math.sqrt(fx*fx + fy*fy + fz*fz);
  const f = [fx/len, fy/len, fz/len];

  let rx = f[1]*up[2] - f[2]*up[1];
  let ry = f[2]*up[0] - f[0]*up[2];
  let rz = f[0]*up[1] - f[1]*up[0];
  len = Math.sqrt(rx*rx + ry*ry + rz*rz);
  const r = [rx/len, ry/len, rz/len];

  const u = [
    r[1]*f[2] - r[2]*f[1],
    r[2]*f[0] - r[0]*f[2],
    r[0]*f[1] - r[1]*f[0],
  ];

  return new Float32Array([
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    -(r[0]*eye[0] + r[1]*eye[1] + r[2]*eye[2]),
    -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]),
     (f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]),
    1,
  ]);
}

export function mat4Perspective(fovy, aspect, near, far) {
  const f  = 1.0 / Math.tan(fovy * 0.5);
  const nf = 1.0 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0,                  0,
    0,          f, 0,                  0,
    0,          0, far * nf,          -1,
    0,          0, (near * far) * nf,  0,
  ]);
}
