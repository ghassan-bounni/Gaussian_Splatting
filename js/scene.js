/**
 * Compute robust scene bounds using mean + standard deviation.
 * Min/max is sensitive to outlier Gaussians; stddev gives a tighter fit.
 */
export function computeSceneBounds(positions, count) {
  let sx = 0, sy = 0, sz = 0;
  for (let i = 0; i < count; i++) {
    sx += positions[i * 3];
    sy += positions[i * 3 + 1];
    sz += positions[i * 3 + 2];
  }
  const cx = sx / count, cy = sy / count, cz = sz / count;

  let vx = 0, vy = 0, vz = 0;
  for (let i = 0; i < count; i++) {
    const dx = positions[i * 3] - cx;
    const dy = positions[i * 3 + 1] - cy;
    const dz = positions[i * 3 + 2] - cz;
    vx += dx * dx;
    vy += dy * dy;
    vz += dz * dz;
  }
  // 2-sigma covers ~95% of Gaussians
  const extent = 2 * Math.sqrt((vx + vy + vz) / count);

  return { center: [cx, cy, cz], extent };
}
