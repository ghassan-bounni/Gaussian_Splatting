/**
 * CPU-side 16-bit radix sort for Gaussian depth ordering.
 * Sorts an index array by associated float depth values.
 * Uses float-to-sortable-uint conversion for correct ordering.
 *
 * @param {Uint32Array} indices  — [0..N-1], will be reordered in-place (back-to-front)
 * @param {Float32Array} depths — one depth per Gaussian (view-space Z, more negative = farther)
 * @param {number} count
 */
export function radixSortByDepth(indices, depths, count) {
  // Convert float depths to sortable uint32 keys.
  // In right-handed view space, z < 0 for visible points (more negative = farther).
  // Ascending radix sort on raw float bits puts most-negative first = back-to-front.
  const DEPTH_QUANT_STEP = 1.0 / 4096.0;
  const keys = new Uint32Array(count);
  const floatBuf = new Float32Array(1);
  const uintView = new Uint32Array(floatBuf.buffer);

  for (let i = 0; i < count; i++) {
    // Quantize to stabilize ordering for nearly-equal depths (reduces shimmer).
    const zq = Math.round(depths[i] / DEPTH_QUANT_STEP) * DEPTH_QUANT_STEP;
    floatBuf[0] = zq; // no negation — ascending sort gives back-to-front
    let bits = uintView[0];
    // Float-to-sortable-uint: if sign bit set, flip all bits; otherwise flip sign bit only
    keys[i] = bits >>> 31 ? ~bits : bits | 0x80000000;
  }

  // 2-pass 16-bit radix sort (sorts full 32-bit key in two passes)
  const RADIX = 65536;
  const tempKeys = new Uint32Array(count);
  const tempIdx = new Uint32Array(count);
  const histogram = new Uint32Array(RADIX);

  // Pass 1: sort by lower 16 bits
  histogram.fill(0);
  for (let i = 0; i < count; i++) histogram[keys[i] & 0xffff]++;
  let sum = 0;
  for (let i = 0; i < RADIX; i++) {
    const c = histogram[i];
    histogram[i] = sum;
    sum += c;
  }
  for (let i = 0; i < count; i++) {
    const k = keys[i] & 0xffff;
    const pos = histogram[k]++;
    tempKeys[pos] = keys[i];
    tempIdx[pos] = indices[i];
  }

  // Pass 2: sort by upper 16 bits
  histogram.fill(0);
  for (let i = 0; i < count; i++) histogram[(tempKeys[i] >>> 16) & 0xffff]++;
  sum = 0;
  for (let i = 0; i < RADIX; i++) {
    const c = histogram[i];
    histogram[i] = sum;
    sum += c;
  }
  for (let i = 0; i < count; i++) {
    const k = (tempKeys[i] >>> 16) & 0xffff;
    const pos = histogram[k]++;
    indices[pos] = tempIdx[i];
  }
}

/**
 * Pre-allocate reusable sort state for a given max count.
 * Avoids GC pressure from allocating each frame.
 */
export function createSorter(maxCount) {
  const indices = new Uint32Array(maxCount);
  const depths = new Float32Array(maxCount);
  for (let i = 0; i < maxCount; i++) indices[i] = i;

  return {
    indices,
    depths,
    /**
     * Compute view-space depths and sort indices back-to-front.
     * @param {Float32Array} positions  — N*3 float array of world positions
     * @param {Float32Array} viewMatrix — 4×4 column-major view matrix
     * @param {number} count
     */
    sort(positions, viewMatrix, count) {
      // Extract the third row of the view matrix (dot with position gives view-space Z)
      const m2 = viewMatrix[2];
      const m6 = viewMatrix[6];
      const m10 = viewMatrix[10];
      const m14 = viewMatrix[14];

      for (let i = 0; i < count; i++) {
        const px = positions[i * 3];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        depths[i] = m2 * px + m6 * py + m10 * pz + m14;
        indices[i] = i;
      }

      radixSortByDepth(indices, depths, count);
    },
  };
}
