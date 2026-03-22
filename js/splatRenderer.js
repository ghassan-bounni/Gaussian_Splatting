import { createSorter } from "./radixSort.js";

/**
 * Creates the WebGPU Gaussian-splat rendering pipeline
 * with support for both splat and point-cloud render modes.
 */
export async function createSplatRenderer(device, canvasFormat) {
  /* ── load shaders ── */
  const [splatResp, pointResp] = await Promise.all([
    fetch("shaders/splat.wgsl"),
    fetch("shaders/point.wgsl"),
  ]);
  if (!splatResp.ok) throw new Error("Failed to load shaders/splat.wgsl");
  if (!pointResp.ok) throw new Error("Failed to load shaders/point.wgsl");

  const splatModule = device.createShaderModule({
    label: "splat shader",
    code: await splatResp.text(),
  });
  const pointModule = device.createShaderModule({
    label: "point shader",
    code: await pointResp.text(),
  });

  /* ── shared bind group layout (uniform + 5 storage) ── */
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  /* ── blend state (shared by both pipelines) ── */
  const blendState = {
    color: {
      srcFactor: "one",
      dstFactor: "one-minus-src-alpha",
      operation: "add",
    },
    alpha: {
      srcFactor: "one",
      dstFactor: "one-minus-src-alpha",
      operation: "add",
    },
  };

  /* ── splat pipeline ── */
  const splatPipeline = device.createRenderPipeline({
    label: "splat pipeline",
    layout: pipelineLayout,
    vertex: { module: splatModule, entryPoint: "vertexMain" },
    fragment: {
      module: splatModule,
      entryPoint: "fragmentMain",
      targets: [{ format: canvasFormat, blend: blendState }],
    },
    primitive: { topology: "triangle-strip" },
  });

  /* ── point pipeline ── */
  const pointPipeline = device.createRenderPipeline({
    label: "point pipeline",
    layout: pipelineLayout,
    vertex: { module: pointModule, entryPoint: "vertexMain" },
    fragment: {
      module: pointModule,
      entryPoint: "fragmentMain",
      targets: [{ format: canvasFormat, blend: blendState }],
    },
    primitive: { topology: "triangle-strip" },
  });

  /* ── uniform buffer ── */
  // viewMatrix(16) + projMatrix(16) + focal(2) + viewport(2) + cameraPos(3) + shDegree/pointSize(1) = 40
  const uniformData = new Float32Array(40);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  /* ── per-scene GPU state ── */
  let positionBuffer = null;
  let colorBuffer = null;
  let cov3dBuffer = null;
  let sortBuffer = null;
  let shBuffer = null;
  let bindGroup = null;
  let gaussianCount = 0;
  let sorter = null;
  let gaussianPositions = null;
  let shDegree = 0;

  /* ── render mode state ── */
  let renderMode = "splats"; // "splats" | "points"
  let pointSize = 2.0;

  /* ── sort throttling: skip re-sort when camera hasn't moved ── */
  let lastSortView = new Float32Array(16);

  function createStorageBuffer(data) {
    const buf = device.createBuffer({
      size: (data.byteLength + 3) & ~3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  function uploadGaussians(data) {
    if (positionBuffer) positionBuffer.destroy();
    if (colorBuffer) colorBuffer.destroy();
    if (cov3dBuffer) cov3dBuffer.destroy();
    if (sortBuffer) sortBuffer.destroy();
    if (shBuffer) shBuffer.destroy();

    gaussianCount = data.count;
    gaussianPositions = data.positions;
    shDegree = data.shDegree || 0;

    positionBuffer = createStorageBuffer(data.positions);
    colorBuffer = createStorageBuffer(data.colors);
    cov3dBuffer = createStorageBuffer(data.cov3d);

    if (data.shCoeffs && data.shCoeffs.length > 0) {
      shBuffer = createStorageBuffer(data.shCoeffs);
    } else {
      shBuffer = createStorageBuffer(new Float32Array([0]));
    }

    const initialIndices = new Uint32Array(gaussianCount);
    for (let i = 0; i < gaussianCount; i++) initialIndices[i] = i;
    sortBuffer = createStorageBuffer(initialIndices);

    sorter = createSorter(gaussianCount);
    lastSortView.fill(0); // force re-sort on next render

    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: colorBuffer } },
        { binding: 3, resource: { buffer: cov3dBuffer } },
        { binding: 4, resource: { buffer: sortBuffer } },
        { binding: 5, resource: { buffer: shBuffer } },
      ],
    });
  }

  function clearScene() {
    if (positionBuffer) positionBuffer.destroy();
    if (colorBuffer) colorBuffer.destroy();
    if (cov3dBuffer) cov3dBuffer.destroy();
    if (sortBuffer) sortBuffer.destroy();
    if (shBuffer) shBuffer.destroy();

    positionBuffer = null;
    colorBuffer = null;
    cov3dBuffer = null;
    sortBuffer = null;
    shBuffer = null;
    bindGroup = null;
    gaussianCount = 0;
    sorter = null;
    gaussianPositions = null;
    shDegree = 0;
    shDegreeOverride = null;
    lastSortView.fill(0);
  }

  function getGaussianCount() {
    return gaussianCount;
  }
  function getRenderMode() {
    return renderMode;
  }
  function setRenderMode(mode) {
    renderMode = mode;
  }
  function getPointSize() {
    return pointSize;
  }
  function setPointSize(s) {
    pointSize = s;
  }
  function getMaxShDegree() {
    return shDegree;
  }
  function getShDegree() {
    return shDegreeOverride ?? shDegree;
  }
  function setShDegree(d) {
    shDegreeOverride = d;
  }

  let shDegreeOverride = null; // null = use model's native degree
  let clearColor = { r: 0, g: 0, b: 0, a: 1 };
  function setClearColor(r, g, b) {
    clearColor = { r, g, b, a: 1 };
  }

  /**
   * @param {GPUCanvasContext} context
   * @param {Float32Array} viewMatrix
   * @param {Float32Array} projMatrix
   * @param {number} vpWidth
   * @param {number} vpHeight
   * @param {number} maxDraw
   * @param {number[]} cameraPos — world-space eye [x, y, z]
   */
  function render(
    context,
    viewMatrix,
    projMatrix,
    vpWidth,
    vpHeight,
    maxDraw,
    cameraPos,
  ) {
    if (!bindGroup || gaussianCount === 0) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        label: "clear pass",
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: clearColor,
            storeOp: "store",
          },
        ],
      });
      pass.end();
      device.queue.submit([encoder.finish()]);
      return;
    }

    const drawCount = Math.min(gaussianCount, maxDraw || gaussianCount);

    /* ── CPU sort (skip if camera hasn't moved) ── */
    let needsSort = false;
    for (let i = 0; i < 16; i++) {
      if (Math.abs(viewMatrix[i] - lastSortView[i]) > 1e-6) {
        needsSort = true;
        break;
      }
    }
    if (needsSort) {
      sorter.sort(gaussianPositions, viewMatrix, gaussianCount);
      device.queue.writeBuffer(
        sortBuffer,
        0,
        sorter.indices.buffer,
        0,
        gaussianCount * 4,
      );
      lastSortView.set(viewMatrix);
    }

    /* ── uniforms ── */
    uniformData.set(viewMatrix, 0);
    uniformData.set(projMatrix, 16);
    uniformData[32] = projMatrix[0] * vpWidth * 0.5; // focal x
    uniformData[33] = projMatrix[5] * vpHeight * 0.5; // focal y
    uniformData[34] = vpWidth;
    uniformData[35] = vpHeight;
    uniformData[36] = cameraPos[0];
    uniformData[37] = cameraPos[1];
    uniformData[38] = cameraPos[2];
    // Slot 39: shDegree for splats, pointSize for points
    uniformData[39] =
      renderMode === "points" ? pointSize : (shDegreeOverride ?? shDegree);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    /* ── render pass ── */
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      label: "render pass",
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: clearColor,
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(renderMode === "points" ? pointPipeline : splatPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4, drawCount);
    pass.end();

    device.queue.submit([encoder.finish()]);
  }

  return {
    uploadGaussians,
    render,
    getGaussianCount,
    getRenderMode,
    setRenderMode,
    getPointSize,
    setPointSize,
    getMaxShDegree,
    getShDegree,
    setShDegree,
    setClearColor,
  };
}
