/**
 * 3D scene helpers: up-vector arrow + ground-plane grid.
 * Renders as a second pass on top of the splat output.
 */
export async function createHelpers(device, canvasFormat) {
  const resp = await fetch("shaders/helpers.wgsl");
  if (!resp.ok) throw new Error("Failed to load shaders/helpers.wgsl");
  const module = device.createShaderModule({ code: await resp.text() });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: "uniform" },
    }],
  });

  const pipeline = device.createRenderPipeline({
    label: "helpers pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: "vertexMain",
      buffers: [{
        arrayStride: 28, // vec3 pos (12) + vec4 color (16) = 28
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x4" },
        ],
      }],
    },
    fragment: {
      module,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasFormat,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "line-list" },
  });

  const uniformData = new Float32Array(32);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  let vertexBuffer = null;
  let vertexCount = 0;
  let showUpVector = false;
  let showPlane = false;
  let sceneCenter = [0, 0, 0];
  let sceneExtent = 5;

  function setSceneInfo(center, extent) {
    sceneCenter = [center[0], center[1], center[2]];
    sceneExtent = extent;
    rebuildGeometry();
  }

  function setShowUpVector(v) { showUpVector = v; rebuildGeometry(); }
  function setShowPlane(v) { showPlane = v; rebuildGeometry(); }

  // Push a line segment: two vertices, each with pos(3) + color(4)
  function pushLine(verts, x0, y0, z0, x1, y1, z1, r, g, b, a) {
    verts.push(x0, y0, z0, r, g, b, a);
    verts.push(x1, y1, z1, r, g, b, a);
  }

  function rebuildGeometry() {
    const verts = [];
    const cx = sceneCenter[0], cy = sceneCenter[1], cz = sceneCenter[2];

    if (showUpVector) {
      const len = sceneExtent * 0.25;
      const head = len * 0.12;

      // Helper: arrow with 4-fin head along an axis
      function pushArrow(ax, ay, az, r, g, b) {
        // Shaft
        pushLine(verts, cx, cy, cz, cx + ax * len, cy + ay * len, cz + az * len, r, g, b, 1);
        // Arrowhead — 4 fins perpendicular to the axis
        // Pick two perpendicular directions to the axis
        let p1, p2;
        if (Math.abs(ay) > Math.abs(ax) && Math.abs(ay) > Math.abs(az)) {
          p1 = [1, 0, 0]; p2 = [0, 0, 1];
        } else if (Math.abs(ax) > Math.abs(az)) {
          p1 = [0, 1, 0]; p2 = [0, 0, 1];
        } else {
          p1 = [1, 0, 0]; p2 = [0, 1, 0];
        }
        const tx = cx + ax * len, ty = cy + ay * len, tz = cz + az * len;
        const bx = tx - ax * head * 2, by = ty - ay * head * 2, bz = tz - az * head * 2;
        pushLine(verts, tx, ty, tz, bx + p1[0] * head, by + p1[1] * head, bz + p1[2] * head, r, g, b, 1);
        pushLine(verts, tx, ty, tz, bx - p1[0] * head, by - p1[1] * head, bz - p1[2] * head, r, g, b, 1);
        pushLine(verts, tx, ty, tz, bx + p2[0] * head, by + p2[1] * head, bz + p2[2] * head, r, g, b, 1);
        pushLine(verts, tx, ty, tz, bx - p2[0] * head, by - p2[1] * head, bz - p2[2] * head, r, g, b, 1);
      }

      // X axis — red
      pushArrow(1, 0, 0, 1.0, 0.15, 0.15);
      // Y axis — green
      pushArrow(0, 1, 0, 0.15, 1.0, 0.15);
      // Z axis — blue
      pushArrow(0, 0, 1, 0.2, 0.4, 1.0);
    }

    if (showPlane) {
      const gridHalf = sceneExtent * 0.5;
      const divisions = 20;
      const step = (gridHalf * 2) / divisions;

      for (let i = 0; i <= divisions; i++) {
        const t = -gridHalf + i * step;
        const isCenter = Math.abs(t) < step * 0.01;
        // Center lines brighter
        const c = isCenter ? 0.6 : 0.35;
        const a = isCenter ? 0.85 : 0.55;

        // Lines along Z
        pushLine(verts,
          cx + t, cy, cz - gridHalf,
          cx + t, cy, cz + gridHalf,
          c, c, c + 0.02, a);
        // Lines along X
        pushLine(verts,
          cx - gridHalf, cy, cz + t,
          cx + gridHalf, cy, cz + t,
          c, c, c + 0.02, a);
      }
    }

    if (verts.length === 0) {
      vertexCount = 0;
      return;
    }

    const data = new Float32Array(verts);
    vertexCount = data.length / 7; // 7 floats per vertex (pos3 + color4)

    if (vertexBuffer) vertexBuffer.destroy();
    vertexBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, data);
  }

  function render(context, viewMatrix, projMatrix) {
    if (vertexCount === 0 || !vertexBuffer) return;

    uniformData.set(viewMatrix, 0);
    uniformData.set(projMatrix, 16);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      label: "helpers pass",
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "load",
        storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertexCount);
    pass.end();

    device.queue.submit([encoder.finish()]);
  }

  return { setSceneInfo, setShowUpVector, setShowPlane, render };
}
