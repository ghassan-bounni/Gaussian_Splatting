import { parsePly } from "./plyLoader.js";
import { createSplatRenderer } from "./splatRenderer.js";
import { createHelpers } from "./helpers.js";
import { computeSceneBounds } from "./scene.js";
import {
  updateCameraMatrix,
  initCameraControls,
  updateKeyboardMovement,
  updateCameraInertia,
  setCameraMode,
} from "./controls.js";

async function main() {
  /* ── WebGPU bootstrap ── */
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const canvas = document.getElementById("viewport");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPUAdapter found");

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  device.lost.then((info) => {
    console.error("WebGPU device lost:", info.message);
    setStatus("GPU device lost: " + info.reason);
  });

  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  /* ── create renderer + helpers ── */
  const renderer = await createSplatRenderer(device, format);
  const helpers = await createHelpers(device, format);

  /* ── UI elements ── */
  const modelSelect = document.getElementById("model-select");
  const iterSelect = document.getElementById("iter-select");
  const gaussSlider = document.getElementById("gauss-slider");
  const gaussLabel = document.getElementById("gauss-label");
  const resetViewBtn = document.getElementById("reset-view");
  const resetOriginBtn = document.getElementById("reset-origin");
  const fpsEl = document.getElementById("fps");
  const pointCountEl = document.getElementById("point-count");
  const emptyHintEl = document.getElementById("empty-hint");
  const speedSlider = document.getElementById("speed-slider");
  const speedLabel = document.getElementById("speed-label");
  const fovSlider = document.getElementById("fov-slider");
  const fovLabel = document.getElementById("fov-label");
  const pointSizeSlider = document.getElementById("point-size-slider");
  const pointSizeLabel = document.getElementById("point-size-label");
  const pointSizeField = document.getElementById("point-size-field");
  const toggleUpBtn = document.getElementById("toggle-up-vector");
  const togglePlaneBtn = document.getElementById("toggle-plane");
  const screenshotBtn = document.getElementById("screenshot-btn");
  const fullscreenBtn = document.getElementById("fullscreen-btn");
  const shDegreeGroup = document.getElementById("sh-degree-group");
  const bgGroup = document.getElementById("bg-group");

  const bgColors = {
    black: [0, 0, 0],
    grey: [0, 0, 0],
    white: [1, 1, 1],
  };
  const defaultBg =
    bgGroup.querySelector("button.active")?.dataset.bg || "grey";
  const initialBg = bgColors[defaultBg] || bgColors.grey;
  renderer.setClearColor(initialBg[0], initialBg[1], initialBg[2]);

  function setEmptyHint(visible) {
    if (!emptyHintEl) return;
    emptyHintEl.style.display = visible ? "block" : "none";
  }

  /* ── Panel toggle logic ── */
  const panelLeft = document.getElementById("panel-left");
  const panelRight = document.getElementById("panel-right");
  const toggleLeft = document.getElementById("toggle-left");
  const toggleRight = document.getElementById("toggle-right");
  const closeLeft = document.getElementById("close-left");
  const closeRight = document.getElementById("close-right");

  closeLeft.addEventListener("click", () => {
    panelLeft.classList.add("collapsed");
    toggleLeft.style.display = "flex";
  });
  closeRight.addEventListener("click", () => {
    panelRight.classList.add("collapsed");
    toggleRight.style.display = "flex";
  });
  toggleLeft.addEventListener("click", () => {
    panelLeft.classList.remove("collapsed");
    toggleLeft.style.display = "none";
  });
  toggleRight.addEventListener("click", () => {
    panelRight.classList.remove("collapsed");
    toggleRight.style.display = "none";
  });

  /* ── Resolution handling ── */
  let resScale = 1;

  function applyResolution() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = Math.round(w * resScale * dpr);
    canvas.height = Math.round(h * resScale * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    camera.aspect = canvas.width / canvas.height;
    context.configure({ device, format, alphaMode: "opaque" });
  }

  // Resolution button group
  const resGroup = document.getElementById("res-group");
  resGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    resGroup
      .querySelectorAll("button")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    resScale = parseFloat(btn.dataset.res);
    applyResolution();
  });
  window.addEventListener("resize", applyResolution);

  /* ── Render mode toggle ── */
  const renderModeGroup = document.getElementById("render-mode-group");
  renderModeGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    renderModeGroup
      .querySelectorAll("button")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    renderer.setRenderMode(btn.dataset.mode);
    pointSizeField.style.display = btn.dataset.mode === "points" ? "" : "none";
  });

  /* ── Camera mode toggle ── */
  const camModeGroup = document.getElementById("cam-mode-group");
  const controlsOrbit = document.getElementById("controls-orbit");
  const controlsFps = document.getElementById("controls-fps");

  camModeGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    camModeGroup
      .querySelectorAll("button")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const mode = btn.dataset.cam;
    setCameraMode(camera, canvas, mode);

    controlsOrbit.style.display = mode === "orbit" ? "" : "none";
    controlsFps.style.display = mode === "fps" ? "" : "none";

    document.body.classList.toggle("fps-mode", mode === "fps");
    document.body.classList.toggle("show-fps-hint", mode === "fps");
  });

  // Hide FPS hint once pointer is locked
  document.addEventListener("pointerlockchange", () => {
    if (document.pointerLockElement === canvas) {
      document.body.classList.remove("show-fps-hint");
      document.body.classList.add("fps-locked");
    } else {
      document.body.classList.remove("fps-locked");
      if (camera.mode === "fps") {
        document.body.classList.add("show-fps-hint");
      }
    }
  });

  /* ── Gaussian limit slider ── */
  let maxDraw = Infinity;
  function updateGaussLabel() {
    const val = parseInt(gaussSlider.value);
    const total = renderer.getGaussianCount();
    if (val >= total && total > 0) {
      gaussLabel.textContent = "All";
      maxDraw = Infinity;
    } else {
      gaussLabel.textContent = val.toLocaleString();
      maxDraw = val;
    }
  }
  gaussSlider.addEventListener("input", updateGaussLabel);

  /* ── Point size slider ── */
  pointSizeSlider.addEventListener("input", () => {
    const v = parseFloat(pointSizeSlider.value);
    pointSizeLabel.textContent = v.toFixed(1);
    renderer.setPointSize(v);
  });

  /* ── Speed slider ── */
  speedSlider.addEventListener("input", () => {
    const v = parseFloat(speedSlider.value);
    speedLabel.textContent = v.toFixed(1);
    camera.moveSpeed = v;
    camera.fpsSpeed = v;
  });

  // Sync slider when scroll-wheel changes speed in FPS mode
  canvas.addEventListener("speedchange", (e) => {
    const v = e.detail;
    speedSlider.value = v;
    speedLabel.textContent = v.toFixed(1);
  });

  // Smooth zoom target from scroll
  canvas.addEventListener("zoomchange", (e) => {
    targetRadius = e.detail;
  });

  /* ── FOV slider ── */
  fovSlider.addEventListener("input", () => {
    const v = parseInt(fovSlider.value);
    fovLabel.innerHTML = v + "&deg;";
    camera.fov = (v * Math.PI) / 180;
  });

  /* ── Screenshot ── */
  let screenshotRequested = false;
  screenshotBtn.addEventListener("click", () => {
    screenshotRequested = true;
  });

  /* ── SH degree toggle ── */
  shDegreeGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    shDegreeGroup
      .querySelectorAll("button")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    renderer.setShDegree(parseInt(btn.dataset.sh));
  });

  /* ── Background color toggle ── */
  bgGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    bgGroup
      .querySelectorAll("button")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const c = bgColors[btn.dataset.bg];
    renderer.setClearColor(c[0], c[1], c[2]);
  });

  /* ── Fullscreen ── */
  fullscreenBtn.addEventListener("click", () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen();
    }
  });

  /* ── Smooth scroll zoom ── */
  let targetRadius = null;
  function updateZoomSmooth() {
    if (targetRadius !== null && camera.mode === "orbit") {
      const diff = targetRadius - camera.radius;
      if (Math.abs(diff) < 0.001) {
        camera.radius = targetRadius;
        targetRadius = null;
      } else {
        camera.radius += diff * 0.15;
      }
    }
  }

  /* ── Scene helper toggles ── */
  toggleUpBtn.addEventListener("click", () => {
    toggleUpBtn.classList.toggle("toggled");
    helpers.setShowUpVector(toggleUpBtn.classList.contains("toggled"));
  });
  togglePlaneBtn.addEventListener("click", () => {
    togglePlaneBtn.classList.toggle("toggled");
    helpers.setShowPlane(togglePlaneBtn.classList.contains("toggled"));
  });

  /* ── persistent camera ── */
  const camera = {
    theta: 0.0,
    phi: Math.PI * 0.35,
    radius: 5.0,
    target: [0, 0, 0],
    up: [0, 1, 0],
    orbitForward: [0, 0, 1],
    orbitRight: [1, 0, 0],
    fov: (55 * Math.PI) / 180,
    aspect: 1,
    near: 0.1,
    far: 100.0,
    moveSpeed: 5,
    mvpMatrix: new Float32Array(16),
    mode: "orbit",
  };

  let defaultCam = {
    theta: 0,
    phi: Math.PI * 0.35,
    radius: 5,
    target: [0, 0, 0],
  };

  initCameraControls(canvas, camera);

  /* ── helper: ensure orbit mode ── */
  function ensureOrbitMode() {
    if (camera.mode === "fps") {
      setCameraMode(camera, canvas, "orbit");
      camModeGroup
        .querySelectorAll("button")
        .forEach((b) => b.classList.remove("active"));
      camModeGroup.querySelector('[data-cam="orbit"]').classList.add("active");
      controlsOrbit.style.display = "";
      controlsFps.style.display = "none";
      document.body.classList.remove("fps-mode", "show-fps-hint");
    }
  }

  /* ── Reset View: restore camera angle/distance but keep current target ── */
  resetViewBtn.addEventListener("click", () => {
    ensureOrbitMode();
    camera.theta = defaultCam.theta;
    camera.phi = defaultCam.phi;
    camera.radius = defaultCam.radius;
  });

  /* ── Reset Origin: move target back to scene center ── */
  resetOriginBtn.addEventListener("click", () => {
    ensureOrbitMode();
    camera.target[0] = defaultCam.target[0];
    camera.target[1] = defaultCam.target[1];
    camera.target[2] = defaultCam.target[2];
  });

  let loaded = false;

  /* ── Keyboard shortcuts (global, only when not in FPS pointer lock) ── */
  window.addEventListener("keydown", (e) => {
    // Skip if typing in an input or FPS pointer-locked
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (document.pointerLockElement === canvas) return;

    switch (e.key.toLowerCase()) {
      case "r":
        resetViewBtn.click();
        break;
      case "f":
        fullscreenBtn.click();
        break;
      case "1":
      case "2":
      case "3":
      case "4": {
        const deg = parseInt(e.key) - 1; // 1→0, 2→1, 3→2, 4→3
        const btn = shDegreeGroup.querySelector(`[data-sh="${deg}"]`);
        if (btn) btn.click();
        break;
      }
    }
  });

  /* ── Drag-and-drop PLY loading ── */
  function loadPlyBuffer(buffer, name) {
    loaded = false;
    setEmptyHint(false);
    setStatus(`Parsing ${(buffer.byteLength / 1e6).toFixed(1)} MB...`);
    const data = parsePly(buffer);

    setStatus(`Uploading ${data.count.toLocaleString()} Gaussians...`);
    renderer.uploadGaussians(data);

    gaussSlider.max = data.count;
    gaussSlider.value = data.count;
    updateGaussLabel();
    pointCountEl.textContent = data.count.toLocaleString() + " pts";

    const { center, extent } = computeSceneBounds(data.positions, data.count);
    helpers.setSceneInfo(center, extent);

    const camRadius = extent * 0.6;
    camera.up = [0, 1, 0];
    camera.orbitForward = [0, 0, 1];
    camera.orbitRight = [1, 0, 0];
    camera.target[0] = center[0];
    camera.target[1] = center[1];
    camera.target[2] = center[2];
    camera.radius = camRadius * 1.4;
    camera.far = extent * 4.0;
    camera.near = camRadius * 0.01;
    camera.theta = 0;
    camera.phi = Math.PI * 0.3;
    camera.moveSpeed = 5;
    camera.fpsSpeed = 5;

    defaultCam = {
      theta: camera.theta,
      phi: camera.phi,
      radius: camera.radius,
      target: [center[0], center[1], center[2]],
    };

    // Update SH degree buttons to reflect model's native degree
    const maxSH = renderer.getMaxShDegree();
    shDegreeGroup.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("active", parseInt(b.dataset.sh) === maxSH);
    });

    loaded = true;
    setStatus(`${name}: ${data.count.toLocaleString()} Gaussians loaded`);
  }

  canvas.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  canvas.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file || !file.name.endsWith(".ply")) {
      setStatus("Drop a .ply file to load");
      return;
    }
    setStatus(`Reading ${file.name}...`);
    file
      .arrayBuffer()
      .then((buf) => loadPlyBuffer(buf, file.name))
      .catch((err) => {
        setStatus("Error: " + err.message);
        console.error(err);
      });
  });

  /* ── apply initial resolution ── */
  applyResolution();

  /* ── load model ── */

  async function loadModel() {
    const model = modelSelect.value;
    const iter = iterSelect.value;

    if (model === "none") {
      loaded = false;
      renderer.clearScene();
      setEmptyHint(true);
      pointCountEl.textContent = "0 pts";
      gaussLabel.textContent = "All";
      setStatus("Select a model and iteration to load");
      return;
    }

    const url = `models/${model}/point_cloud/iteration_${iter}/point_cloud.ply`;

    loaded = false;
    setEmptyHint(false);
    setStatus(`Loading ${model} (iter ${iter})...`);

    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buffer = await resp.arrayBuffer();

      setStatus(`Parsing ${(buffer.byteLength / 1e6).toFixed(1)} MB...`);
      const data = parsePly(buffer);

      setStatus(`Uploading ${data.count.toLocaleString()} Gaussians...`);
      renderer.uploadGaussians(data);

      // Update slider range
      gaussSlider.max = data.count;
      gaussSlider.value = data.count;
      updateGaussLabel();
      pointCountEl.textContent = data.count.toLocaleString() + " pts";

      // Compute scene bounds
      const { center, extent } = computeSceneBounds(data.positions, data.count);
      helpers.setSceneInfo(center, extent);

      // Load training cameras for starting radius
      let camRadius = extent * 0.15;
      try {
        const camResp = await fetch(`models/${model}/cameras.json`);
        if (camResp.ok) {
          const cams = await camResp.json();
          let totalDist = 0;
          for (const cam of cams) {
            const cx = cam.position[0] - center[0];
            const cy = -cam.position[1] - center[1];
            const cz = -cam.position[2] - center[2];
            totalDist += Math.sqrt(cx * cx + cy * cy + cz * cz);
          }
          camRadius = totalDist / cams.length;

          // Log estimated scene up for diagnostics, but always use Y-up
          const estimatedUp = estimateSceneUp(cams);
          console.log(
            `[GS] Estimated scene up: [${estimatedUp[0].toFixed(3)}, ${estimatedUp[1].toFixed(3)}, ${estimatedUp[2].toFixed(3)}]`,
          );

          // Always Y-up: fixed orbit frame
          camera.up = [0, 1, 0];
          camera.orbitForward = [0, 0, 1];
          camera.orbitRight = [1, 0, 0];
        }
      } catch (_) {
        /* use fallback */
      }

      camera.target[0] = center[0];
      camera.target[1] = center[1];
      camera.target[2] = center[2];
      camera.radius = camRadius * 1.4;
      camera.far = extent * 4.0;
      camera.near = camRadius * 0.01;
      camera.theta = 0.0;
      camera.phi = Math.PI * 0.3;
      camera.moveSpeed = 5;
      camera.fpsSpeed = 5;

      // Update speed slider to match
      speedSlider.value = camera.moveSpeed;
      speedLabel.textContent = camera.moveSpeed.toFixed(1);
      fovSlider.value = 55;
      fovLabel.innerHTML = "55&deg;";

      // If in FPS mode, recalculate FPS eye
      if (camera.mode === "fps") {
        setCameraMode(camera, canvas, "fps");
      }

      defaultCam = {
        theta: camera.theta,
        phi: camera.phi,
        radius: camera.radius,
        target: [center[0], center[1], center[2]],
      };

      // Update SH degree buttons
      const maxSH = renderer.getMaxShDegree();
      shDegreeGroup.querySelectorAll("button").forEach((b) => {
        b.classList.toggle("active", parseInt(b.dataset.sh) === maxSH);
      });
      renderer.setShDegree(maxSH);

      loaded = true;
      setEmptyHint(false);
      setStatus(`${data.count.toLocaleString()} Gaussians loaded`);
    } catch (err) {
      setEmptyHint(true);
      setStatus("Error: " + err.message);
      console.error(err);
    }
  }

  modelSelect.addEventListener("change", loadModel);
  iterSelect.addEventListener("change", loadModel);

  setStatus("Select a model and iteration to load");
  pointCountEl.textContent = "0 pts";
  setEmptyHint(true);

  /* ── FPS counter ── */
  let frameCount = 0;
  let lastFpsTime = performance.now();

  function updateFps() {
    frameCount++;
    const now = performance.now();
    const elapsed = now - lastFpsTime;
    if (elapsed >= 500) {
      const fps = ((frameCount / elapsed) * 1000).toFixed(0);
      fpsEl.textContent = fps + " FPS";
      frameCount = 0;
      lastFpsTime = now;
    }
  }

  /* ── render loop ── */
  let lastFrameTime = performance.now();
  function frame() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastFrameTime) / 1000);
    lastFrameTime = now;

    if (loaded) {
      updateKeyboardMovement(camera, dt);
      updateCameraInertia(camera);
      updateZoomSmooth();
    }

    updateCameraMatrix(camera);
    renderer.render(
      context,
      camera.viewMatrix,
      camera.projMatrix,
      canvas.width,
      canvas.height,
      maxDraw,
      camera.eye,
    );

    if (loaded) {
      helpers.render(context, camera.viewMatrix, camera.projMatrix);

      if (screenshotRequested) {
        screenshotRequested = false;
        canvas.toBlob((blob) => {
          if (!blob) return;
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = `screenshot_${Date.now()}.png`;
          a.click();
          URL.revokeObjectURL(a.href);
        }, "image/png");
      }

      updateFps();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* ── helpers ── */

function estimateSceneUp(cams) {
  let ux = 0,
    uy = 0,
    uz = 0;
  for (const cam of cams) {
    const r = cam.rotation;
    const upColmap = [-r[0][1], -r[1][1], -r[2][1]];
    ux += upColmap[0];
    uy += -upColmap[1];
    uz += -upColmap[2];
  }
  const len = Math.hypot(ux, uy, uz);
  if (len < 1e-6) return [0, 1, 0];
  return [ux / len, uy / len, uz / len];
}

function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
  console.log("[GS]", msg);
}

main();
