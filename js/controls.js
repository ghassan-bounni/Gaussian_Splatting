import { mat4LookAt, mat4Perspective, mat4Multiply } from "./math.js";

function normalize3(v) {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-8) return [0, 1, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function rejectFromAxis(v, axis) {
  const d = dot3(v, axis);
  return [v[0] - d * axis[0], v[1] - d * axis[1], v[2] - d * axis[2]];
}

function ensureOrbitFrame(camera) {
  const up = normalize3(camera.up || [0, 1, 0]);
  camera.up = up;

  if (!camera.orbitForward || !camera.orbitRight) {
    const seed = Math.abs(up[1]) < 0.99 ? [0, 1, 0] : [0, 0, 1];
    let forward = normalize3(rejectFromAxis(seed, up));
    if (Math.hypot(forward[0], forward[1], forward[2]) < 1e-6) {
      forward = normalize3(rejectFromAxis([1, 0, 0], up));
    }
    const right = normalize3(cross3(forward, up));
    camera.orbitRight = right;
    camera.orbitForward = normalize3(cross3(up, right));
  }
}

function computeEye(camera) {
  ensureOrbitFrame(camera);

  const elevation = Math.PI * 0.5 - camera.phi;
  const c = Math.cos(elevation);
  const s = Math.sin(elevation);
  const sinTheta = Math.sin(camera.theta);
  const cosTheta = Math.cos(camera.theta);

  const offset = [
    c * (sinTheta * camera.orbitRight[0] + cosTheta * camera.orbitForward[0]) +
      s * camera.up[0],
    c * (sinTheta * camera.orbitRight[1] + cosTheta * camera.orbitForward[1]) +
      s * camera.up[1],
    c * (sinTheta * camera.orbitRight[2] + cosTheta * camera.orbitForward[2]) +
      s * camera.up[2],
  ];

  return [
    camera.target[0] + camera.radius * offset[0],
    camera.target[1] + camera.radius * offset[1],
    camera.target[2] + camera.radius * offset[2],
  ];
}

function getViewBasis(camera) {
  const eye = computeEye(camera);
  const forward = normalize3([
    camera.target[0] - eye[0],
    camera.target[1] - eye[1],
    camera.target[2] - eye[2],
  ]);
  const right = normalize3(cross3(forward, camera.up || [0, 1, 0]));
  const camUp = normalize3(cross3(right, forward));
  return { right, camUp, forward };
}

/* ── FPS helpers ── */

function fpsForward(camera) {
  const cp = Math.cos(camera.fpsPitch);
  return [
    Math.sin(camera.fpsYaw) * cp,
    Math.sin(camera.fpsPitch),
    -Math.cos(camera.fpsYaw) * cp,
  ];
}

function fpsRight(camera) {
  return [Math.cos(camera.fpsYaw), 0, Math.sin(camera.fpsYaw)];
}

/* ── Camera mode switching ── */

export function setCameraMode(camera, canvas, mode) {
  const prev = camera.mode || "orbit";

  if (mode === "fps" && prev !== "fps") {
    // Orbit → FPS: place eye at current orbit position, convert angles directly.
    // Works because orbit frame is axis-aligned (Y-up, forward=[0,0,1], right=[1,0,0]).
    camera.phi = Math.max(0.05, Math.min(Math.PI - 0.05, camera.phi));
    const eye = computeEye(camera);
    camera.fpsEye = [eye[0], eye[1], eye[2]];
    camera.fpsYaw = -camera.theta;
    camera.fpsPitch = camera.phi - Math.PI * 0.5;
    camera.fpsSpeed = camera.moveSpeed || 5;
    camera._skipPointerLockFrames = 2; // guard first mousemove after lock
  } else if (mode === "orbit" && prev === "fps") {
    // FPS → Orbit: exit pointer lock, set orbit target ahead of eye
    if (document.pointerLockElement === canvas) {
      document.exitPointerLock();
    }
    // FPS → Orbit: convert angles back, keep axis-aligned orbit frame
    const fwd = fpsForward(camera);
    const dist = camera.radius || 5.0;
    camera.target = [
      camera.fpsEye[0] + fwd[0] * dist,
      camera.fpsEye[1] + fwd[1] * dist,
      camera.fpsEye[2] + fwd[2] * dist,
    ];
    camera.theta = -camera.fpsYaw;
    camera.phi = camera.fpsPitch + Math.PI * 0.5;
    camera.orbitForward = [0, 0, 1];
    camera.orbitRight = [1, 0, 0];
  }

  camera.mode = mode;
}

/* ── Keyboard movement (both modes) ── */

export function updateKeyboardMovement(camera, dtSeconds) {
  // WASD movement is FPS-only
  if (camera.mode !== "fps") return;

  const keys = camera.keys;
  if (!keys) return;

  const mz = (keys.w ? 1 : 0) - (keys.s ? 1 : 0);
  const mx = (keys.d ? 1 : 0) - (keys.a ? 1 : 0);
  const my = (keys.e ? 1 : 0) - (keys.q ? 1 : 0);
  if (mx === 0 && my === 0 && mz === 0) return;

  const fwd = fpsForward(camera);
  const right = fpsRight(camera);
  const speed = (camera.fpsSpeed || 2.5) * dtSeconds;

  camera.fpsEye[0] += (mz * fwd[0] + mx * right[0]) * speed;
  camera.fpsEye[1] += (mz * fwd[1] + my) * speed;
  camera.fpsEye[2] += (mz * fwd[2] + mx * right[2]) * speed;
}

/**
 * Apply and decay orbit inertia. Call once per frame.
 */
export function updateCameraInertia(camera) {
  if (camera._inertia) camera._inertia.update();
}

/**
 * Recomputes camera matrices from its current state.
 * Handles both orbit and FPS modes.
 */
export function updateCameraMatrix(camera) {
  if (camera.mode === "fps") {
    const eye = camera.fpsEye;
    const fwd = fpsForward(camera);
    const target = [eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2]];

    camera.eye = [eye[0], eye[1], eye[2]];
    // Use Y-up for FPS (standard first-person convention)
    camera.viewMatrix = mat4LookAt(eye, target, camera.up || [0, 1, 0]);
    camera.projMatrix = mat4Perspective(
      camera.fov,
      camera.aspect,
      camera.near,
      camera.far,
    );
    camera.mvpMatrix = mat4Multiply(camera.projMatrix, camera.viewMatrix);
    return;
  }

  // Orbit mode (existing)
  camera.phi = Math.max(0.05, Math.min(Math.PI - 0.05, camera.phi));
  const eye = computeEye(camera);
  const view = mat4LookAt(eye, camera.target, camera.up || [0, 1, 0]);
  const proj = mat4Perspective(
    camera.fov,
    camera.aspect,
    camera.near,
    camera.far,
  );

  camera.eye = eye;
  camera.viewMatrix = view;
  camera.projMatrix = proj;
  camera.mvpMatrix = mat4Multiply(proj, view);
}

/**
 * Attaches mouse, wheel, keyboard, and pointer-lock listeners.
 * Supports both orbit and FPS modes.
 */
export function initCameraControls(canvas, camera) {
  let isDragging = false;
  let isPanDrag = false;
  let activePointerId = null;
  let lastX = 0;
  let lastY = 0;

  camera.mode = camera.mode || "orbit";
  camera.keys = camera.keys || {
    w: false,
    a: false,
    s: false,
    d: false,
    q: false,
    e: false,
  };
  camera.fpsEye = camera.fpsEye || [0, 0, 5];
  camera.fpsYaw = camera.fpsYaw || 0;
  camera.fpsPitch = camera.fpsPitch || 0;
  camera.fpsSpeed = camera.fpsSpeed || 2.5;

  /* ── Inertia state ── */
  let velocityTheta = 0;
  let velocityPhi = 0;
  let velocityPanX = 0;
  let velocityPanY = 0;
  const DAMPING = 0.92; // per-frame multiplier (lower = faster stop)
  const MIN_VELOCITY = 0.0001;

  camera._inertia = {
    update() {
      if (isDragging || camera.mode === "fps") return;
      if (Math.abs(velocityTheta) > MIN_VELOCITY || Math.abs(velocityPhi) > MIN_VELOCITY) {
        camera.theta += velocityTheta;
        camera.phi += velocityPhi;
        velocityTheta *= DAMPING;
        velocityPhi *= DAMPING;
      }
      if (Math.abs(velocityPanX) > MIN_VELOCITY || Math.abs(velocityPanY) > MIN_VELOCITY) {
        const eye = computeEye(camera);
        const viewDir = normalize3([
          camera.target[0] - eye[0], camera.target[1] - eye[1], camera.target[2] - eye[2],
        ]);
        const right = normalize3(cross3(viewDir, camera.up || [0, 1, 0]));
        const camUp = normalize3(cross3(right, viewDir));
        const panSpeed = camera.radius * 0.004;
        camera.target[0] += (velocityPanX * right[0] + velocityPanY * camUp[0]) * panSpeed;
        camera.target[1] += (velocityPanX * right[1] + velocityPanY * camUp[1]) * panSpeed;
        camera.target[2] += (velocityPanX * right[2] + velocityPanY * camUp[2]) * panSpeed;
        velocityPanX *= DAMPING;
        velocityPanY *= DAMPING;
      }
    },
  };

  canvas.style.touchAction = "none";

  /* ── Pointer (orbit mode) ── */
  const onPointerDown = (e) => {
    if (camera.mode === "fps") return; // handled by pointer lock
    activePointerId = e.pointerId;
    isDragging = true;
    isPanDrag = e.button === 2 || e.shiftKey;
    lastX = e.clientX;
    lastY = e.clientY;
    // Kill inertia on new drag
    velocityTheta = 0;
    velocityPhi = 0;
    velocityPanX = 0;
    velocityPanY = 0;
    canvas.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e) => {
    if (camera.mode === "fps") return;
    if (activePointerId !== e.pointerId) return;
    if (!isDragging) return;

    e.preventDefault();
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (isPanDrag) {
      // "Grab the scene" convention: scene follows mouse direction
      const eye = computeEye(camera);
      const viewDir = normalize3([
        camera.target[0] - eye[0],
        camera.target[1] - eye[1],
        camera.target[2] - eye[2],
      ]);
      const right = normalize3(cross3(viewDir, camera.up || [0, 1, 0]));
      const camUp = normalize3(cross3(right, viewDir));
      const panSpeed = camera.radius * 0.004;
      camera.target[0] += (-dx * right[0] + dy * camUp[0]) * panSpeed;
      camera.target[1] += (-dx * right[1] + dy * camUp[1]) * panSpeed;
      camera.target[2] += (-dx * right[2] + dy * camUp[2]) * panSpeed;
      velocityPanX = -dx;
      velocityPanY = dy;
    } else {
      // Turntable orbit
      const dTheta = -dx * 0.005;
      const dPhi = -dy * 0.005;
      camera.theta += dTheta;
      camera.phi += dPhi;
      velocityTheta = dTheta;
      velocityPhi = dPhi;
    }
  };

  const onPointerUp = (e) => {
    if (activePointerId !== e.pointerId) return;
    isDragging = false;
    isPanDrag = false;
    activePointerId = null;
    canvas.releasePointerCapture(e.pointerId);
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove, { passive: false });
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);

  /* ── Scroll (orbit zoom + FPS speed adjust) ── */
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      if (camera.mode === "fps") {
        camera.fpsSpeed = Math.min(
          15,
          Math.max(0.5, camera.fpsSpeed * (1 - e.deltaY * 0.001)),
        );
        camera.moveSpeed = camera.fpsSpeed;
        canvas.dispatchEvent(
          new CustomEvent("speedchange", { detail: camera.fpsSpeed }),
        );
      } else {
        const newRadius = Math.max(0.1, camera.radius + e.deltaY * 0.01);
        canvas.dispatchEvent(new CustomEvent("zoomchange", { detail: newRadius }));
      }
    },
    { passive: false },
  );

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  /* ── Pointer lock (FPS mode) ── */
  canvas.addEventListener("click", () => {
    if (camera.mode === "fps" && document.pointerLockElement !== canvas) {
      camera._skipPointerLockFrames = 2;
      canvas.requestPointerLock();
    }
  });

  document.addEventListener("mousemove", (e) => {
    if (camera.mode !== "fps") return;
    if (document.pointerLockElement !== canvas) return;
    // Skip first events after pointer lock to avoid cursor-centering jump
    if (camera._skipPointerLockFrames > 0) {
      camera._skipPointerLockFrames--;
      return;
    }
    camera.fpsYaw += e.movementX * 0.002;
    camera.fpsPitch -= e.movementY * 0.002;
    camera.fpsPitch = Math.max(
      -Math.PI / 2 + 0.01,
      Math.min(Math.PI / 2 - 0.01, camera.fpsPitch),
    );
  });

  /* ── Keyboard ── */
  const onKeyDown = (e) => {
    const k = e.key.toLowerCase();
    if (k in camera.keys) {
      camera.keys[k] = true;
      e.preventDefault();
    }
  };
  const onKeyUp = (e) => {
    const k = e.key.toLowerCase();
    if (k in camera.keys) {
      camera.keys[k] = false;
      e.preventDefault();
    }
  };

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
}
