# 3D Gaussian Splatting Viewer (WebGPU)

A browser-based viewer for Gaussian Splatting scenes.

This project loads pre-trained Gaussian point clouds (`.ply`) and renders them in real time using WebGPU. It includes orbit and FPS camera controls, model/iteration switching, spherical harmonics degree selection, resolution scaling, point-mode vs splat-mode rendering, and helper overlays.

## What Is Happening In This Project

At a high level:

1. The app starts from `index.html` + `js/main.js`.
2. A WebGPU device/context is created.
3. The selected model PLY is fetched from:
   - `models/<model>/point_cloud/iteration_<iter>/point_cloud.ply`
4. The PLY is parsed (`js/plyLoader.js`) into GPU-ready arrays:
   - positions
   - covariance data
   - colors / SH coefficients
5. Data is uploaded to GPU buffers (`js/splatRenderer.js`).
6. Every frame:
   - camera transforms are updated (`js/controls.js`)
   - Gaussians are depth-sorted (`js/radixSort.js`)
   - the renderer draws either splats or points
7. Optional scene helpers (up vector / plane) are rendered (`js/helpers.js`).

## Requirements

- Python 3.x (used only to serve static files locally)
- A modern WebGPU-capable browser (latest Chrome/Edge recommended)

## Run Locally

From the project root, run:

```bash
python -m http.server {port}
```

Example:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:{port}
```

Example:

```text
http://localhost:8000
```

Important: do not open `index.html` directly via `file://` because model and shader loading uses `fetch()` and requires an HTTP server.

## Download Models (INRIA)

Download the Gaussian Splatting scenes from INRIA's official 3D Gaussian Splatting page:

[https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)

After downloading/extracting, place each scene under `models/` so the structure looks like:

```text
models/
  bicycle/
    cameras.json
    cfg_args
    point_cloud/
      iteration_7000/
        point_cloud.ply
      iteration_30000/
        point_cloud.ply
  garden/
    ...
```

The viewer expects exactly this pattern:

- `models/<scene>/cameras.json`
- `models/<scene>/point_cloud/iteration_<N>/point_cloud.ply`

If these files are missing, model loading will fail in the UI.

## Controls (Quick)

- Orbit mode:
  - Left mouse: rotate
  - Middle/right drag: pan
  - Wheel: zoom
- FPS mode:
  - Click canvas to lock pointer
  - `W/A/S/D`: move
  - Mouse move: look around

The left and right panels in the UI expose additional render and camera options.

## Notes

- Rendering quality/performance depends on your GPU and browser WebGPU implementation.
- Large scenes can take time to parse/upload on first load.
- You can also drag and drop a local `.ply` file directly onto the canvas.
