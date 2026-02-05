# CLAUDE.md — nngpu

This file provides guidance for AI assistants working in this repository.

## Project Overview

**nngpu** is a JavaScript library and demo for exchanging data between **WebGPU** and **WebNN**. The core idea: create a single `GPUDevice`, derive an `MLContext` from it, and use shared-backend interop utilities to move data between GPU compute shaders and neural-network inference without unnecessary copies.

## Repository Status

- **State**: Active development
- **Remote**: `origin` at `switcc/nngpu`
- **Primary language**: JavaScript (ES modules) + WGSL (shaders)
- **Runtime**: Browser (Chrome 128+ with WebGPU and WebNN)

## Directory Structure

```
nngpu/
├── CLAUDE.md              # This file — AI assistant guide
├── index.html             # Demo page
├── package.json           # Project metadata, dev server script
├── .gitignore
├── src/
│   ├── device.js          # Shared GPUDevice + MLContext initialisation
│   ├── interop.js         # Data exchange: GPUBuffer ↔ MLTensor
│   ├── webgpu.js          # WebGPU compute helpers (dispatch, buffer I/O)
│   ├── webnn.js           # WebNN graph building + inference helpers
│   └── index.js           # Demo entry point (full round-trip pipeline)
└── shaders/
    └── normalize.wgsl     # Example compute shader (min-max normalisation)
```

## Build & Run

No build step — the project uses native ES modules loaded directly by the browser.

```sh
# Start a local dev server (requires Node.js / npx)
npm start          # runs: npx serve .
```

Then open `http://localhost:3000` in Chrome 128+ (or any browser with both WebGPU and WebNN).

## Testing

No automated test framework yet. Manual verification:

1. Run `npm start` and open the demo page.
2. Check the on-page log and DevTools console for the full pipeline output.
3. Confirm the round-trip: raw data → WebGPU normalise → MLTensor → WebNN inference → GPUBuffer → CPU readback.

## Key Modules

### `src/device.js`
Initialises the shared context. Call `createSharedContext()` to get a `{ adapter, device, mlContext }` tuple where the MLContext is backed by the same GPU device.

### `src/interop.js`
Core data-exchange functions:
- `gpuBufferToMLTensor()` — reads a GPUBuffer into a new MLTensor (allocating)
- `mlTensorToGPUBuffer()` — reads an MLTensor into a new GPUBuffer (allocating)
- `copyGPUBufferToMLTensor()` — copies into an existing MLTensor (in-place)
- `copyMLTensorToGPUBuffer()` — copies into an existing GPUBuffer (in-place)

All transfers currently go through a CPU-side staging step (`mapAsync` / `readTensor` / `writeTensor`). The shared MLContext still helps the browser avoid full device↔host round-trips internally.

### `src/webgpu.js`
Thin helpers for compute dispatch:
- `runComputeShader()` — bind a storage buffer and dispatch workgroups
- `createFloat32Buffer()` — upload a Float32Array to a GPUBuffer
- `readFloat32Buffer()` — map a GPUBuffer back to CPU

### `src/webnn.js`
Graph-building helpers:
- `buildLinearModel()` — constructs a `relu(input × W + b)` graph
- `runInference()` — dispatches `mlContext.compute()`
- `createTensorPair()` — allocates paired input/output MLTensors

### `shaders/normalize.wgsl`
Example WGSL compute shader that normalises float32 values from [0, 255] to [0, 1].

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Browser                          │
│                                                      │
│   ┌────────────┐    interop.js    ┌──────────────┐   │
│   │  WebGPU    │ ───────────────→ │   WebNN      │   │
│   │  GPUBuffer │ ←─────────────── │   MLTensor   │   │
│   └────────────┘                  └──────────────┘   │
│         ↑                               ↑            │
│         └───── shared GPUDevice ────────┘            │
│                  (device.js)                         │
└──────────────────────────────────────────────────────┘
```

**Data flow in the demo:**
1. CPU → GPUBuffer (upload raw pixel values)
2. GPUBuffer → GPUBuffer (compute shader normalises in-place)
3. GPUBuffer → MLTensor (interop: staging + writeTensor)
4. MLTensor → MLTensor (WebNN inference)
5. MLTensor → GPUBuffer (interop: readTensor + upload)
6. GPUBuffer → CPU (readback for display)

## Development Workflow

### Branching

- Feature branches follow the pattern `claude/<description>-<session-id>`
- Develop on your assigned branch; do not push directly to `main`

### Commits

- Write clear, descriptive commit messages
- GPG signing is enabled and required (SSH format)
- Keep commits focused — one logical change per commit

### Code Style

- ES modules (`import`/`export`), no bundler
- JSDoc type annotations on all public functions
- WGSL shaders live in `shaders/` and are fetched at runtime
- Prefer `async`/`await` over raw Promises

## Key Conventions

1. **Keep this file up to date** — When adding new modules, build steps, or conventions, update CLAUDE.md so future AI sessions have accurate context.
2. **Prefer simplicity** — Avoid over-engineering; add complexity only when justified by requirements.
3. **Shared context is mandatory** — Always derive `MLContext` from the `GPUDevice` via `createSharedContext()`. Never create independent contexts.
4. **Destroy GPU resources** — Call `.destroy()` on GPUBuffers and MLTensors when done.
5. **No secrets in the repo** — Never commit API keys, credentials, or `.env` files.

## Dependencies

- **Runtime**: None — uses only browser-native WebGPU and WebNN APIs.
- **Dev server**: `npx serve` (fetched on demand, not installed).
- **Browser requirement**: Chrome 128+ or equivalent with both WebGPU and WebNN enabled.

## Useful Commands

| Task | Command |
|------|---------|
| Dev server | `npm start` |
| Lint | Not configured yet |
| Test | Manual — open demo page in browser |
