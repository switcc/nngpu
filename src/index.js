/**
 * nngpu demo — full WebGPU ↔ WebNN round-trip.
 *
 * Pipeline:
 *   1. Create raw data (simulated pixel values 0–255).
 *   2. Upload to a GPUBuffer.
 *   3. Run a WebGPU compute shader to normalise the data to [0, 1].
 *   4. Transfer the GPUBuffer contents into an MLTensor  (GPU → NN).
 *   5. Run a tiny WebNN linear model on the normalised data.
 *   6. Transfer the MLTensor output back to a GPUBuffer  (NN → GPU).
 *   7. Read the final result on the CPU and display it.
 */

import { createSharedContext } from './device.js';
import { gpuBufferToMLTensor, mlTensorToGPUBuffer } from './interop.js';
import { runComputeShader, createFloat32Buffer, readFloat32Buffer } from './webgpu.js';
import { buildLinearModel, runInference, createTensorPair } from './webnn.js';

const SIZE = 4; // vector length for the demo

function log(msg) {
  const el = document.getElementById('log');
  el.textContent += msg + '\n';
  console.log(msg);
}

async function main() {
  try {
    // ── 1. Shared context ───────────────────────────────────────────
    log('Initialising shared WebGPU + WebNN context…');
    const { device, mlContext } = await createSharedContext();
    log('  ✓ GPUDevice and MLContext ready (shared GPU backend).');

    // ── 2. Raw data → GPUBuffer ─────────────────────────────────────
    const raw = new Float32Array([50, 100, 200, 255]);
    log(`\nRaw input data: [${raw}]`);

    const gpuBuf = createFloat32Buffer(device, raw);
    log('  ✓ Uploaded to GPUBuffer.');

    // ── 3. WebGPU compute: normalise to [0, 1] ─────────────────────
    const shaderResp = await fetch('../shaders/normalize.wgsl');
    const shaderCode = await shaderResp.text();
    const workgroups = Math.ceil(SIZE / 64);

    await runComputeShader(device, shaderCode, gpuBuf, workgroups);
    const normalised = await readFloat32Buffer(device, gpuBuf);
    log(`  ✓ After WebGPU normalisation: [${Array.from(normalised).map(v => v.toFixed(4))}]`);

    // ── 4. GPUBuffer → MLTensor  (WebGPU → WebNN) ──────────────────
    log('\nTransferring GPUBuffer → MLTensor…');
    const inputTensor = await gpuBufferToMLTensor(device, mlContext, gpuBuf, {
      dataType: 'float32',
      shape: [1, SIZE],
    });
    log('  ✓ Data now lives in an MLTensor.');

    // ── 5. WebNN inference ──────────────────────────────────────────
    // Tiny model: output = relu(input * W + b)
    // W = identity, b = 0.1  →  output ≈ normalised + 0.1
    const weights = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < SIZE; i++) weights[i * SIZE + i] = 1.0; // identity
    const bias = new Float32Array(SIZE).fill(0.1);

    log('\nBuilding WebNN model (linear + relu)…');
    const { graph, inputName, outputName } = await buildLinearModel(
      mlContext, SIZE, SIZE, weights, bias,
    );
    log('  ✓ Graph compiled.');

    const { outputTensor } = await createTensorPair(mlContext, SIZE, SIZE);
    // We already have inputTensor from the interop step — use it directly.

    log('Running inference…');
    await runInference(mlContext, graph, inputName, inputTensor, outputName, outputTensor);
    log('  ✓ Inference complete.');

    // ── 6. MLTensor → GPUBuffer  (WebNN → WebGPU) ──────────────────
    log('\nTransferring MLTensor → GPUBuffer…');
    const resultBuf = await mlTensorToGPUBuffer(device, mlContext, outputTensor);
    log('  ✓ Data back in a GPUBuffer.');

    // ── 7. Read final result ────────────────────────────────────────
    const result = await readFloat32Buffer(device, resultBuf);
    log(`\nFinal output: [${Array.from(result).map(v => v.toFixed(4))}]`);
    log('\n✓ Full round-trip complete: WebGPU → WebNN → WebGPU');

    // Cleanup
    gpuBuf.destroy();
    resultBuf.destroy();
    inputTensor.destroy();
    outputTensor.destroy();

  } catch (err) {
    log(`\n✗ Error: ${err.message}`);
    console.error(err);
  }
}

main();
