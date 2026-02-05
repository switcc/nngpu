/**
 * Shared device initialization.
 *
 * Creates a single GPUDevice and derives an MLContext from it so that
 * WebGPU and WebNN operate on the same hardware backend.
 */

/**
 * @typedef {Object} SharedContext
 * @property {GPUAdapter}  adapter
 * @property {GPUDevice}   device
 * @property {MLContext}    mlContext
 */

/**
 * Initialise a shared WebGPU + WebNN context.
 *
 * @returns {Promise<SharedContext>}
 */
export async function createSharedContext() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser.');
  }
  if (!navigator.ml) {
    throw new Error('WebNN is not supported in this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('Failed to obtain a GPUAdapter.');
  }

  const device = await adapter.requestDevice();

  // Derive an MLContext that shares the same GPU device.
  const mlContext = await navigator.ml.createContext(device);

  return { adapter, device, mlContext };
}
