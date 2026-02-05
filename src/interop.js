/**
 * Data-exchange utilities between WebGPU buffers and WebNN tensors.
 *
 * Both directions go through a CPU-side ArrayBuffer because the current
 * WebNN spec does not expose a direct GPUBuffer↔MLTensor copy path.
 * Creating the MLContext from the GPUDevice (see device.js) still helps:
 * the browser can internally keep data on the same GPU and only stage
 * through a pinned host buffer rather than a full device→host→device copy.
 */

// ── GPUBuffer → MLTensor ────────────────────────────────────────────

/**
 * Read a GPUBuffer into a new MLTensor.
 *
 * @param {GPUDevice}  device     WebGPU device that owns `srcBuffer`.
 * @param {MLContext}  mlContext  WebNN context (should share the same GPU).
 * @param {GPUBuffer}  srcBuffer  Source buffer (must have COPY_SRC usage).
 * @param {Object}     tensorDesc MLTensor descriptor — { dataType, shape }.
 * @returns {Promise<MLTensor>}
 */
export async function gpuBufferToMLTensor(device, mlContext, srcBuffer, tensorDesc) {
  const byteLength = srcBuffer.size;

  // 1. Copy srcBuffer → a mappable staging buffer.
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(srcBuffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  // 2. Map the staging buffer and grab the bytes.
  await staging.mapAsync(GPUMapMode.READ);
  const cpuData = new ArrayBuffer(byteLength);
  new Uint8Array(cpuData).set(new Uint8Array(staging.getMappedRange()));
  staging.unmap();
  staging.destroy();

  // 3. Create an MLTensor and write the bytes into it.
  const tensor = await mlContext.createTensor({
    dataType: tensorDesc.dataType,
    shape: tensorDesc.shape,
    writable: true,
    readable: true,
  });
  mlContext.writeTensor(tensor, cpuData);

  return tensor;
}

// ── MLTensor → GPUBuffer ────────────────────────────────────────────

/**
 * Read an MLTensor into a new GPUBuffer.
 *
 * @param {GPUDevice}  device     WebGPU device.
 * @param {MLContext}  mlContext  WebNN context that owns `srcTensor`.
 * @param {MLTensor}   srcTensor  Source tensor (must be readable).
 * @param {number}     [usage]    Extra GPUBufferUsage flags for the result
 *                                buffer. COPY_SRC is always added.
 * @returns {Promise<GPUBuffer>}
 */
export async function mlTensorToGPUBuffer(device, mlContext, srcTensor, usage = 0) {
  // 1. Read tensor data to CPU.
  const cpuData = await mlContext.readTensor(srcTensor);

  // 2. Create a GPU buffer and upload.
  const buffer = device.createBuffer({
    size: cpuData.byteLength,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE | usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(cpuData));
  buffer.unmap();

  return buffer;
}

// ── In-place transfers (reuse existing objects) ─────────────────────

/**
 * Copy the contents of a GPUBuffer into an existing MLTensor.
 *
 * @param {GPUDevice}  device
 * @param {MLContext}  mlContext
 * @param {GPUBuffer}  srcBuffer
 * @param {MLTensor}   dstTensor  Must have been created with `writable: true`.
 */
export async function copyGPUBufferToMLTensor(device, mlContext, srcBuffer, dstTensor) {
  const byteLength = srcBuffer.size;

  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(srcBuffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const cpuData = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();

  mlContext.writeTensor(dstTensor, cpuData);
}

/**
 * Copy the contents of an MLTensor into an existing GPUBuffer.
 *
 * @param {GPUDevice}  device
 * @param {MLContext}  mlContext
 * @param {MLTensor}   srcTensor  Must have been created with `readable: true`.
 * @param {GPUBuffer}  dstBuffer  Must have COPY_DST usage.
 */
export async function copyMLTensorToGPUBuffer(device, mlContext, srcTensor, dstBuffer) {
  const cpuData = await mlContext.readTensor(srcTensor);
  device.queue.writeBuffer(dstBuffer, 0, cpuData);
}
