/**
 * WebGPU compute helpers.
 *
 * Provides a thin wrapper around dispatching compute shaders and reading
 * results back.  Used in the demo to pre-process data on the GPU before
 * handing it to WebNN.
 */

/**
 * Run a compute shader that operates on a storage buffer in-place.
 *
 * @param {GPUDevice} device
 * @param {string}    wgslSource  WGSL shader source. Must declare a
 *                                @group(0) @binding(0) storage buffer.
 * @param {GPUBuffer} buffer      Storage buffer (STORAGE usage required).
 * @param {number}    workgroups  Number of workgroups to dispatch (x-axis).
 * @returns {Promise<void>}       Resolves after the GPU work completes.
 */
export async function runComputeShader(device, wgslSource, buffer, workgroups) {
  const module = device.createShaderModule({ code: wgslSource });

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer } }],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();

  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

/**
 * Create a GPUBuffer pre-filled with float32 data.
 *
 * @param {GPUDevice}    device
 * @param {Float32Array} data
 * @param {number}       [extraUsage]  Additional GPUBufferUsage flags.
 * @returns {GPUBuffer}
 */
export function createFloat32Buffer(device, data, extraUsage = 0) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Read a GPUBuffer back to the CPU as a Float32Array.
 *
 * @param {GPUDevice}  device
 * @param {GPUBuffer}  srcBuffer  Must have COPY_SRC usage.
 * @returns {Promise<Float32Array>}
 */
export async function readFloat32Buffer(device, srcBuffer) {
  const staging = device.createBuffer({
    size: srcBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(srcBuffer, 0, staging, 0, srcBuffer.size);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}
