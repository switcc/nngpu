/**
 * WebNN graph-building helpers.
 *
 * Wraps MLGraphBuilder to make it easier to construct small models and
 * run inference through an MLContext that shares a GPU device with WebGPU.
 */

/**
 * Build a simple single-layer linear model:  output = relu(input * weights + bias)
 *
 * This is intentionally minimal — just enough to demonstrate the interop
 * pipeline.  Replace with a real model topology as needed.
 *
 * @param {MLContext} mlContext
 * @param {number}   inputSize
 * @param {number}   outputSize
 * @param {Float32Array} weightsData  Row-major [inputSize, outputSize].
 * @param {Float32Array} biasData     [outputSize].
 * @returns {Promise<{ graph: MLGraph, inputName: string, outputName: string }>}
 */
export async function buildLinearModel(mlContext, inputSize, outputSize, weightsData, biasData) {
  const builder = new MLGraphBuilder(mlContext);

  const input = builder.input('input', {
    dataType: 'float32',
    shape: [1, inputSize],
  });

  const weights = builder.constant(
    { dataType: 'float32', shape: [inputSize, outputSize] },
    weightsData,
  );

  const bias = builder.constant(
    { dataType: 'float32', shape: [1, outputSize] },
    biasData,
  );

  const matmul = builder.matmul(input, weights);
  const add = builder.add(matmul, bias);
  const output = builder.relu(add);

  const graph = await builder.build({ output });
  return { graph, inputName: 'input', outputName: 'output' };
}

/**
 * Run inference on a pre-built graph.
 *
 * @param {MLContext}  mlContext
 * @param {MLGraph}   graph
 * @param {string}    inputName   Name used in `builder.input()`.
 * @param {MLTensor}  inputTensor Writable tensor with data already written.
 * @param {string}    outputName  Name used in `builder.build()`.
 * @param {MLTensor}  outputTensor Readable tensor to receive results.
 * @returns {Promise<void>}
 */
export async function runInference(mlContext, graph, inputName, inputTensor, outputName, outputTensor) {
  const inputs = { [inputName]: inputTensor };
  const outputs = { [outputName]: outputTensor };
  await mlContext.compute(graph, inputs, outputs);
}

/**
 * Create a pair of input/output MLTensors for a graph.
 *
 * @param {MLContext} mlContext
 * @param {number}   inputSize
 * @param {number}   outputSize
 * @returns {Promise<{ inputTensor: MLTensor, outputTensor: MLTensor }>}
 */
export async function createTensorPair(mlContext, inputSize, outputSize) {
  const inputTensor = await mlContext.createTensor({
    dataType: 'float32',
    shape: [1, inputSize],
    writable: true,
    readable: true,
  });

  const outputTensor = await mlContext.createTensor({
    dataType: 'float32',
    shape: [1, outputSize],
    writable: true,
    readable: true,
  });

  return { inputTensor, outputTensor };
}
