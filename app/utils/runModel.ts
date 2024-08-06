import { InferenceSession, Tensor } from 'onnxruntime-web';
import { getPredictedClass } from '~/utils/imagenet';

/*
function init() {
  // env.wasm.simd = false;
}

export async function createModelCpu(model: ArrayBuffer): Promise<InferenceSession> {
  init();
  return await InferenceSession.create(model, {executionProviders: ['wasm']});
}

export async function createModelGpu(model: ArrayBuffer): Promise<InferenceSession> {
  init();
  return await InferenceSession.create(model, {executionProviders: ['webgl']});
}

export async function warmupModel(model: InferenceSession, dims: number[]) {
  // OK. we generate a random input and call Session.run() as a warmup query
  const size = dims.reduce((a, b) => a * b);
  const warmupTensor = new Tensor('float32', new Float32Array(size), dims);

  for (let i = 0; i < size; i++) {
    warmupTensor.data[i] = Math.random() * 2.0 - 1.0;  // random value [-1.0, 1.0)
  }
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = warmupTensor;
    await model.run(feeds);
  } catch (e) {
    console.error(e);
  }
}
*/

export async function runModel(model: InferenceSession, preprocessedData: Tensor) {
  try {
    const start = Date.now();
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = preprocessedData;
    const outputData = await model.run(feeds);
    const end = Date.now();
    const inferenceTime = (end - start);

    const output = outputData[model.outputNames[0]];
    const results = getPredictedClass(output.data as Float32Array);
    
    return {results, inferenceTime};
  } catch (e) {
    console.error(e);
    throw new Error();
  }
}