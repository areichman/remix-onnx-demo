import { InferenceSession, Tensor, TensorFromImageElementOptions } from "onnxruntime-web"
import { toNHWC } from "~/utils/toNHWC"

export async function classifyImage(session: InferenceSession, image: HTMLImageElement, options?: TensorFromImageElementOptions) {
  // TODO: Use the native NHWC option when available
  // Setting OnnxProvider.options.tensorLayout = 'NHWC' currently produces: 
  // Error: NHWC Tensor layout is not supported yet
  const __options = {...options}
  let __useInternalNHWC = false
  if (options?.tensorLayout === 'NHWC') {
    __useInternalNHWC = true
    delete __options.tensorLayout
  } 

  // Tensor.fromImage does not return the correct dimensions when we use an img element, 
  // so we are drawing the image data to a temporary canvas instead
  // https://github.com/microsoft/onnxruntime/issues/17094
  const canvas = document.createElement('canvas')
  canvas.width = image.width
  canvas.height = image.height
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(image, 0, 0)
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

  // resize the tensor to match the model requirements
  const tensor = await Tensor.fromImage(imageData, __options)

  // TODO: Use the native NHWC option when available
  const convertedTensor = __useInternalNHWC ? toNHWC(tensor) : tensor

  // run the model
  const start = Date.now()
  const feeds: Record<string, Tensor> = {}
  feeds[session.inputNames[0]] = convertedTensor
  const outputData = await session.run(feeds);
  const end = Date.now();
  const inferenceTime = (end - start);
  const output = outputData[session.outputNames[0]];
 
  return {
    output: output.data as Float32Array, 
    time: inferenceTime,
  }

  // debug
  // document.body.appendChild(canvas)
}