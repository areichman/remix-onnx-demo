import { InferenceSession, Tensor, TensorFromImageElementOptions } from "onnxruntime-web"

export async function classifyImage(session: InferenceSession, image: HTMLImageElement, options?: TensorFromImageElementOptions) {
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
  const tensor = await Tensor.fromImage(imageData, options)

  // TODO: override the missing NHWC capability and change the tensor layout ourselves 

  // run the model
  const start = Date.now()
  const feeds: Record<string, Tensor> = {}
  feeds[session.inputNames[0]] = tensor
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