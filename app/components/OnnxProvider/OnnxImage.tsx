import { Tensor } from "onnxruntime-web"
import { createRef, useState } from "react"

import { useModel } from "./OnnxProvider"

interface Results {
  output: Float32Array,
  time: number
}

interface Props {
  src: string
  style? : React.CSSProperties
  children?: (results: Results | undefined) => React.ReactNode
}

export function OnnxImage({src, style, children}: Props) {
  const ref = createRef<HTMLImageElement>()
  const {session, width, height, reverseDimensions} = useModel()
  const [results, setResults] = useState<Results | undefined>()

  const runModel = async () => {
    if (!ref.current || !session) {
      return 
    }

    // Tensor.fromImage does not return the correct dimensions when we use an img element, 
    // so we are drawing the image data to a temporary canvas instead
    // https://github.com/microsoft/onnxruntime/issues/17094
    const canvas = document.createElement('canvas')
    canvas.width = ref.current.width
    canvas.height = ref.current.height
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }
    ctx.drawImage(ref.current, 0, 0)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

    // resize the tensor to match the model requirements
    const tensor = await Tensor.fromImage(imageData, {resizedWidth: width, resizedHeight: height})

    // optionally reverse the tensor dimensions
    const tensorReshaped = reverseDimensions ? tensor.reshape([1, width, height, 3]) : tensor

    // run the model
    const start = Date.now()
    const feeds: Record<string, Tensor> = {}
    feeds[session.inputNames[0]] = tensorReshaped
    const outputData = await session.run(feeds);
    const end = Date.now();
    const inferenceTime = (end - start);
    const output = outputData[session.outputNames[0]];
   
    setResults({
      output: output.data as Float32Array, 
      time: inferenceTime,
    })

    // document.body.appendChild(canvas)
  }

  return (
    <>
      <img ref={ref} alt="onnx-image" src={src} style={style} onLoad={runModel} />
      {children && children(results)}
    </>
  )
}