import { createRef, useEffect, useRef } from 'react'
import { InferenceSession } from 'onnxruntime-web'

import { preprocess } from '~/utils/preprocess'
import { runModel } from '~/utils/runModel'
import { OnnxProvider, useModel } from '~/components/OnnxProvider/OnnxProvider'

function Canvas() {
  const session = useModel()
  const canvas = createRef<HTMLCanvasElement>()
  const image = useRef<HTMLImageElement>()

  useEffect(() => {
    image.current = new Image()
    image.current.onload = async () => {
      const ctx = canvas.current?.getContext('2d')
      if (ctx) {
        ctx.drawImage(image.current!, 0, 0)
        const preprocessed = preprocess(ctx)
        const results = await runModel(session as InferenceSession, preprocessed)
        console.log(results)
      }

      /*
      const canvasElement = document.createElement('canvas')
      canvasElement.width = 224
      canvasElement.height = 224
      const canvasCtx = canvasElement.getContext('2d')
      canvasCtx!.drawImage(image.current!, 0, 0)
      const p2 = preprocess(canvasCtx!)
      const r2 = await runModel(session as InferenceSession, p2)
      console.log('r2', r2)
      */
    }

    
  })

  useEffect(() => {
    image.current!.src = '/images/webb-pillars.jpg'
  })

  return <canvas ref={canvas} width="224" height="224" />
}

export default function Route() {
  return (
    <OnnxProvider modelUrl='/models/squeezenet1_1.onnx'>
      <Canvas />
    </OnnxProvider>
  )
}