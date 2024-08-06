import { useLoaderData } from '@remix-run/react'
import { createRef, useEffect, useRef } from 'react'
import { InferenceSession } from 'onnxruntime-web'

import { preprocess } from '~/utils/preprocess'
import { runModel } from '~/utils/runModel'

export async function clientLoader() {
  const response = await fetch('/models/squeezenet1_1.onnx')
  const model = await response.arrayBuffer()
  const session = await InferenceSession.create(model, {executionProviders: ['webgl']})
  return session
}

export function HydrateFallback() {
  return <p>Loading...</p>;
}

export default function Predict() {
  const session = useLoaderData()
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
    } 
  })

  useEffect(() => {
    image.current!.src = '/images/webb-pillars.jpg'
  })

  return (
    <canvas ref={canvas} width="224" height="224" />
  )
}