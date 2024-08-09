/* Sample usage:

<OnnxProvider url="/models/squeezenet.onnx">

<OnnxProvider 
  url="/models/internal.onnx"
  executionProviders={['wasm']} 
  options={{
    tensorLayout: 'NHWC',
    resizedHeight: 384,
    resizedWidth: 384,
    norm: {mean: 1, bias: 0},
  }}
>
*/

import { InferenceSession, TensorFromImageElementOptions } from "onnxruntime-web"
import { InferenceSession as InferenceSessionWebGPU } from "onnxruntime-web/webgpu"
import { createContext, useContext, useEffect, useState } from "react"

import { classifyImage } from "./classifyImage"

interface Results {
  output: Float32Array
  time: number
}

interface ContextType {
  session?: InferenceSession,
  run: (image: HTMLImageElement) => Promise<Results>
}

export const OnnxContext = createContext<ContextType>({
  session: undefined,
  run: async () => ({output: new Float32Array, time: 0})
})

export function useModel() {
  return useContext(OnnxContext)
}

interface Props {
  url: string
  executionProviders?: InferenceSession.ExecutionProviderConfig[]
  options?: TensorFromImageElementOptions
  children: React.ReactNode
}

export function OnnxProvider({url, executionProviders = ['webgl'], options, children}: Props) {
  const [session, setSession] = useState<InferenceSession>()

  useEffect(
    () => {
      async function createSession() {
        const cache = await caches.open('models');
        let response = await cache.match(url)

        if (response) {
          console.log('using cached model')
        } else {
          console.log('fetching model and adding to cache')
          await cache.add(url)
          response = await cache.match(url)
        }

        const model = await response!.arrayBuffer()
        let session: InferenceSession

        if (executionProviders.includes('webgpu')) {
          session = await InferenceSessionWebGPU.create(model, {executionProviders: ['webgpu']})
        } else {
          session = await InferenceSession.create(model, {executionProviders})
        }
        setSession(session)
      }

      createSession()
    }, 
    [] // eslint-disable-line react-hooks/exhaustive-deps
  )

  if (!session) {
    return <p>Loading model...</p>
  }

  const value = {
    session,
    run: (image: HTMLImageElement) => classifyImage(session!, image, options),
  }

  return (
    <OnnxContext.Provider value={value}>
      {children}
    </OnnxContext.Provider>
  )
}