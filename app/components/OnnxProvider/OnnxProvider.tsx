import { InferenceSession } from "onnxruntime-web"
import { InferenceSession as InferenceSessionWebGPU } from "onnxruntime-web/webgpu"
import { createContext, useContext, useEffect, useState } from "react"

interface ContextType {
  session: InferenceSession | null,
  width: number,
  height: number,
  reverseDimensions: boolean,
}

export const OnnxContext = createContext<ContextType>({
  session: null,
  width: 224,
  height: 224,
  reverseDimensions: true,
})

export function useModel() {
  return useContext(OnnxContext)
}

interface Props {
  children: React.ReactNode
  url: string
  width?: number
  height?: number
  reverseDimensions?: boolean
  executionProvider?: 'webgl' | 'wasm' | 'webgpu'
}

export function OnnxProvider({
  url, 
  width = 224, 
  height = 224, 
  reverseDimensions = false,
  executionProvider = 'webgl',
  children, 
}: Props) {
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

        if (executionProvider === 'webgpu') {
          session = await InferenceSessionWebGPU.create(model, {executionProviders: ['webgpu']})
        } else {
          session = await InferenceSession.create(model, {executionProviders: [executionProvider]})
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
    width,
    height,
    reverseDimensions,
  }

  return (
    <OnnxContext.Provider value={value}>
      {children}
    </OnnxContext.Provider>
  )
}