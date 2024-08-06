import { InferenceSession } from "onnxruntime-web"
import { createContext, useContext, useEffect, useState } from "react"

export const OnnxContext = createContext<InferenceSession | null>(null)

export function useModel() {
  return useContext(OnnxContext)
}

interface Props {
  children: React.ReactNode
  modelUrl: string
}

export function OnnxProvider({children, modelUrl}: Props) {
  const [session, setSession] = useState<InferenceSession>()

  useEffect(() => {
    async function createSession() {
      const cache = await caches.open('models');
      // const keys = await cache.keys()
      // console.log(keys)
      cache.add(modelUrl);
      const response = await cache.match(modelUrl);
      const model = await response!.arrayBuffer()
      const session = await InferenceSession.create(model, {executionProviders: ['webgl']})
      setSession(session)
    }

    createSession()
  }, [modelUrl])

  if (!session) {
    return null
  }

  return (
    <OnnxContext.Provider value={session}>
      {children}
    </OnnxContext.Provider>
  )
}