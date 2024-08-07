import { InferenceSession } from "onnxruntime-web"
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
  executionProviders?: ('webgl' | 'wasm')[]
}

export function OnnxProvider({
  url, 
  width = 224, 
  height = 224, 
  reverseDimensions = false,
  executionProviders = ['webgl'],
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
        const session = await InferenceSession.create(model, {executionProviders})
        setSession(session)
      }

      createSession()
    }, 
    [] // eslint-disable-line react-hooks/exhaustive-deps
  )

  if (!session) {
    return null
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