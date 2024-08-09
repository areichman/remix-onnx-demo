/* Sample usage:
<OnnxImage src="foo.jpg">
  {(results) => (
    <div>
      <p>{JSON.stringify(results.output, null, 2}</p>
      <p>{results.time} msec</p>
    </div>
  )}
</OnnxImage>
*/

import { createRef, useState } from "react"

import { useModel } from "./OnnxProvider"

export interface Results {
  output: Float32Array,
  time: number
}

interface Props {
  src: string
  size?: number
  children: (results: Results) => React.ReactNode
}

export function OnnxImage({src, size = 224, children}: Props) {
  const ref = createRef<HTMLImageElement>()
  const model = useModel()
  const [results, setResults] = useState<Results | undefined>()

  const runModel = async () => {
    if (model.session && ref.current) {
      const results = await model.run(ref.current)
      setResults(results)
    } else {
      console.error('ONNX session not available')
    }
  }

  return (
    <div>
      <img ref={ref} alt="onnx-image" src={src} style={{maxWidth: size, maxHeight: size}} onLoad={runModel}/>
      {!results && <p>Running model...</p>}
      {results && children(results)}
    </div>
  )
}