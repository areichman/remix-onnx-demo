import { OnnxProvider } from '~/components/OnnxProvider/OnnxProvider'
import { OnnxImage } from '~/components/OnnxProvider/OnnxImage'
import { getPredictedClass } from '~/utils/imagenet'

const models = {
  squeezenet: '/models/squeezenet1_1.onnx',
  mobilenet: '/models/mobilenetv2-7.onnx',
  internal: '/models/internal/internal.onnx',
}

const urls = [
  '/images/webb-pillars.jpg',
  '/images/spongebob.jpg',
  '/images/rock.jpg',
  '/images/dogs.jpg',
]

// <OnnxProvider url={models.squeezenet}>
// <OnnxProvider url={models.internal} executionProvider="wasm" width={384} height={384} reverseDimensions={true}>

export default function Route() {
  const url = models.squeezenet

  return (
    <OnnxProvider url={url}>
      <p className="mb-5">{url}</p>

      <div className="container flex flex-row flex-wrap gap-5">
        {urls.map(url => (
          <div key={url}>
            <OnnxImage src={url}>
              {(results) => {
                if (!results) {
                  return <p>Running model...</p>
                }

                // squeezenet only
                const {name, probability} = getPredictedClass(results.output)[0];
                
                return (
                  <pre className="my-2 text-xs">
                    {/* <p>{JSON.stringify(results.output, null, 2)}</p> */}
                    <p>{name} ({(probability*100).toFixed(1)}%)</p>
                    <p>{results.time} msec</p>
                  </pre>
                )
              }}
            </OnnxImage>
          </div>
        ))}
      </div>
     
    </OnnxProvider>
  )
}