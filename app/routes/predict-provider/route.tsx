import { OnnxProvider } from '~/components/OnnxProvider/OnnxProvider'
import { OnnxImage } from '~/components/OnnxProvider/OnnxImage'

const models = {
  squeezenet: '/models/squeezenet1_1.onnx',
  mobilenet: '/models/mobilenetv2-7.onnx',
  thorn: '/models/model.onnx',
}

export default function Route() {
  return (
    <OnnxProvider url={models.squeezenet}>
      <OnnxImage src="/images/webb-pillars.jpg" />
    </OnnxProvider>
  )

  /*
  return (
    <OnnxProvider url={models.thorn} executionProviders={['wasm']} width={384} height={384} reverseDimensions={true}>
      <OnnxImage src="/images/webb-pillars.jpg" />
    </OnnxProvider>
  )
  */
}