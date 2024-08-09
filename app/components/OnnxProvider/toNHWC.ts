import { Tensor } from "onnxruntime-web"

export function toNHWC(t0: Tensor): Tensor {
  const C = t0.dims[1]
  const H = t0.dims[2]
  const W = t0.dims[3]

  const t0Data = t0.data as Float32Array
  const nhwc = new Float32Array(C * H * W)
  
  let rt0Pointer = 0, gt0Pointer = H * W, bt0Pointer = H * W * 2
  let rt1Pointer = 0, gt1Pointer = 1, bt1Pointer = 2

  for (let i = 0; i < H * W; i++, rt0Pointer++, gt0Pointer++, bt0Pointer++, rt1Pointer+=C, gt1Pointer+=C, bt1Pointer+=C) {
    nhwc[rt0Pointer] = t0Data[rt1Pointer]
    nhwc[gt0Pointer] = t0Data[gt1Pointer]
    nhwc[bt0Pointer] = t0Data[bt1Pointer]
  }

  return new Tensor('float32', nhwc, [1, H, W, C])
}