import {isTypedArray, reverse, take, sortBy} from 'lodash-es';
import {imagenetClasses} from '~/data/imagenet';


function softmax(arr: number[]) {
  const C = Math.max(...arr);
  const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
  return arr.map((value) => {
    return Math.exp(value - C) / d;
  });
}

/**
 * Find top k imagenet classes
 */
function imagenetClassesTopK(classProbabilities: any, k = 5) {
  const probs =
      isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities;

  const sorted = reverse(sortBy(probs.map((prob: any, index: number) => [prob, index]), probIndex => probIndex[0]));

  const topK = take(sorted, k).map(probIndex => {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1], 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0]
    };
  });
  return topK;
}

export function getPredictedClass(res: Float32Array) {
  if (!res || res.length === 0) {
    const empty = [];
    for (let i = 0; i < 5; i++) {
      empty.push({ name: "-", probability: 0, index: 0 });
    }
    return empty;
  }
  const output = softmax(Array.prototype.slice.call(res));
  return imagenetClassesTopK(output, 5);
}