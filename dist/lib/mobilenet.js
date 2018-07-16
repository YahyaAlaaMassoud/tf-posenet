"use strict";
/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs");
// clang-format off
const mobileNet100Architecture = [
    ['conv2d', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1]
];
const mobileNet75Architecture = [
    ['conv2d', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1]
];
const mobileNet50Architecture = [
    ['conv2d', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1]
];
// clang-format on
const VALID_OUTPUT_STRIDES = [8, 16, 32];
// tslint:disable-next-line:no-any
function assertValidOutputStride(outputStride) {
    tf.util.assert(typeof outputStride === 'number', 'outputStride is not a number');
    tf.util.assert(VALID_OUTPUT_STRIDES.indexOf(outputStride) >= 0, `outputStride of ${outputStride} is invalid. ` +
        `It must be either 8, 16, or 32`);
}
exports.assertValidOutputStride = assertValidOutputStride;
// tslint:disable-next-line:no-any
function assertValidResolution(resolution, outputStride) {
    tf.util.assert(typeof resolution === 'number', 'resolution is not a number');
    tf.util.assert((resolution - 1) % outputStride === 0, `resolution of ${resolution} is invalid for output stride ` +
        `${outputStride}.`);
}
exports.assertValidResolution = assertValidResolution;
// tslint:disable-next-line:no-any
function assertValidScaleFactor(imageScaleFactor) {
    tf.util.assert(typeof imageScaleFactor === 'number', 'imageScaleFactor is not a number');
    tf.util.assert(imageScaleFactor >= 0.2 && imageScaleFactor <= 1.0, 'imageScaleFactor must be between 0.2 and 1.0');
}
exports.assertValidScaleFactor = assertValidScaleFactor;
exports.mobileNetArchitectures = {
    100: mobileNet100Architecture,
    75: mobileNet75Architecture,
    50: mobileNet50Architecture
};
/**
 * Takes a mobilenet architectures' convolution definitions and converts them
 * into definitions for convolutional layers that will generate outputs with the
 * desired output stride. It does this by reducing the input stride in certain
 * layers and applying atrous convolution in subsequent layers. Raises an error
 * if the output stride is not possible with the architecture.
 */
function toOutputStridedLayers(convolutionDefinition, outputStride) {
    // The currentStride variable keeps track of the output stride of
    // the activations, i.e., the running product of convolution
    // strides up to the current network layer. This allows us to
    // invoke atrous convolution whenever applying the next
    // convolution would result in the activations having output
    // stride larger than the target outputStride.
    let currentStride = 1;
    // The atrous convolution rate parameter.
    let rate = 1;
    return convolutionDefinition.map(([convType, stride], blockId) => {
        let layerStride, layerRate;
        if (currentStride === outputStride) {
            // If we have reached the target outputStride, then we need to
            // employ atrous convolution with stride=1 and multiply the atrous
            // rate by the current unit's stride for use in subsequent layers.
            layerStride = 1;
            layerRate = rate;
            rate *= stride;
        }
        else {
            layerStride = stride;
            layerRate = 1;
            currentStride *= stride;
        }
        return {
            blockId,
            convType,
            stride: layerStride,
            rate: layerRate,
            outputStride: currentStride
        };
    });
}
class MobileNet {
    constructor(variables, convolutionDefinitions) {
        this.PREPROCESS_DIVISOR = tf.scalar(255.0 / 2);
        this.ONE = tf.scalar(1);
        this.variables = variables;
        this.convolutionDefinitions = convolutionDefinitions;
    }
    predict(input, outputStride) {
        // Normalize the pixels [0, 255] to be between [-1, 1].
        const preprocessedInput = tf.cast(input, 'float32').div(this.PREPROCESS_DIVISOR).sub(this.ONE);
        const layers = toOutputStridedLayers(this.convolutionDefinitions, outputStride);
        return layers.reduce((previousLayer, { blockId, stride, convType, rate }) => {
            if (convType === 'conv2d') {
                return this.conv(previousLayer, stride, blockId);
            }
            else if (convType === 'separableConv') {
                return this.separableConv(previousLayer, stride, blockId, rate);
            }
            else {
                throw Error(`Unknown conv type of ${convType}`);
            }
        }, preprocessedInput);
    }
    convToOutput(mobileNetOutput, outputLayerName) {
        return mobileNetOutput.conv2d(this.weights(outputLayerName), 1, 'same')
            .add(this.biases(outputLayerName));
    }
    conv(inputs, stride, blockId) {
        return inputs
            .conv2d(this.weights(`Conv2d_${String(blockId)}`), stride, 'same')
            .add(this.biases(`Conv2d_${String(blockId)}`))
            // relu6
            .clipByValue(0, 6);
    }
    separableConv(inputs, stride, blockID, dilations = 1) {
        const dwLayer = `Conv2d_${String(blockID)}_depthwise`;
        const pwLayer = `Conv2d_${String(blockID)}_pointwise`;
        const x1 = inputs
            .depthwiseConv2D(this.depthwiseWeights(dwLayer), stride, 'same', 'NHWC', dilations)
            .add(this.biases(dwLayer))
            // relu6
            .clipByValue(0, 6);
        const x2 = x1.conv2d(this.weights(pwLayer), [1, 1], 'same')
            .add(this.biases(pwLayer))
            // relu6
            .clipByValue(0, 6);
        return x2;
    }
    weights(layerName) {
        return this.variables[`MobilenetV1/${layerName}/weights`];
    }
    biases(layerName) {
        return this.variables[`MobilenetV1/${layerName}/biases`];
    }
    depthwiseWeights(layerName) {
        return this.variables[`MobilenetV1/${layerName}/depthwise_weights`];
    }
    dispose() {
        for (const varName in this.variables) {
            this.variables[varName].dispose();
        }
    }
}
exports.MobileNet = MobileNet;
