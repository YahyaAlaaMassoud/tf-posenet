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
import * as tf from '@tensorflow/tfjs';
export declare type MobileNetMultiplier = 0.50 | 0.75 | 1.0 | 1.01;
export declare type ConvolutionType = 'conv2d' | 'separableConv';
export declare type ConvolutionDefinition = [ConvolutionType, number];
export declare type OutputStride = 32 | 16 | 8;
export declare function assertValidOutputStride(outputStride: any): void;
export declare function assertValidResolution(resolution: any, outputStride: number): void;
export declare function assertValidScaleFactor(imageScaleFactor: any): void;
export declare const mobileNetArchitectures: {
    [name: string]: ConvolutionDefinition[];
};
export declare class MobileNet {
    variables: {
        [varName: string]: tf.Tensor;
    };
    private convolutionDefinitions;
    private PREPROCESS_DIVISOR;
    private ONE;
    constructor(variables: {
        [varName: string]: tf.Tensor;
    }, convolutionDefinitions: ConvolutionDefinition[]);
    predict(input: tf.Tensor3D, outputStride: OutputStride): tf.Tensor<tf.Rank.R3>;
    convToOutput(mobileNetOutput: tf.Tensor3D, outputLayerName: string): tf.Tensor3D;
    private conv;
    private separableConv;
    private weights;
    private biases;
    private depthwiseWeights;
    dispose(): void;
}
