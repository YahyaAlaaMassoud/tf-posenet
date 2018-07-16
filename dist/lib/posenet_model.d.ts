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
import { MobileNet, MobileNetMultiplier, OutputStride } from './mobilenet';
import { Pose } from './types';
export declare type PoseNetResolution = 161 | 193 | 257 | 289 | 321 | 353 | 385 | 417 | 449 | 481 | 513;
export declare type InputType = ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | tf.Tensor3D;
export declare class PoseNet {
    mobileNet: MobileNet;
    constructor(mobileNet: MobileNet);
    /**
     * Infer through PoseNet. This does standard ImageNet pre-processing before
     * inferring through the model. The image should pixels should have values
     * [0-255]. This method returns the heatmaps and offsets.  Infers through the
     * outputs that are needed for single pose decoding
     *
     * @param input un-preprocessed input image, with values in range [0-255]
     * @param outputStride the desired stride for the outputs.  Must be 32, 16,
     * or 8. Defaults to 16.  The output width and height will be will be
     * (inputDimension - 1)/outputStride + 1
     * @return heatmapScores, offsets
     */
    predictForSinglePose(input: tf.Tensor3D, outputStride?: OutputStride): {
        heatmapScores: tf.Tensor3D;
        offsets: tf.Tensor3D;
    };
    /**
     * Infer through PoseNet. This does standard ImageNet pre-processing before
     * inferring through the model. The image should pixels should have values
     * [0-255]. Infers through the outputs that are needed for multiple pose
     * decoding. This method returns the heatmaps offsets, and mid-range
     * displacements.
     *
     * @param input un-preprocessed input image, with values in range [0-255]
     * @param outputStride the desired stride for the outputs.  Must be 32, 16,
     * or 8. Defaults to 16. The output width and height will be will be
     * (inputDimension - 1)/outputStride + 1
     * @return heatmapScores, offsets, displacementFwd, displacementBwd
     */
    predictForMultiPose(input: tf.Tensor3D, outputStride?: OutputStride): {
        heatmapScores: tf.Tensor3D;
        offsets: tf.Tensor3D;
        displacementFwd: tf.Tensor3D;
        displacementBwd: tf.Tensor3D;
    };
    /**
     * Infer through PoseNet, and estimates a single pose using the outputs. This
     * does standard ImageNet pre-processing before inferring through the model.
     * The image should pixels should have values [0-255].
     * This method returns a single pose.
     *
     * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
     * The input image to feed through the network.
     *
     * @param imageScaleFactor A number between 0.2 and 1. Defaults to 0.50. What
     * to scale the image by before feeding it through the network.  Set this
     * number lower to scale down the image and increase the speed when feeding
     * through the network at the cost of accuracy.
     *
     * @param flipHorizontal.  Defaults to false.  If the poses should be
     * flipped/mirrored  horizontally.  This should be set to true for videos
     * where the video is by default flipped horizontally (i.e. a webcam), and you
     * want the poses to be returned in the proper orientation.
     *
     * @param outputStride the desired stride for the outputs.  Must be 32, 16,
     * or 8. Defaults to 16. The output width and height will be will be
     * (inputDimension - 1)/outputStride + 1
     * @return A single pose with a confidence score, which contains an array of
     * keypoints indexed by part id, each with a score and position.  The
     * positions of the keypoints are in the same scale as the original image
     */
    estimateSinglePose(input: InputType, imageScaleFactor?: number, flipHorizontal?: boolean, outputStride?: OutputStride): Promise<Pose>;
    /**
     * Infer through PoseNet, and estimates multiple poses using the outputs.
     * This does standard ImageNet pre-processing before inferring through the
     * model. The image should pixels should have values [0-255]. It detects
     * multiple poses and finds their parts from part scores and displacement
     * vectors using a fast greedy decoding algorithm.  It returns up to
     * `maxDetections` object instance detections in decreasing root score order.
     *
     * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
     * The input image to feed through the network.
     *
     * @param imageScaleFactor  A number between 0.2 and 1. Defaults to 0.50. What
     * to scale the image by before feeding it through the network.  Set this
     * number lower to scale down the image and increase the speed when feeding
     * through the network at the cost of accuracy.
     *
     * @param flipHorizontal Defaults to false.  If the poses should be
     * flipped/mirrored  horizontally.  This should be set to true for videos
     * where the video is by default flipped horizontally (i.e. a webcam), and you
     * want the poses to be returned in the proper orientation.
     *
     * @param outputStride the desired stride for the outputs.  Must be 32, 16,
     * or 8. Defaults to 16. The output width and height will be will be
     * (inputSize - 1)/outputStride + 1
     *
     * @param maxDetections Maximum number of returned instance detections per
     * image. Defaults to 5.
     *
     * @param scoreThreshold Only return instance detections that have root part
     * score greater or equal to this value. Defaults to 0.5
     *
     * @param nmsRadius Non-maximum suppression part distance in pixels. It needs
     * to be strictly positive. Two parts suppress each other if they are less
     * than `nmsRadius` pixels away. Defaults to 20.
     *
     * @return An array of poses and their scores, each containing keypoints and
     * the corresponding keypoint scores.  The positions of the keypoints are
     * in the same scale as the original image
     */
    estimateMultiplePoses(input: InputType, imageScaleFactor?: number, flipHorizontal?: boolean, outputStride?: OutputStride, maxDetections?: number, scoreThreshold?: number, nmsRadius?: number): Promise<Pose[]>;
    dispose(): void;
}
/**
 * Loads the PoseNet model instance from a checkpoint, with the MobileNet
 * architecture specified by the multiplier.
 *
 * @param multiplier An optional number with values: 1.01, 1.0, 0.75, or
 * 0.50. Defaults to 1.01. It is the float multiplier for the depth (number of
 * channels) for all convolution ops. The value corresponds to a MobileNet
 * architecture and checkpoint.  The larger the value, the larger the size of
 * the layers, and more accurate the model at the cost of speed.  Set this to a
 * smaller value to increase speed at the cost of accuracy.
 *
 */
export declare function load(multiplier?: MobileNetMultiplier, local?: boolean): Promise<PoseNet>;
export declare const mobilenetLoader: {
    loadLocal: (multiplier: MobileNetMultiplier) => Promise<MobileNet>;
    loadFromAPI: (multiplier: MobileNetMultiplier) => Promise<MobileNet>;
};
