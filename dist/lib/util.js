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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs");
const keypoints_1 = require("./keypoints");
function eitherPointDoesntMeetConfidence(a, b, minConfidence) {
    return (a < minConfidence || b < minConfidence);
}
function getAdjacentKeyPoints(keypoints, minConfidence) {
    return keypoints_1.connectedPartIndices.reduce((result, [leftJoint, rightJoint]) => {
        if (eitherPointDoesntMeetConfidence(keypoints[leftJoint].score, keypoints[rightJoint].score, minConfidence)) {
            return result;
        }
        result.push([keypoints[leftJoint], keypoints[rightJoint]]);
        return result;
    }, []);
}
exports.getAdjacentKeyPoints = getAdjacentKeyPoints;
const { NEGATIVE_INFINITY, POSITIVE_INFINITY } = Number;
function getBoundingBox(keypoints) {
    return keypoints.reduce(({ maxX, maxY, minX, minY }, { position: { x, y } }) => {
        return {
            maxX: Math.max(maxX, x),
            maxY: Math.max(maxY, y),
            minX: Math.min(minX, x),
            minY: Math.min(minY, y)
        };
    }, {
        maxX: NEGATIVE_INFINITY,
        maxY: NEGATIVE_INFINITY,
        minX: POSITIVE_INFINITY,
        minY: POSITIVE_INFINITY
    });
}
exports.getBoundingBox = getBoundingBox;
function getBoundingBoxPoints(keypoints) {
    const { minX, minY, maxX, maxY } = getBoundingBox(keypoints);
    return [
        { x: minX, y: minY }, { x: maxX, y: minY }, { x: maxX, y: maxY },
        { x: minX, y: maxY }
    ];
}
exports.getBoundingBoxPoints = getBoundingBoxPoints;
function toTensorBuffer(tensor, type = 'float32') {
    return __awaiter(this, void 0, void 0, function* () {
        const tensorData = yield tensor.data();
        return new tf.TensorBuffer(tensor.shape, type, tensorData);
    });
}
exports.toTensorBuffer = toTensorBuffer;
function toTensorBuffers3D(tensors) {
    return __awaiter(this, void 0, void 0, function* () {
        return Promise.all(tensors.map(tensor => toTensorBuffer(tensor, 'float32')));
    });
}
exports.toTensorBuffers3D = toTensorBuffers3D;
function scalePose(pose, scaleX, scaleY) {
    return {
        score: pose.score,
        keypoints: pose.keypoints.map(({ score, part, position }) => ({
            score,
            part,
            position: { x: position.x * scaleX, y: position.y * scaleY }
        }))
    };
}
exports.scalePose = scalePose;
function scalePoses(poses, scaleY, scaleX) {
    if (scaleX === 1 && scaleY === 1) {
        return poses;
    }
    return poses.map(pose => scalePose(pose, scaleX, scaleY));
}
exports.scalePoses = scalePoses;
function getValidResolution(imageScaleFactor, inputDimension, outputStride) {
    const evenResolution = inputDimension * imageScaleFactor - 1;
    return evenResolution - (evenResolution % outputStride) + 1;
}
exports.getValidResolution = getValidResolution;
