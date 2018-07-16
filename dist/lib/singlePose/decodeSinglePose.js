"use strict";
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
const keypoints_1 = require("../keypoints");
const util_1 = require("../util");
const argmax2d_1 = require("./argmax2d");
const util_2 = require("./util");
/**
 * Detects a single pose and finds its parts from part scores and offset
 * vectors. It returns a single pose detection. It works as follows:
 * argmax2d is done on the scores to get the y and x index in the heatmap
 * with the highest score for each part, which is essentially where the
 * part is most likely to exist. This produces a tensor of size 17x2, with
 * each row being the y and x index in the heatmap for each keypoint.
 * The offset vector for each for each part is retrieved by getting the
 * y and x from the offsets corresponding to the y and x index in the
 * heatmap for that part. This produces a tensor of size 17x2, with each
 * row being the offset vector for the corresponding keypoint.
 * To get the keypoint, each part’s heatmap y and x are multiplied
 * by the output stride then added to their corresponding offset vector,
 * which is in the same scale as the original image.
 *
 * @param heatmapScores 3-D tensor with shape `[height, width, numParts]`.
 * The value of heatmapScores[y, x, k]` is the score of placing the `k`-th
 * object part at position `(y, x)`.
 *
 * @param offsets 3-D tensor with shape `[height, width, numParts * 2]`.
 * The value of [offsets[y, x, k], offsets[y, x, k + numParts]]` is the
 * short range offset vector of the `k`-th  object part at heatmap
 * position `(y, x)`.
 *
 * @param outputStride The output stride that was used when feed-forwarding
 * through the PoseNet model.  Must be 32, 16, or 8.
 *
 * @return A promise that resolves with single pose with a confidence score,
 * which contains an array of keypoints indexed by part id, each with a score
 * and position.
 */
function decodeSinglePose(heatmapScores, offsets, outputStride) {
    return __awaiter(this, void 0, void 0, function* () {
        let totalScore = 0.0;
        const heatmapValues = argmax2d_1.argmax2d(heatmapScores);
        const [scoresBuffer, offsetsBuffer, heatmapValuesBuffer] = yield Promise.all([
            util_1.toTensorBuffer(heatmapScores), util_1.toTensorBuffer(offsets),
            util_1.toTensorBuffer(heatmapValues, 'int32')
        ]);
        const offsetPoints = util_2.getOffsetPoints(heatmapValuesBuffer, outputStride, offsetsBuffer);
        const offsetPointsBuffer = yield util_1.toTensorBuffer(offsetPoints);
        const keypointConfidence = Array.from(util_2.getPointsConfidence(scoresBuffer, heatmapValuesBuffer));
        const keypoints = keypointConfidence.map((score, keypointId) => {
            totalScore += score;
            return {
                position: {
                    y: offsetPointsBuffer.get(keypointId, 0),
                    x: offsetPointsBuffer.get(keypointId, 1)
                },
                part: keypoints_1.partNames[keypointId],
                score
            };
        });
        heatmapValues.dispose();
        offsetPoints.dispose();
        return { keypoints, score: totalScore / keypoints.length };
    });
}
exports.decodeSinglePose = decodeSinglePose;
