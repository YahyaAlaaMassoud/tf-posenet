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
Object.defineProperty(exports, "__esModule", { value: true });
const keypoints_1 = require("../keypoints");
const util_1 = require("./util");
const util_2 = require("./util");
const parentChildrenTuples = keypoints_1.poseChain.map(([parentJoinName, childJoinName]) => ([keypoints_1.partIds[parentJoinName], keypoints_1.partIds[childJoinName]]));
const parentToChildEdges = parentChildrenTuples.map(([, childJointId]) => childJointId);
const childToParentEdges = parentChildrenTuples.map(([parentJointId,]) => parentJointId);
function getDisplacement(edgeId, point, displacements) {
    const numEdges = displacements.shape[2] / 2;
    return {
        y: displacements.get(point.y, point.x, edgeId),
        x: displacements.get(point.y, point.x, numEdges + edgeId)
    };
}
function getStridedIndexNearPoint(point, outputStride, height, width) {
    return {
        y: util_1.clamp(Math.round(point.y / outputStride), 0, height - 1),
        x: util_1.clamp(Math.round(point.x / outputStride), 0, width - 1)
    };
}
/**
 * We get a new keypoint along the `edgeId` for the pose instance, assuming
 * that the position of the `idSource` part is already known. For this, we
 * follow the displacement vector from the source to target part (stored in
 * the `i`-t channel of the displacement tensor).
 */
function traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scoresBuffer, offsets, outputStride, displacements) {
    const [height, width] = scoresBuffer.shape;
    // Nearest neighbor interpolation for the source->target displacements.
    const sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypoint.position, outputStride, height, width);
    const displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements);
    const displacedPoint = util_2.addVectors(sourceKeypoint.position, displacement);
    const displacedPointIndices = getStridedIndexNearPoint(displacedPoint, outputStride, height, width);
    const offsetPoint = util_1.getOffsetPoint(displacedPointIndices.y, displacedPointIndices.x, targetKeypointId, offsets);
    const score = scoresBuffer.get(displacedPointIndices.y, displacedPointIndices.x, targetKeypointId);
    const targetKeypoint = util_2.addVectors({
        x: displacedPointIndices.x * outputStride,
        y: displacedPointIndices.y * outputStride
    }, { x: offsetPoint.x, y: offsetPoint.y });
    return { position: targetKeypoint, part: keypoints_1.partNames[targetKeypointId], score };
}
/**
 * Follows the displacement fields to decode the full pose of the object
 * instance given the position of a part that acts as root.
 *
 * @return An array of decoded keypoints and their scores for a single pose
 */
function decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd) {
    const numParts = scores.shape[2];
    const numEdges = parentToChildEdges.length;
    const instanceKeypoints = new Array(numParts);
    // Start a new detection instance at the position of the root.
    const { part: rootPart, score: rootScore } = root;
    const rootPoint = util_2.getImageCoords(rootPart, outputStride, offsets);
    instanceKeypoints[rootPart.id] = {
        score: rootScore,
        part: keypoints_1.partNames[rootPart.id],
        position: rootPoint
    };
    // Decode the part positions upwards in the tree, following the backward
    // displacements.
    for (let edge = numEdges - 1; edge >= 0; --edge) {
        const sourceKeypointId = parentToChildEdges[edge];
        const targetKeypointId = childToParentEdges[edge];
        if (instanceKeypoints[sourceKeypointId] &&
            !instanceKeypoints[targetKeypointId]) {
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsBwd);
        }
    }
    // Decode the part positions downwards in the tree, following the forward
    // displacements.
    for (let edge = 0; edge < numEdges; ++edge) {
        const sourceKeypointId = childToParentEdges[edge];
        const targetKeypointId = parentToChildEdges[edge];
        if (instanceKeypoints[sourceKeypointId] &&
            !instanceKeypoints[targetKeypointId]) {
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsFwd);
        }
    }
    return instanceKeypoints;
}
exports.decodePose = decodePose;
