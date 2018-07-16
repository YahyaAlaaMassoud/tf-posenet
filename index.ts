/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {decodeMultiplePoses} from './lib/multiPose/decodeMultiplePoses';
import {load, PoseNet} from './lib/posenet_model';
import {decodeSinglePose} from './lib/singlePose/decodeSinglePose';

export {checkpoints, localCheckpoints} from './checkpoints';
export {partIds, partNames, poseChain} from './lib/keypoints';
export {Keypoint, Pose} from './lib/types';
// tslint:disable-next-line:max-line-length
export {getAdjacentKeyPoints, getBoundingBox, getBoundingBoxPoints} from './lib/util';
export {decodeMultiplePoses, decodeSinglePose};
export {load, PoseNet};
