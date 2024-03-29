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
export declare type Tuple<T> = [T, T];
export declare type StringTuple = Tuple<string>;
export declare type NumberTuple = Tuple<number>;
export declare const partNames: string[];
export declare const NUM_KEYPOINTS: number;
export interface NumberDict {
    [jointName: string]: number;
}
export declare const partIds: NumberDict;
export declare const poseChain: StringTuple[];
export declare const connectedPartIndices: number[][];
