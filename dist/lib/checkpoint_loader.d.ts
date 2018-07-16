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
import { Tensor } from '@tensorflow/tfjs';
/**
 * @hidden
 */
export interface CheckpointVariable {
    filename: string;
    shape: number[];
}
export interface CheckpointVariableFile {
    values: Float32Array;
}
/**
 * @hidden
 */
export declare type CheckpointManifest = {
    [varName: string]: CheckpointVariable;
};
export declare class CheckpointLoader {
    private urlPath;
    private checkpointManifest;
    private variables;
    constructor(urlPath: string);
    private loadManifest;
    getCheckpointManifest(): Promise<CheckpointManifest>;
    getAllVariables(): Promise<{
        [varName: string]: Tensor;
    }>;
    getVariable(varName: string): Promise<Tensor>;
    saveWeights(filename: any, values: any, shape: any): void;
}
