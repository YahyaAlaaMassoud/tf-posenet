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
const tf = require("@tensorflow/tfjs");
function mod(a, b) {
    return tf.tidy(() => {
        const floored = a.div(tf.scalar(b, 'int32'));
        return a.sub(floored.mul(tf.scalar(b, 'int32')));
    });
}
function argmax2d(inputs) {
    const [height, width, depth] = inputs.shape;
    return tf.tidy(() => {
        const reshaped = inputs.reshape([height * width, depth]);
        const coords = reshaped.argMax(0);
        const yCoords = coords.div(tf.scalar(width, 'int32')).expandDims(1);
        const xCoords = mod(coords, width).expandDims(1);
        return tf.concat([yCoords, xCoords], 1);
    });
}
exports.argmax2d = argmax2d;
