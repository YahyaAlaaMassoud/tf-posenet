"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tfjs_1 = require("@tensorflow/tfjs");
// import * as fs1 from 'ts-fs';
const fs = require("fs");
const MANIFEST_FILE = 'manifest.json';
class LocalCheckpointLoader {
    constructor(weightsPath) {
        this.weightsPath = weightsPath;
        if (this.weightsPath.charAt(this.weightsPath.length - 1) !== '/') {
            this.weightsPath += '/';
        }
    }
    loadManifest() {
        return new Promise((resolve, reject) => {
            // console.log(this.weightsPath + MANIFEST_FILE);
            try {
                let buff = fs.readFileSync(this.weightsPath + MANIFEST_FILE);
                this.checkpointManifest = JSON.parse(buff.toString());
                resolve();
            }
            catch (error) {
                throw new Error(`${MANIFEST_FILE} not found at ${this.weightsPath}. ${error}`);
            }
            // fs.readFile(this.weightsPath + MANIFEST_FILE)
            //   .then((manifest: any) => {
            //     this.checkpointManifest = JSON.parse(manifest);
            //     resolve();
            //   })
            //   .catch((error: any) => {
            //       console.log(error);
            //     throw new Error(`${MANIFEST_FILE} not found at ${this.weightsPath}. ${error}`);
            //   });
        });
    }
    getCheckpointManifest() {
        if (this.checkpointManifest == null) {
            return new Promise((resolve, reject) => {
                this.loadManifest()
                    .then(() => {
                    resolve(this.checkpointManifest);
                })
                    .catch((error) => {
                    throw new Error(`Error in loading checkpoint manifest file. ${error}`);
                });
            });
        }
        return new Promise((resolve, reject) => {
            resolve(this.checkpointManifest);
        });
    }
    getAllVariables() {
        if (this.variables != null) {
            return new Promise((resolve, reject) => {
                resolve(this.variables);
            });
        }
        return new Promise((resolve, reject) => {
            this.getCheckpointManifest().then((checkpointDefinition) => {
                const variableNames = Object.keys(this.checkpointManifest);
                const variablePromises = [];
                for (let i = 0; i < variableNames.length; i++) {
                    variablePromises.push(this.getVariable(variableNames[i]));
                }
                Promise.all(variablePromises).then(variables => {
                    this.variables = {};
                    for (let i = 0; i < variables.length; i++) {
                        this.variables[variableNames[i]] = variables[i];
                    }
                    resolve(this.variables);
                });
            });
        });
    }
    getVariable(varName) {
        if (!(varName in this.checkpointManifest)) {
            throw new Error('Cannot load non-existant variable ' + varName);
        }
        const variableRequestPromiseMethod = (resolve, reject) => {
            const filename = this.checkpointManifest[varName].filename;
            try {
                let buff = fs.readFileSync(this.weightsPath + 'weights/' + filename + '.json');
                const result = JSON.parse(buff.toString());
                let arr = [];
                for (let r in result) {
                    arr.push(result[r]);
                }
                const values = Float32Array.from(arr);
                const tensor = tfjs_1.Tensor.make(this.checkpointManifest[varName].shape, { values });
                resolve(tensor);
            }
            catch (error) {
                throw new Error(`Could not fetch variable ${varName}: ${error}`);
            }
            // fs1.readFile(this.weightsPath + 'weights/' + filename + '.json')
            //   .then((res: any) => {
            //       const result = JSON.parse(res);
            //       let arr = [];
            //       for(let r in result) {
            //         arr.push(result[r]);
            //       }
            //       const values = Float32Array.from(arr);
            //       const tensor = Tensor.make(this.checkpointManifest[varName].shape, {values});
            //       resolve(tensor);
            //   })
            //   .catch((error: any) => {
            //       throw new Error(`Could not fetch variable ${varName}: ${error}`);
            //   })
        };
        if (this.checkpointManifest == null) {
            return new Promise((resolve, reject) => {
                this.loadManifest()
                    .then(() => {
                    new Promise(variableRequestPromiseMethod).then(resolve);
                })
                    .catch((error) => {
                    throw new Error(`Error in loading checkpoint manifest file. ${error}`);
                });
            });
        }
        return new Promise(variableRequestPromiseMethod);
    }
}
exports.LocalCheckpointLoader = LocalCheckpointLoader;
