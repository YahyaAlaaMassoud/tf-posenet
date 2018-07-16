
import { Tensor } from '@tensorflow/tfjs';
import { CheckpointManifest } from './checkpoint_loader';
import * as fs from 'ts-fs';

const MANIFEST_FILE = 'manifest.json';

export class LocalCheckpointLoader {

    public checkpointManifest!: CheckpointManifest;
    private variables!: {[varName: string]: Tensor};

    constructor(private weightsPath: string) {
        if (this.weightsPath.charAt(this.weightsPath.length - 1) !== '/') {
            this.weightsPath += '/';
        }
    }

    public loadManifest(): Promise<void> {
        return new Promise<void>((resolve, reject) => {
            console.log(this.weightsPath + MANIFEST_FILE);
            debugger;
            fs.readFile(this.weightsPath + MANIFEST_FILE)
              .then((manifest: any) => {
                this.checkpointManifest = JSON.parse(manifest);
                resolve();
              })
              .catch((error: any) => {
                  console.log(error);
                throw new Error(`${MANIFEST_FILE} not found at ${this.weightsPath}. ${error}`);
              });
        });
    }

    getCheckpointManifest(): Promise<CheckpointManifest> {
        if (this.checkpointManifest == null) {
            return new Promise<CheckpointManifest>((resolve, reject) => {
                this.loadManifest()
                    .then(() => {
                        resolve(this.checkpointManifest);
                    });
            });
        }
        return new Promise<CheckpointManifest>((resolve, reject) => {
            resolve(this.checkpointManifest);
        });
    }

    getAllVariables(): Promise<{[varName: string]: Tensor}> {
        if (this.variables != null) {
            return new Promise<{[varName: string]: Tensor}>((resolve, reject) => {
                resolve(this.variables);
            });
        }
    
        return new Promise<{[varName: string]: Tensor}>((resolve, reject) => {
            this.getCheckpointManifest().then(
                (checkpointDefinition: CheckpointManifest) => {
                    const variableNames = Object.keys(this.checkpointManifest);
        
                    const variablePromises: Array<Promise<Tensor>> = [];
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

    getVariable(varName: string): Promise<Tensor> {
        if (!(varName in this.checkpointManifest)) {
            throw new Error('Cannot load non-existant variable ' + varName);
        }
    
        const variableRequestPromiseMethod =
            (resolve: (tensor: Tensor) => void, reject: () => void) => {
                const filename = this.checkpointManifest[varName].filename;
                fs.readFile(this.weightsPath + 'weights/' + filename + '.json')
                  .then((res: any) => {
                      const result = JSON.parse(res);

                      let arr = [];
                      for(let r in result) {
                        arr.push(result[r]);
                      }

                      const values = Float32Array.from(arr);
                      const tensor = Tensor.make(this.checkpointManifest[varName].shape, {values});

                      resolve(tensor);
                  })
                  .catch((error: any) => {
                      throw new Error(`Could not fetch variable ${varName}: ${error}`);
                  })
            };
    
        if (this.checkpointManifest == null) {
            return new Promise<Tensor>((resolve, reject) => {
                this.loadManifest()
                    .then(() => {
                        new Promise<Tensor>(variableRequestPromiseMethod).then(resolve);
                    });
            });
        }
        return new Promise<Tensor>(variableRequestPromiseMethod);
    }
}
