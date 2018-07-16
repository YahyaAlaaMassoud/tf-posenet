import { Tensor } from '@tensorflow/tfjs';
import { CheckpointManifest } from './checkpoint_loader';
export declare class LocalCheckpointLoader {
    private weightsPath;
    checkpointManifest: CheckpointManifest;
    private variables;
    constructor(weightsPath: string);
    loadManifest(): Promise<void>;
    getCheckpointManifest(): Promise<CheckpointManifest>;
    getAllVariables(): Promise<{
        [varName: string]: Tensor;
    }>;
    getVariable(varName: string): Promise<Tensor>;
}
