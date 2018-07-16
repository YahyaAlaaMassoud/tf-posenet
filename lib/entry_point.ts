import * as posenet from './index';
import * as fs from 'ts-fs';

// let x = new LocalCheckpointLoader('../../PoseNet - Weights/posenet - weights/mobilenet_v1_050/');
// // x.loadManifest().then(() => {
// //     console.log(x.checkpointManifest["MobilenetV1/Conv2d_0/biases"].filename);
// // });
// x.loadManifest().then(() => {
//     x.getAllVariables().then((res: any) => {
//         for(let r in res) {
//             console.log(r);
//         }
//     });
// });

// // console.log(typeof(x.checkpointManifest))
// console.log(process.cwd());
posenet.load(0.50).then((res: posenet.PoseNet) => {
    // for(let i in res.mobileNet.variables) {
    //     console.log(i);
    // }
});
