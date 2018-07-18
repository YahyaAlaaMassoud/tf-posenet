# tf-posenet
NPM package for Tensorflow's PoseNet Pose Estimation Deep Learning Algorithm.</br>
This package allows you to use the PoseNet Model locally without internet connection to test it.

## Installation 
- Open your terminal and excute this command.
```sh
npm install tf-posenet --save
```
- Then download this compressed [weights file](https://drive.google.com/open?id=13MZkIMUJqyqszcNbrNp14Z5dOUWLgvqo) from Google Drive (~127 MB).
- Create a new directory in you project's folder with name **"assets"**.
- Extract the weight's downloaded file into that new directory "assets".

## Usage
### TypeScript
```typescript
import * as posenet from 'tf-posenet';

async function test(input) {
  var net = await posenet.load(0.75, true);
  var pose_single = net.estimatSinglePose(input);
  var pose_multiple = net.estimatMultiplePose(input);
}
```

## Test 
```sh
npm run test
```
