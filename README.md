# tf-posenet
NPM package for Tensorflow's PoseNet Pose Estimation Deep Learning Algorithm.</br>
This package allows you to use the PoseNet Model locally without internet connection to test it.

## Installation 
```sh
npm install tf-posenet --save
```
Then download this compressed weights [file](https://drive.google.com/open?id=13MZkIMUJqyqszcNbrNp14Z5dOUWLgvqo), and extract it in your project's **node_modules/tf-posenet/dist/** path.

## Usage
### TypeScript
```typescript
import * as posenet from 'tf-posenet';
let x = await posenet.load();
let pose = x.estimateSinglePose(input);
```
```sh
Output should be 
```

## Test 
```sh
npm run test
```
