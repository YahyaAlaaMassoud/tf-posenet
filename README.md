# posenet
NPM package for PoseNet Pose Estimation Deep Learning Algorithm.

## Installation 
```sh
npm install tf-posenet --save
```

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