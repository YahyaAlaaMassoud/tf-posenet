'use strict';
var expect = require('chai').expect;
var index = require('../dist/index.js');
describe('posenet loading weights locally test', () => {
    index.load(0.75, true)
         .then(res => {
            console.log(res.mobileNet.variables);
         }),
    index.load(0.75, false)
         .then(res => {
            console.log(res.mobileNet.variables);
         })
});