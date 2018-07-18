'use strict';
var expect = require('chai').expect;
var index = require('../dist/index.js');
describe('PoseNet loading weights locally and from API test', () => {
    // it('should return model from API request', () => {
    //     index.load(0.75, false)
    //          .then((res) => {
    //              console.log('done API');
    //          })
    //          .catch((error) => {
    //              console.log(error);
    //          });
    //     // expect(result).to.equal('Boys');
    // });
    it('should return model from local disk', () => {
        index.load(0.75, true)
             .then((res) => {
                 console.log('done local');
                //  console.log(res);
             })
             .catch((error) => {
                 console.log(error);
             });
        // expect(result).to.equal('Girls');
    });
});