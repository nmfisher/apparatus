  /*
  Copyright (c) 2012 Andrej Karpathy

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  */

  var util = require('util'),
       Classifier = require('./classifier');


  var RandomForestClassifier = function(options) {
      Classifier.call(this);
      this.options = options;
      this.examples = [];
      this.example_size = 0;
      this.labels = [];
      this.labelLookup = [];
  };

  /*
   data is 2D array of size N x D of examples
   labels is a 1D array of labels (only -1 or 1 for now). In future will support multiclass or maybe even regression
   options.numTrees can be used to customize number of trees to train (default = 100)
   options.maxDepth is the maximum depth of each tree in the forest (default = 4)
   options.numTries is the number of random hypotheses generated at each node during training (default = 10)
   options.trainFun is a function with signature "function myWeakTrain(data, labels, ix, options)". Here, ix is a list of
                    indexes into data of the instances that should be payed attention to. Everything not in the list
                    should be ignored. This is done for efficiency. The function should return a model where you store
                    variables. (i.e. model = {}; model.myvar = 5;) This will be passed to testFun.
   options.testFun is a function with signature "funtion myWeakTest(inst, model)" where inst is 1D array specifying an example,
                    and model will be the same model that you return in options.trainFun. For example, model.myvar will be 5.
                    see decisionStumpTrain() and decisionStumpTest() below for example.
  */
  function trainRandomForest(data, labels, numClasses, options) {
      options = options || {};
      this.numTrees = options.numTrees || 100;
      
      // initialize many trees and train them all independently
      this.trees= new Array(this.numTrees);
      for(var i=0;i<this.numTrees;i++) {
          this.trees[i] = new DecisionTree(options, numClasses);
          this.trees[i].train(data, labels, options);
      }
  }

  /*
   inst is a 1D array of length D of an example.
   returns the probability of label 1, i.e. a number in range [0, 1]
  */
  function predictOne(inst) {
        // have each tree predict and average out all votes
        var dec= zeros(this.numClasses);
        for(var i=0;i<this.numTrees;i++) {
          var probs = this.trees[i].predictOne(inst);
          for(var j=0;j<this.numClasses;j++) {
            dec[j] += probs[j]
          }
        }
        for(var i=0;i<this.numClasses;i++) {
          dec[i] /= this.numTrees;
        }

        return dec;
  }

  /*
   convenience function. Here, data is NxD array.
   returns probabilities of being 1 for all data in an array.
  */
  function predict(data) {
      var probabilities= new Array(data.length);
      for(var i=0;i<data.length;i++) {
          probabilities[i]= this.predictOne(data[i]);
      }
      return probabilities;
  }

  // represents a single decision tree
  var DecisionTree = function(options, numClasses) {
    this.numClasses = numClasses;
  };

  function trainDT(data, labels, options) {
      options = options || {};
      var maxDepth = options.maxDepth || 4;
      var weakType = options.type || 0;

      var trainFun = decision2DStumpTrain;
      var testFun = decision2DStumpTest;

      if(options.trainFun) trainFun = options.trainFun;
      if(options.testFun) testFun = options.testFun;

      if(weakType == 0) {
          trainFun= decisionStumpTrain;
          testFun= decisionStumpTest;
      }
      if(weakType == 1) {
          trainFun= decision2DStumpTrain;
          testFun= decision2DStumpTest;
      }

      // initialize various helper variables
      var numInternals= Math.pow(2, maxDepth)-1;
      var numNodes= Math.pow(2, maxDepth + 1)-1;
      var ixs= new Array(numNodes);
      for(var i=1;i<ixs.length;i++) ixs[i]=[];
      ixs[0]= new Array(labels.length);
      for(var i=0;i<labels.length;i++) {
        ixs[0][i]= i; // root node starts out with all nodes as relevant
      }
      var models = new Array(numInternals);

      // train
      for(var n=0; n < numInternals; n++) {

          // few base cases
          var ixhere= ixs[n];
          if(ixhere.length == 0) { continue; }
          if(ixhere.length == 1) { ixs[n*2+1] = [ixhere[0]]; continue; } // arbitrary send it down left

          // learn a weak model on relevant data for this node
          var model = trainFun.call(this, data, labels, ixhere);
          models[n]= model; // back it up model

          // split the data according to the learned model
          var ixleft=[];
          var ixright=[];
          for(var i=0; i<ixhere.length;i++) {
              var label= testFun(data[ixhere[i]], model);
              if(label === 1) ixleft.push(ixhere[i]);
              else ixright.push(ixhere[i]);
              
          }
          ixs[n*2+1]= ixleft;
          ixs[n*2+2]= ixright;
      }

      // compute data distributions at the leafs
      var leafDistributions = new Array(numNodes);
      for(var n=numInternals; n < numNodes; n++) {
          leafDistributions[n] = zeros(this.numClasses)
          for(var i=0;i<ixs[n].length;i++) {
            leafDistributions[n][labels[ixs[n][i]]] += 1;
          }
      }

      // back up important prediction variables for predicting later
      this.models= models;
      this.leafDistributions = leafDistributions;
      this.maxDepth= maxDepth;
      this.trainFun = trainFun;
      this.testFun = testFun;
  }


  // returns probability that example inst is 1.
  function predictOneDT(inst) {

      var n=0;
      for(var i=0;i<this.maxDepth;i++) {
          var dir= this.testFun(inst, this.models[n]);
          if(dir === 1) n= n*2+1; // descend left
          else n= n*2+2; // descend right
      }
      var probs = new Array(this.numClasses);
      // get total number of labels at this leaf
      var total = this.leafDistributions[n].reduce(add, 0);
      for (var i=0;i<this.numClasses;i++) {
        probs[i] = (this.leafDistributions[n][i] + 1) /(total + 2); // bayesian smoothing!
      }
      return probs;
  }

  // returns model
  function decisionStumpTrain(data, labels, ix, options) {
        options = options || {};
        var numtries = options.numTries || 10;

        // choose a dimension at random and pick a best split
        var ri = randi(0, data[0].length);
        var N = ix.length;

        // evaluate class entropy of incoming data
        var H = entropy(labels, ix, this.numClasses);
        var bestGain = 0;
        var bestThr = 0;
        for(var i=0;i<numtries;i++) {
            // pick a random splitting threshold
            var ix1 = ix[randi(0, N)];
            var ix2 = ix[randi(0, N)];
            while(ix2==ix1) ix2= ix[randi(0, N)]; // enforce distinctness of ix2

            var a= Math.random();
            var thr= data[ix1][ri]*a + data[ix2][ri]*(1-a);

            var counts = zeros(this.numClasses * 2);

            let ltotal = 0, rtotal = 0;
            // measure information gain we'd get from split with thr
            for(var j=0;j<ix.length;j++) {
              if(data[ix[j]][ri] < thr) {
                counts[labels[j]]++;
                ltotal++;
              } else {
                counts[this.numClasses + labels[j]]++;
                rtotal++;
              }
            }
            let LH = 0, lp = 0, RH=0, rp=0;          
            for (var j = 0; j < this.numClasses; j++) {
                lp = counts[j] /= ltotal;
                LH -= lp*Math.log(lp);
            }
          
            for (var j = this.numClasses; j < counts.length; j++) {
                rp = counts[j] /= rtotal;
                RH -= rp*Math.log(rp);
            }          

            var informationGain = H - LH - RH;
            //console.log("Considering split %f, entropy %f -> %f, %f. Gain %f", thr, H, LH, RH, informationGain);
            if(informationGain > bestGain || i === 0) {
              bestGain= informationGain;
              bestThr= thr;
            }
        }



        model= {};
        model.thr= bestThr;
        model.ri= ri;
        return model;
}

  // returns a decision for a single data instance
  function decisionStumpTest(inst, model) {
      if(!model) {
          // this is a leaf that never received any data...
          return 1;
      }
      return inst[model.ri] < model.thr ? 1 : -1;
  }

  // returns model. Code duplication with decisionStumpTrain :(
  function decision2DStumpTrain(data, labels, ix, options) {

      options = options || {};
      var numtries = options.numTries || 10;

      // choose a dimension at random and pick a best split
      var N = ix.length;

      var ri1= 0;
      var ri2= 1;
      if(data[0].length > 2) {
        // more than 2D data. Pick 2 random dimensions
        ri1= randi(0, data[0].length);
        ri2= randi(0, data[0].length);
        while(ri2 == ri1) ri2= randi(0, data[0].length); // must be distinct!
      }

      // evaluate class entropy of incoming data
      var H = entropy(labels, ix, this.numClasses);
      var bestGain=0;
      var bestw1, bestw2, bestthr;
      var dots= new Array(ix.length);
      for(var i=0;i<numtries;i++) {

          // pick random line parameters
          var alpha= randf(0, 2*Math.PI);
          var w1= Math.cos(alpha);
          var w2= Math.sin(alpha);

          // project data on this line and get the dot products
          for(var j=0;j<ix.length;j++) {
            dots[j]= w1*data[ix[j]][ri1] + w2*data[ix[j]][ri2];
          }

          // we are in a tricky situation because data dot product distribution
          // can be skewed. So we don't want to select just randomly between
          // min and max. But we also don't want to sort as that is too expensive
          // let's pick two random points and make the threshold be somewhere between them.
          // for skewed datasets, the selected points will with relatively high likelihood
          // be in the high-desnity regions, so the thresholds will make sense
          var ix1= ix[randi(0, N)];
          var ix2= ix[randi(0, N)];
          while(ix2==ix1) ix2= ix[randi(0, N)]; // enforce distinctness of ix2
          var a = Math.random();
          var dotthr= dots[ix1]*a + dots[ix2]*(1-a);

          var counts = zeros(this.numClasses * 2);
          let ltotal = 0, rtotal = 0;
          // measure information gain we'd get from split with thr
          for(var j=0;j<ix.length;j++) {
              if(dots[j] < dotthr) {
                counts[labels[j]]++;
                ltotal++;
              } else {
                counts[this.numClasses + labels[j]]++;
                rtotal++;
              }
          }
          let LH = 0, lp = 0, RH=0, rp=0;          
          for (var j = 0; j < this.numClasses; j++) {
            lp = counts[j] /= ltotal;
            LH -= lp*Math.log(lp);
          }
          
          for (var j = this.numClasses; j < counts.length; j++) {
            rp = counts[j] /= rtotal;
            RH -= rp*Math.log(rp);
          }          

          var informationGain= H - LH - RH;
          //console.log("Considering split %f, entropy %f -> %f, %f. Gain %f", thr, H, LH, RH, informationGain);
          if(informationGain > bestGain || i === 0) {
              bestGain= informationGain;
              bestw1= w1;
              bestw2= w2;
              bestthr= dotthr;
          }
      }

      model= {};
      model.w1= bestw1;
      model.w2= bestw2;
      model.dotthr= bestthr;
      return model;
  }

  // returns label for a single data instance
  function decision2DStumpTest(inst, model) {
      if(!model) {
          // this is a leaf that never received any data...
          return 1;
      }
      return inst[0]*model.w1 + inst[1]*model.w2 < model.dotthr ? 1 : -1;
  }

  // Misc utility functions
  function entropy(labels, ix, numClasses) {
      let counts=zeros(numClasses), total = 0;
      for(var i=0;i<ix.length;i++) {
          counts[labels[ix[i]]]++;
          total++;
      }
      total += 2;
      var h = 0;
      for(var i=0;i<counts.length;i++) {
        p = (counts[i] + 1)/ total;
        h -= p*Math.log(p);
      }
      return h;
}
  
function toProbs(list) {
      var total = list.reduce(add, 0);
      return list.map(function(a) { a / total });
  }

  function zeros(size) {
    var a = new Array(size);
    for (var i = 0; i < size;) a[i++] = 0;
    return a
  }


  function add(a,b) {
    return a + b;
  }
 
  // generate random floating point number between a and b
  function randf(a, b) {
      return Math.random()*(b-a)+a;
  }

  // generate random integer between a and b (b excluded)
  function randi(a, b) {
      return Math.floor(Math.random()*(b-a)+a);
  }

  // apparatus adapter

  util.inherits(RandomForestClassifier, Classifier);

  function restore(classifier) {
      classifier = Classifier.restore(classifier);
      classifier.__proto__ = RandomForestClassifier.prototype;

      // change prototypes recursively for the trees?

      return classifier;
  }

  function addExample(observation, classification) {

        if(observation.length > this.example_size) {
            var new_size = observation.length;
            this.example_size = new_size;
            for(var i = 0; i < this.examples.length; i++){
                var e = this.examples[i];
                for(var j=e.length; e<new_size; j++){
                    e.push(0.0);
                }
            }
        }
        this.examples.push(observation);
      // represent each label by an index
        if(typeof(classification) === "undefined") {
            throw "Error: label is empty";
        } else if(this.labelLookup.indexOf(classification) == -1) {
            this.labelLookup.push(classification);
        }
        var label_idx = this.labelLookup.indexOf(classification);
        this.labels.push(label_idx); // swap actual label for label index
  }

  function train() {
      this.numClasses = this.labelLookup.length;
      this.trainRandomForest(this.examples, this.labels, this.numClasses, this.options);
  }

  function getLabel(idx) {
    return this.labelLookup[idx];
  }

  function getClassifications(observation) {
      if(observation.length != this.example_size) {
        throw 'Observation should be of length ' + this.example_size;
      }
      var probs = this.predictOne(observation);

      var idxmax = 0;
      for (var i =0;i < probs.length; i++) {
        if(probs[i] > probs[idxmax]) idxmax = i;
      }
    return [ { value: probs[idxmax],
                label: this.getLabel(idxmax) }]
  }

  RandomForestClassifier.prototype.train = train;
  RandomForestClassifier.prototype.trainRandomForest = trainRandomForest;
  RandomForestClassifier.prototype.predictOne = predictOne;
  RandomForestClassifier.prototype.restore = restore;
  RandomForestClassifier.prototype.addExample = addExample;
  RandomForestClassifier.prototype.getClassifications = getClassifications;
  RandomForestClassifier.prototype.getLabel = getLabel;

  DecisionTree.prototype.train = trainDT;
  DecisionTree.prototype.predictOne = predictOneDT;

  RandomForestClassifier.DecisionTree = DecisionTree;
  RandomForestClassifier.decisionStumpTrain = decisionStumpTrain;
  RandomForestClassifier.decisionStumpTest = decisionStumpTest;
  RandomForestClassifier.decision2DStumpTrain = decision2DStumpTrain;
  RandomForestClassifier.decision2DStumpTest = decision2DStumpTest;

  module.exports = RandomForestClassifier;
